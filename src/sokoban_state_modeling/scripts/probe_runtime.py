#!/usr/bin/env python3
"""CPU-side contract probe for the pinned VERL and Qwen runtime."""

from __future__ import annotations

import argparse
import asyncio
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform

from sokoban_state_modeling.compat.verl_contract import verify_source_lock
from sokoban_state_modeling.reward.reasoning_adapter import (
    decode_preserving_reasoning_markers,
    split_reasoning,
)
from sokoban_state_modeling.training.main_ppo import compose_project_config


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verl-checkout", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, required=True)
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH"), required=False)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not args.model_path:
        parser.error("--model-path or MODEL_PATH is required")

    checkout = args.verl_checkout.expanduser().resolve()
    lock_path = PROJECT_ROOT / "constraints" / "verl.lock.json"
    source = verify_source_lock(checkout, lock_path)
    config = compose_project_config(
        checkout / "verl" / "trainer" / "config",
        "ppo_trainer",
        args.project_config.expanduser().resolve(),
    )

    from transformers import AutoProcessor
    from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
    from verl.utils.dataset.rl_dataset import collate_fn, get_dataset_class
    from verl.utils.import_utils import load_extern_object

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", processor)
    dataset_cls = get_dataset_class(config.data)
    train_dataset = dataset_cls(
        config.data.train_files,
        tokenizer=tokenizer,
        config=config.data,
        processor=processor,
        max_samples=1,
    )
    val_dataset = dataset_cls(
        config.data.val_files,
        tokenizer=tokenizer,
        config=config.data,
        processor=processor,
        max_samples=1,
    )
    sample = train_dataset[0]
    collated = collate_fn([sample])
    images, videos, audios = asyncio.run(
        dataset_cls.process_multi_modal_info(
            sample["raw_prompt"],
            image_patch_size=int(config.data.image_patch_size),
            config=config.data,
        )
    )
    if not images or videos is not None or audios is not None:
        raise RuntimeError("Sokoban multimodal extraction returned an unexpected schema")
    if len(val_dataset) != 1 or "raw_prompt" not in collated:
        raise RuntimeError("custom Dataset/collate contract failed")
    reward_manager_cls = load_extern_object(
        config.reward.reward_manager.module.path,
        config.reward.reward_manager.name,
    )
    if not issubclass(reward_manager_cls, RewardManagerBase):
        raise RuntimeError("configured reward manager does not implement RewardManagerBase")

    rendered = processor.apply_chat_template(
        sample["raw_prompt"],
        tokenize=False,
        add_generation_prompt=True,
        **dict(config.data.apply_chat_template_kwargs),
    )
    prompt_open = bool(config.data.thinking_prompt_open)
    observed_prompt_open = rendered.rstrip().endswith("<think>")
    if observed_prompt_open != prompt_open:
        raise RuntimeError(
            "chat-template thinking boundary disagrees with data.thinking_prompt_open: "
            f"expected={prompt_open}, observed={observed_prompt_open}"
        )

    response = "probe</think>answer" if prompt_open else "answer"
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    decoded = decode_preserving_reasoning_markers(tokenizer, response_ids)
    separated = split_reasoning(
        decoded,
        str(config.data.reasoning_mode),
        prompt_open=prompt_open,
    )
    if not separated.boundary_valid or separated.answer_payload != "answer":
        raise RuntimeError(f"reward thinking-boundary probe failed: {decoded!r}")

    report = {
        "passed": True,
        "verl": source,
        "model_path": args.model_path,
        "algorithm": str(config.algorithm.adv_estimator),
        "reasoning_mode": str(config.data.reasoning_mode),
        "thinking_prompt_open": prompt_open,
        "multimodal_images": len(images),
        "versions": {
            "python": platform.python_version(),
            **{
                name: _version(name)
                for name in (
                "torch",
                "transformers",
                "hydra-core",
                "omegaconf",
                "qwen-vl-utils",
                "vllm",
                "sglang",
                )
            },
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
