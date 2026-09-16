#!/usr/bin/env python3
"""Run one deterministic 5x5 Sokoban vision/state-transition probe."""

from __future__ import annotations

import argparse
import base64
from contextlib import nullcontext
from io import BytesIO
import json
from pathlib import Path

from PIL import Image

from sokoban_state_modeling.engine import render_state
from sokoban_state_modeling.prompt.template import build_task_prompt, system_prompt
from sokoban_state_modeling.reward.reasoning_adapter import (
    decode_preserving_reasoning_markers,
    split_reasoning,
)
from sokoban_state_modeling.reward.reward import score_response
from sokoban_state_modeling.state.codec import grid_to_state


CURRENT_GRID = (
    "#####",
    "#_O_#",
    "#_X_#",
    "#P__#",
    "#####",
)
NEXT_GRID = (
    "#####",
    "#_O_#",
    "#PX_#",
    "#___#",
    "#####",
)
ACTIONS = ("UP",)


def _image_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_messages(image: Image.Image) -> list[dict]:
    prompt = build_task_prompt(5, 5, ACTIONS, "<image>").removeprefix("<image>\n")
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt()}]},
        {
            "role": "user",
            "content": [
                # A data URL follows the official multimodal chat schema and avoids
                # version-specific PIL-object handling in apply_chat_template().
                {"type": "image", "url": _image_data_url(image)},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def _load_model(model_path: str, trust_remote_code: bool, attn_implementation: str | None):
    try:
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        model_cls = AutoModelForMultimodalLM
    except ImportError:
        try:
            from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

            model_cls = Qwen3_5ForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "This Transformers build has no Qwen3.5 multimodal model class. "
                "Install the current Transformers main branch as documented by Qwen."
            ) from exc

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )
    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": trust_remote_code,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    model = model_cls.from_pretrained(model_path, **model_kwargs)
    model.eval()
    return model, processor


def _report(answer: str) -> dict:
    current = grid_to_state(CURRENT_GRID)
    next_state = grid_to_state(NEXT_GRID)
    exact = score_response(answer, current, next_state, list(ACTIONS), 1, "exact_only")
    dense = score_response(answer, current, next_state, list(ACTIONS), 1, "dense")
    return {
        "format_valid": exact.parsed.format_valid,
        "format_exact": exact.parsed.format_exact,
        "perception_exact": bool(exact.components["perception_exact"]),
        "prediction_exact": bool(exact.components["prediction_exact"]),
        "joint_exact": bool(exact.components["exact_all"]),
        "exact_only_reward": exact.reward,
        "dense_reward": dense.reward,
        "delta_score": dense.components["delta_score"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test Qwen3.5-4B on one 5x5 Sokoban image and one UP action",
    )
    parser.add_argument("--model-path", default="Qwen/Qwen3.5-4B")
    parser.add_argument(
        "--mode",
        choices=("thinking", "non_thinking"),
        default="non_thinking",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/small_map_probe"),
    )
    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")

    current = grid_to_state(CURRENT_GRID)
    image = Image.fromarray(render_state(current), mode="RGB")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_path = args.output_dir / "5x5_probe.png"
    image.save(image_path)

    model, processor = _load_model(
        args.model_path,
        args.trust_remote_code,
        args.attn_implementation,
    )
    messages = build_messages(image)
    prompt_open = args.mode == "thinking"
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=prompt_open,
    ).to(model.device)

    try:
        import torch

        inference_context = torch.inference_mode()
    except ImportError:  # pragma: no cover - model loading already requires torch
        inference_context = nullcontext()

    with inference_context:
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    generated_ids = output_ids[0, inputs["input_ids"].shape[-1] :].tolist()
    tokenizer = getattr(processor, "tokenizer", processor)
    raw_response = decode_preserving_reasoning_markers(tokenizer, generated_ids)
    separated = split_reasoning(raw_response, args.mode, prompt_open=prompt_open)
    answer = separated.answer_payload

    report = {
        "model_path": args.model_path,
        "mode": args.mode,
        "image": str(image_path.resolve()),
        "actions": list(ACTIONS),
        "reasoning_boundary_valid": separated.boundary_valid,
        "reasoning_present": separated.reasoning_present,
        **_report(answer),
    }
    report_path = args.output_dir / f"result_{args.mode}.json"
    report_path.write_text(
        json.dumps(
            {
                **report,
                "raw_response": raw_response,
                "answer_payload": answer,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=== raw response ===")
    print(raw_response)
    print("=== answer payload ===")
    print(answer)
    print("=== automatic check ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"full result: {report_path.resolve()}")


if __name__ == "__main__":
    main()
