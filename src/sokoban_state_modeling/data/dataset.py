"""Random-access manifest Dataset for current VERL AgentLoop contracts."""

from __future__ import annotations

import asyncio
import copy
from pathlib import Path
from typing import Any

from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - importing domain modules stays lightweight
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass

from sokoban_state_modeling.engine import render_state
from sokoban_state_modeling.generator.manifest_store import ManifestStore
from sokoban_state_modeling.prompt.template import build_task_prompt, system_prompt
from sokoban_state_modeling.state.codec import grid_to_state

from .sampler import CoverageRef


def _config_get(config: Any, name: str, default: Any = None) -> Any:
    getter = getattr(config, "get", None)
    return getter(name, default) if callable(getter) else getattr(config, name, default)


class SokobanStateDataset(Dataset):
    """Render one immutable manifest row into a structured multimodal prompt."""

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: Any,
        config: Any,
        processor: Any = None,
        max_samples: int = -1,
    ):
        files = [data_files] if isinstance(data_files, (str, Path)) else list(data_files)
        if len(files) != 1:
            raise ValueError("SokobanStateDataset requires exactly one JSONL manifest per split")
        self.manifest_path = Path(files[0]).expanduser().resolve()
        index_value = _config_get(config, "manifest_index_path")
        if not index_value:
            split_index_fields = (
                ("train_files", "train_manifest_index_path"),
                ("val_files", "validation_manifest_index_path"),
            )
            for manifest_field, index_field in split_index_fields:
                configured_files = _config_get(config, manifest_field, []) or []
                if isinstance(configured_files, (str, Path)):
                    configured_files = [configured_files]
                resolved_files = {Path(value).expanduser().resolve() for value in configured_files}
                if self.manifest_path in resolved_files:
                    index_value = _config_get(config, index_field)
                    break
        self.index_path = Path(index_value).expanduser().resolve() if index_value else self.manifest_path.with_suffix(".idx")
        self.store = ManifestStore(self.manifest_path, self.index_path)
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.verify_rows = bool(_config_get(config, "verify_manifest_rows", True))
        self.reward_profile = str(_config_get(config, "reward_profile", "dense"))
        self.reasoning_mode = str(_config_get(config, "reasoning_mode", "non_thinking"))
        self.data_source = str(_config_get(config, "data_source", f"sokoban_state_modeling_{self.store.read(0, False)['split']}"))
        self.agent_name = str(_config_get(config, "agent_name", "single_turn_agent"))
        self.thinking_prompt_open = bool(_config_get(config, "thinking_prompt_open", self.reasoning_mode == "thinking"))
        if self.reward_profile not in {"dense", "exact_only"}:
            raise ValueError("reward_profile must be dense or exact_only")
        if self.reasoning_mode not in {"thinking", "non_thinking"}:
            raise ValueError("reasoning_mode must be thinking or non_thinking")
        if max_samples == 0 or max_samples < -1:
            raise ValueError("max_samples must be -1 or positive")
        self._length = len(self.store) if max_samples == -1 else min(len(self.store), int(max_samples))

    def __len__(self) -> int:
        return self._length

    @classmethod
    def _process_multi_modal_info(
        cls,
        messages: list[dict[str, Any]],
        image_patch_size: int,
        config: Any,
    ) -> tuple[list[Image.Image] | None, list[Any] | None, None]:
        """Extract visual inputs using the contract of the locked VERL Dataset.

        Imports stay lazy so data generation and reward-only tooling do not need
        the rollout runtime installed. This task has no video or audio inputs,
        but returning the three-tuple keeps it compatible with current VERL.
        """
        del config
        has_visual = any(
            isinstance(message.get("content"), list)
            and any(
                isinstance(item, dict) and item.get("type") in {"image", "video"}
                for item in message["content"]
            )
            for message in messages
        )
        if not has_visual:
            return None, None, None

        from qwen_vl_utils import process_vision_info

        images, videos = process_vision_info(
            messages,
            image_patch_size=image_patch_size,
            return_video_metadata=True,
        )
        return images, videos, None

    @classmethod
    async def process_multi_modal_info(
        cls,
        messages: list[dict[str, Any]],
        image_patch_size: int,
        config: Any,
    ) -> tuple[list[Image.Image] | None, list[Any] | None, None]:
        """Async entry point called by VERL ``AgentLoopBase`` workers."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: cls._process_multi_modal_info(
                messages,
                image_patch_size=image_patch_size,
                config=config,
            ),
        )

    @staticmethod
    def _unwrap_index(item: int | CoverageRef) -> tuple[int, dict[str, Any]]:
        if isinstance(item, CoverageRef):
            return item.task_index, item.to_dict()
        index = int(item)
        return index, {"task_index": index, "cycle": 0, "position": index, "is_padding": False}

    def build_raw_sample(self, item: int | CoverageRef) -> dict[str, Any]:
        """Build the non-tensor AgentLoop payload for data-only contract tests."""
        index, coverage = self._unwrap_index(item)
        if not 0 <= index < self._length:
            raise IndexError(index)
        task = self.store.read(index, verify=self.verify_rows)
        rows, cols = map(int, task["resolved_generator_config"]["dim_room"])
        current = grid_to_state(task["current_state"]["grid"])
        image = Image.fromarray(render_state(current), mode="RGB")
        prompt_with_placeholder = build_task_prompt(rows, cols, task["query_actions"], "<image>")
        prompt_text = prompt_with_placeholder.removeprefix("<image>\n")
        raw_prompt = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt()}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            },
        ]
        ground_truth = {
            "task_id": task["task_id"],
            "task_index": int(task["task_index"]),
            "current_state": copy.deepcopy(task["current_state"]),
            "next_state": copy.deepcopy(task["next_state"]),
            "query_actions": list(task["query_actions"]),
            "num_boxes": int(task["resolved_generator_config"]["num_boxes"]),
            "reward_profile": self.reward_profile,
            "reasoning_mode": self.reasoning_mode,
            "thinking_prompt_open": self.thinking_prompt_open,
        }
        extra_info = {
            "index": index,
            "task_id": task["task_id"],
            "task_index": int(task["task_index"]),
            "split": task["split"],
            "query_length": len(task["query_actions"]),
            "events": list(task["event_sequence"]),
            "size": f"{rows}x{cols}",
            "boxes": ground_truth["num_boxes"],
            "terminated": bool(task["terminated"]),
            "coverage": coverage,
        }
        return {
            "raw_prompt": raw_prompt,
            "data_source": self.data_source,
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": extra_info,
            "agent_name": self.agent_name,
            "index": index,
            "priority": index,
        }

    def __getitem__(self, item: int | CoverageRef) -> dict[str, Any]:
        if torch is None:
            raise RuntimeError("PyTorch is required to instantiate the VERL Dataset")
        sample = self.build_raw_sample(item)
        sample["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)
        return sample
