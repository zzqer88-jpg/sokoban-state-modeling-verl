"""VERL MultiTurnSFTDataset adapter for renderer-backed Sokoban JSONL.

The exported SFT records intentionally keep symbolic states instead of copied
image files.  This adapter renders the image lazily with the same renderer used
by RL and delegates tokenization, multimodal processing, loss masking, position
IDs, padding, and collation to the VERL checkout pinned by this project.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sokoban_state_modeling.engine import render_state
from sokoban_state_modeling.state.codec import grid_to_state

from sokoban_state_modeling.sft.thinking_trace import SFT_SCHEMA_VERSION

try:  # VERL is deliberately an optional, server-side dependency.
    from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
except ImportError:  # pragma: no cover - exercised only without the training checkout
    class MultiTurnSFTDataset:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(
                "SokobanThinkingSFTDataset requires the pinned VERL checkout on PYTHONPATH"
            )


def prepare_thinking_sft_record(record: dict[str, Any]) -> dict[str, Any]:
    """Validate one exported row and append its supervised assistant turn."""
    if record.get("sft_schema_version") != SFT_SCHEMA_VERSION:
        raise ValueError(f"unsupported SFT schema: {record.get('sft_schema_version')!r}")
    messages = record.get("messages")
    if not isinstance(messages, list) or [item.get("role") for item in messages] != ["system", "user"]:
        raise ValueError("SFT messages must contain exactly one system and one user turn")
    if sum(str(item.get("content", "")).count("<image>") for item in messages) != 1:
        raise ValueError("SFT prompt must contain exactly one <image> placeholder")
    if record.get("enable_thinking") is not True or record.get("thinking_prompt_open") is not True:
        raise ValueError("SFT record must use prompt-open thinking")
    target = record.get("target")
    if not isinstance(target, str) or target.count("</think>") != 1 or "<think>" in target:
        raise ValueError("SFT target must contain one closing thinking tag and no opening tag")
    state = record.get("current_state")
    if not isinstance(state, dict) or not isinstance(state.get("grid"), list):
        raise ValueError("SFT record is missing current_state.grid")

    prepared = dict(record)
    prepared["messages"] = copy.deepcopy(messages) + [{"role": "assistant", "content": target}]
    return prepared


def render_record_image(record: dict[str, Any]) -> Image.Image:
    """Render a prepared or raw SFT record with the canonical RGB renderer."""
    state = grid_to_state(record["current_state"]["grid"])
    return Image.fromarray(render_state(state), mode="RGB")


class SokobanThinkingSFTDataset(MultiTurnSFTDataset):
    """Read exported JSONL and feed renderer-backed conversations to VERL SFT."""

    def _download(self) -> None:
        resolved: list[str] = []
        for data_file in self.parquet_files:
            path = Path(str(data_file)).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            resolved.append(str(path))
        self.parquet_files = resolved

    def _read_files_and_process(self) -> None:
        import pandas as pd
        from verl.utils.tokenizer.chat_template import extract_system_prompt_and_generation

        records: list[dict[str, Any]] = []
        for data_file in self.parquet_files:
            with Path(data_file).open("r", encoding="utf-8") as source:
                for line_number, line in enumerate(source, start=1):
                    if not line.strip():
                        continue
                    try:
                        raw = json.loads(line)
                        records.append(prepare_thinking_sft_record(raw))
                    except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
                        raise ValueError(f"invalid SFT record at {data_file}:{line_number}: {exc}") from exc
        if not records:
            raise ValueError("SFT dataset is empty")

        total = len(records)
        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rng = np.random.default_rng(self.seed)
                indices = rng.choice(total, size=self.max_samples, replace=False).tolist()
            else:
                indices = list(range(self.max_samples))
            records = [records[index] for index in indices]

        self.dataframe = pd.DataFrame.from_records(records)
        self.messages = self.dataframe[self.messages_key].tolist()
        self.tools = self.dataframe[self.tools_key].tolist() if self.tools_key in self.dataframe.columns else None
        if self.enable_thinking_key in self.dataframe.columns:
            self.enable_thinking = self.dataframe[self.enable_thinking_key].tolist()
        else:
            self.enable_thinking = None
        self.system_prompt, self.generation_prompt = extract_system_prompt_and_generation(
            self.tokenizer, **self.apply_chat_template_kwargs
        )

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        row = dict(example)
        row[self.image_key] = [render_record_image(row)]
        return super()._build_messages(row)
