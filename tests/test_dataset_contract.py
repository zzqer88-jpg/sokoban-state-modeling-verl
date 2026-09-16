import asyncio
import json
from pathlib import Path
import sys
import types

from sokoban_state_modeling import GENERATOR_VERSION, RENDERER_VERSION, SCHEMA_VERSION
from sokoban_state_modeling.data import CoverageRef, SokobanStateDataset
from sokoban_state_modeling.generator.manifest_store import build_index
from sokoban_state_modeling.state.codec import grid_to_state, map_hash, state_hash, task_hash
from sokoban_state_modeling.state.transition import apply_actions


def _manifest(tmp_path: Path) -> tuple[Path, Path]:
    current = grid_to_state(["######", "#____#", "#_PXO#", "#____#", "#____#", "######"])
    nxt, events, subtypes, terminal = apply_actions(current, ["Right"])
    row = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "renderer_version": RENDERER_VERSION,
        "task_id": "test_00000001",
        "task_index": 0,
        "accepted_attempt": 0,
        "split": "test",
        "resolved_generator_config": {"dim_room": [6, 6], "num_boxes": 1},
        "map_hash": map_hash(current),
        "current_state_hash": state_hash(current),
        "next_state_hash": state_hash(nxt),
        "query_actions": ["Right"],
        "task_hash": task_hash(current, ["Right"]),
        "current_state": current.to_dict(),
        "next_state": nxt.to_dict(),
        "event_sequence": events,
        "event_subtype_sequence": subtypes,
        "executed_action_count": 1,
        "terminated": terminal,
    }
    manifest = tmp_path / "test.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    index = tmp_path / "test.idx"
    build_index(manifest, index)
    return manifest, index


def test_dataset_returns_agent_loop_schema_and_coverage_identity(tmp_path):
    manifest, index = _manifest(tmp_path)
    dataset = SokobanStateDataset(
        str(manifest),
        tokenizer=None,
        processor=None,
        config={
            "manifest_index_path": str(index),
            "reward_profile": "dense",
            "reasoning_mode": "non_thinking",
            "data_source": "sokoban_test",
        },
    )
    sample = dataset.build_raw_sample(CoverageRef(task_index=0, cycle=3, position=7, is_padding=True))
    assert set(sample) == {
        "raw_prompt", "data_source", "reward_model", "extra_info",
        "agent_name", "index", "priority",
    }
    assert sample["agent_name"] == "single_turn_agent"
    assert sample["extra_info"]["coverage"] == {
        "task_index": 0, "cycle": 3, "position": 7, "is_padding": True,
    }
    image_parts = [
        part for message in sample["raw_prompt"] for part in message["content"]
        if part["type"] == "image"
    ]
    assert len(image_parts) == 1 and image_parts[0]["image"].size == (96, 96)
    visible_text = "".join(
        part["text"] for message in sample["raw_prompt"] for part in message["content"]
        if part["type"] == "text"
    )
    assert "next_state" not in visible_text and "box_enters_target" not in visible_text


def test_dataset_selects_split_specific_index(tmp_path):
    manifest, index = _manifest(tmp_path)
    dataset = SokobanStateDataset(
        str(manifest),
        tokenizer=None,
        processor=None,
        config={
            "val_files": [str(manifest)],
            "validation_manifest_index_path": str(index),
        },
    )
    assert dataset.index_path == index.resolve()


def test_dataset_exposes_locked_verl_multimodal_entrypoint(monkeypatch):
    image = object()
    observed = {}

    def process_vision_info(messages, **kwargs):
        observed["messages"] = messages
        observed.update(kwargs)
        return [image], None

    monkeypatch.setitem(
        sys.modules,
        "qwen_vl_utils",
        types.SimpleNamespace(process_vision_info=process_vision_info),
    )
    messages = [{"role": "user", "content": [{"type": "image", "image": image}]}]
    result = asyncio.run(
        SokobanStateDataset.process_multi_modal_info(
            messages,
            image_patch_size=16,
            config={},
        )
    )
    assert result == ([image], None, None)
    assert observed == {
        "messages": messages,
        "image_patch_size": 16,
        "return_video_metadata": True,
    }


def test_dataset_multimodal_entrypoint_skips_dependency_for_text_only():
    messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    result = asyncio.run(
        SokobanStateDataset.process_multi_modal_info(
            messages,
            image_patch_size=14,
            config={},
        )
    )
    assert result == (None, None, None)
