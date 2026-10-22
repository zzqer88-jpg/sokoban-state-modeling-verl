import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from sokoban_state_modeling.sft.perception import (
    build_perception_record,
    export_perception_sft,
    validate_perception_record,
)
from sokoban_state_modeling.sft.verl_dataset import prepare_thinking_sft_record, render_record_image


def source(grid=None, split="train"):
    return {"split": split, "current_state": {"grid": grid or [
        "######", "#____#", "#PXO_#", "#____#", "#____#", "######",
    ]}}


def write_source(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_exact_target_and_renderer_adapter():
    record = build_perception_record(source())
    assert record["reasoning"] == "Initial: player=(3,2); boxes=[(3,3)]; targets=[(3,4)]."
    assert record["target"] == record["reasoning"] + "</think>" + record["answer"]
    assert record["target"].count("</think>") == 1
    assert "<think>" not in record["target"]
    assert "prediction" not in json.dumps(record)
    assert "action" not in json.dumps(record)
    prepared = prepare_thinking_sft_record(record)
    assert [item["role"] for item in prepared["messages"]] == ["system", "user", "assistant"]
    assert len(record["messages"]) == 2
    assert prepared["messages"][-1]["content"] == record["target"]
    assert render_record_image(record).mode == "RGB"


def test_targets_under_player_and_box():
    record = build_perception_record(source(["######", "#+*X_#", "#____#", "#____#", "#____#", "######"]))
    assert "player=(2,2); boxes=[(2,3),(2,4)]; targets=[(2,2),(2,3)]" in record["reasoning"]
    assert "#+*X_#" in record["answer"]


@pytest.mark.parametrize("field", ["answer", "reasoning", "target", "enable_thinking", "messages"])
def test_tampering_rejected(field):
    record = build_perception_record(source())
    record[field] = "tampered"
    with pytest.raises(ValueError):
        prepare_thinking_sft_record(record)


def test_bad_state_rejected():
    with pytest.raises(ValueError, match="exactly one player"):
        build_perception_record(source(["#####", "#_XO#", "#####"]))
    row = source()
    row["current_state_hash"] = "wrong"
    with pytest.raises(ValueError, match="hash mismatch"):
        build_perception_record(row)


def test_export_deduplicates_and_drops_transition_fields(tmp_path):
    first = source()
    second = copy.deepcopy(first)
    second.update(query_actions=["Right"], next_state={"grid": ["invalid"]}, event_sequence=["irrelevant"])
    path = tmp_path / "source.jsonl"
    write_source(path, [first, second])
    output = tmp_path / "out.jsonl"
    report = export_perception_sft(path, output)
    assert report["source_records"] == 2
    assert report["exported_records"] == 1
    assert report["duplicates_removed"] == 1
    assert report["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    record = json.loads(output.read_text())
    validate_perception_record(record)
    assert "query_actions" not in record and "next_state" not in record
    other = tmp_path / "other.jsonl"
    export_perception_sft(path, other)
    assert output.read_bytes() == other.read_bytes()
    with pytest.raises(FileExistsError):
        export_perception_sft(path, output)


@pytest.mark.parametrize("rows", [[], [source(), source(split="test")]])
def test_invalid_export_not_published(tmp_path, rows):
    path, output = tmp_path / "source.jsonl", tmp_path / "out.jsonl"
    write_source(path, rows)
    with pytest.raises(ValueError):
        export_perception_sft(path, output)
    assert not output.exists()
    assert not output.with_suffix(".jsonl.audit.json").exists()


def test_cli(tmp_path):
    path, output = tmp_path / "source.jsonl", tmp_path / "out.jsonl"
    write_source(path, [source()])
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    result = subprocess.run([
        sys.executable, "-m", "sokoban_state_modeling.scripts.export_perception_sft",
        "--source", str(path), "--output", str(output),
    ], env=env, check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)["exported_records"] == 1
