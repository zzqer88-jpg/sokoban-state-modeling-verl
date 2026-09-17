import json
from pathlib import Path

import pytest

from sokoban_state_modeling import GENERATOR_VERSION, RENDERER_VERSION, SCHEMA_VERSION
from sokoban_state_modeling.generator.manifest_store import build_index
from sokoban_state_modeling.reward.reasoning_adapter import split_reasoning
from sokoban_state_modeling.reward.reward import score_response
from sokoban_state_modeling.sft.thinking_export import export_thinking_sft
from sokoban_state_modeling.sft.thinking_trace import (
    SFT_SCHEMA_VERSION,
    build_reasoning_trace,
    build_thinking_sft_record,
)
from sokoban_state_modeling.sft.verl_dataset import prepare_thinking_sft_record, render_record_image
from sokoban_state_modeling.scripts.preflight_sft import FORMAL_COUNTS, inspect_export
from sokoban_state_modeling.state.codec import grid_to_state, map_hash, state_hash, task_hash
from sokoban_state_modeling.state.transition import apply_actions


CASES = (
    ("player_move", ("#####", "#P__#", "#_XO#", "#___#", "#####"), "Right"),
    ("wall_noop", ("#####", "#P__#", "#_XO#", "#___#", "#####"), "Up"),
    ("blocked_box_noop", ("#####", "#PX##", "#__O#", "#___#", "#####"), "Right"),
    ("box_push", ("#####", "#PX_#", "#__O#", "#___#", "#####"), "Right"),
    ("box_enters_target", ("#####", "#PXO#", "#___#", "#___#", "#####"), "Right"),
    ("box_leaves_target", ("#####", "#P*_#", "#___#", "#___#", "#####"), "Right"),
    ("player_target_transition", ("#####", "#PO_#", "#X__#", "#___#", "#####"), "Right"),
    ("player_target_transition", ("#####", "#+__#", "#X__#", "#___#", "#####"), "Right"),
)


def _task(rows, actions=("Right",), task_index=0):
    current = grid_to_state(rows)
    nxt, events, subtypes, terminal = apply_actions(current, actions)
    num_boxes = sum(row.count("X") + row.count("*") for row in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "renderer_version": RENDERER_VERSION,
        "task_id": f"train_{task_index + 1:08d}",
        "task_index": task_index,
        "accepted_attempt": 0,
        "split": "train",
        "resolved_generator_config": {"dim_room": list(current.shape), "num_boxes": num_boxes},
        "map_hash": map_hash(current),
        "current_state_hash": state_hash(current),
        "next_state_hash": state_hash(nxt),
        "query_actions": list(actions),
        "task_hash": task_hash(current, actions),
        "current_state": current.to_dict(),
        "next_state": nxt.to_dict(),
        "event_sequence": events,
        "event_subtype_sequence": subtypes,
        "executed_action_count": len(actions),
        "terminated": terminal,
    }


@pytest.mark.parametrize(("expected_event", "rows", "action"), CASES)
def test_reasoning_trace_covers_every_transition_event(expected_event, rows, action):
    trace = build_reasoning_trace(grid_to_state(rows), [action])
    assert trace.events == (expected_event,)
    assert f"Given action 1/1 = {action}:" in trace.reasoning
    assert f"event={expected_event}" in trace.reasoning
    assert "Initial: player=" in trace.reasoning
    assert "Final: player=" in trace.reasoning


def test_multistep_trace_uses_the_updated_state():
    rows = ("#####", "#X_O#", "#PXO#", "#___#", "#####")
    trace = build_reasoning_trace(grid_to_state(rows), ["Right", "Left"])
    assert trace.events == ("box_enters_target", "player_move")
    assert "Given action 1/2 = Right:" in trace.reasoning
    assert "Given action 2/2 = Left: front=(3,2)=floor" in trace.reasoning
    assert "Final: player=(3,2); boxes=[(2,2),(3,4)]; solved=false." in trace.reasoning


def test_sft_record_matches_prompt_open_thinking_and_exact_answer():
    task = _task(("#####", "#___#", "#PXO#", "#___#", "#####"))
    record = build_thinking_sft_record(task)
    assert record["sft_schema_version"] == SFT_SCHEMA_VERSION
    assert record["target"].count("<think>") == 0
    assert record["target"].count("</think>") == 1
    separated = split_reasoning(record["target"], "thinking", prompt_open=True)
    assert separated.boundary_valid
    assert separated.reasoning_content == record["reasoning"]
    assert separated.answer_payload == record["answer"]
    current = grid_to_state(task["current_state"]["grid"])
    nxt = grid_to_state(task["next_state"]["grid"])
    assert score_response(record["answer"], current, nxt, ["Right"], 1, "exact_only").reward == 1.0


def test_verl_sft_adapter_appends_assistant_without_mutating_export():
    record = build_thinking_sft_record(_task(("#####", "#___#", "#PXO#", "#___#", "#####")))
    prepared = prepare_thinking_sft_record(record)
    assert [message["role"] for message in record["messages"]] == ["system", "user"]
    assert [message["role"] for message in prepared["messages"]] == ["system", "user", "assistant"]
    assert prepared["messages"][-1]["content"] == record["target"]
    image = render_record_image(prepared)
    assert image.mode == "RGB"
    assert image.width > 0 and image.height > 0


def test_verl_sft_adapter_rejects_a_target_that_reopens_thinking():
    record = build_thinking_sft_record(_task(("#####", "#___#", "#PXO#", "#___#", "#####")))
    record["target"] = "<think>bad</think>" + record["answer"]
    with pytest.raises(ValueError, match="one closing thinking tag and no opening tag"):
        prepare_thinking_sft_record(record)


def test_sft_preflight_inspects_exported_jsonl(tmp_path: Path):
    record = build_thinking_sft_record(_task(("#####", "#___#", "#PXO#", "#___#", "#####")))
    export = tmp_path / "train.thinking.jsonl"
    export.write_text(json.dumps(record) + "\n", encoding="utf-8")
    report = inspect_export(export, "train")
    assert report["records"] == 1
    assert report["sizes"] == {"5x5": 1}
    assert report["query_lengths"] == {"1": 1}
    assert report["boxes"] == {"1": 1}
    assert FORMAL_COUNTS["train"]["records"] == 12_000
    assert FORMAL_COUNTS["validation"]["records"] == 1_200


@pytest.mark.parametrize("field", ("next_state", "event_sequence", "event_subtype_sequence", "terminated"))
def test_sft_record_rejects_tampered_manifest_truth(field):
    task = _task(("#####", "#___#", "#PXO#", "#___#", "#####"))
    if field == "next_state":
        task[field] = task["current_state"]
    elif field == "event_sequence":
        task[field] = ["box_push"]
    elif field == "event_subtype_sequence":
        task[field] = ["enters_target"]
    else:
        task[field] = not task[field]
    with pytest.raises(ValueError, match="disagrees with action replay"):
        build_thinking_sft_record(task)


def test_export_is_deterministic_and_audited(tmp_path: Path):
    tasks = [
        _task(("#####", "#P__#", "#_XO#", "#___#", "#####"), ("Right",), 0),
        _task(("#####", "#___#", "#PXO#", "#___#", "#####"), ("Right",), 1),
    ]
    manifest = tmp_path / "train.jsonl"
    manifest.write_text("".join(json.dumps(task) + "\n" for task in tasks), encoding="utf-8")
    index = tmp_path / "train.idx"
    build_index(manifest, index)
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    first_report = export_thinking_sft(manifest, first, index_path=index)
    second_report = export_thinking_sft(manifest, second, index_path=index)
    assert first.read_bytes() == second.read_bytes()
    assert first_report["passed"] and second_report["passed"]
    assert first_report["exported_records"] == 2
    assert first_report["event_counts"] == {"box_enters_target": 1, "player_move": 1}
    records = [json.loads(line) for line in first.read_text(encoding="utf-8").splitlines()]
    assert [record["task_index"] for record in records] == [0, 1]
    assert first.with_suffix(".jsonl.audit.json").exists()


def test_export_refuses_to_overwrite(tmp_path: Path):
    task = _task(("#####", "#P__#", "#_XO#", "#___#", "#####"))
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(json.dumps(task) + "\n", encoding="utf-8")
    index = tmp_path / "train.idx"
    build_index(manifest, index)
    output = tmp_path / "sft.jsonl"
    export_thinking_sft(manifest, output, index_path=index)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        export_thinking_sft(manifest, output, index_path=index)
