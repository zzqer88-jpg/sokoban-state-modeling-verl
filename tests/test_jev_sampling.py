import json
from pathlib import Path

from sokoban_state_modeling.jev.dataset import BoardDataset
from sokoban_state_modeling.jev.models import SAMPLED_SCHEMA
from sokoban_state_modeling.jev.sampling import SAMPLER_VERSION, select_training_cells
from sokoban_state_modeling.sft.jev_export import build_record, export_jev
from sokoban_state_modeling.state.codec import grid_to_state


TRAIN_GRIDS = [
    ["########", "#P_____#", "#__#___#", "#___X__#", "#___O__#", "#______#", "#______#", "########"],
    ["########", "#_P____#", "#__#___#", "#___X__#", "#___O__#", "#______#", "#______#", "########"],
    ["########", "#__P___#", "#__#___#", "#___X__#", "#___O__#", "#______#", "#______#", "########"],
]


def source(grid, split="train", task_id="sample"):
    return {"current_state": {"grid": grid}, "split": split, "task_id": task_id}


def test_semantic_sampler_is_deterministic_and_covers_key_cell_types():
    state = grid_to_state(TRAIN_GRIDS[0])
    first = select_training_cells(state, budget=16)
    assert first == select_training_cells(state, budget=16)
    assert len(first) == len(set(first)) == 16
    assert first == tuple(sorted(first))
    assert {(1, 1), (3, 4), (4, 4)} <= set(first)
    assert any(state.fixed[row, column] == 0 and
               (row in {0, 7} or column in {0, 7}) for row, column in first)
    assert any(state.fixed[row, column] == 0 and row not in {0, 7}
               and column not in {0, 7} for row, column in first)


def test_sampled_record_has_metadata_and_sixteen_questions():
    record, _ = build_record(source(TRAIN_GRIDS[0]), cell_budget=16)
    assert record["schema_version"] == SAMPLED_SCHEMA
    assert len(record["input"]["questions"]) == 16
    assert len(record["target"]["answers"]) == 16
    assert record["input"]["sampling"]["version"] == SAMPLER_VERSION
    assert record["input"]["sampling"]["budget"] == 16


def test_export_caps_unique_train_boards_and_keeps_validation_full(tmp_path: Path):
    validation_grid = ["#######", "#P____#", "#_X_O_#", "#_____#", "#######"]
    rows = [source(grid, task_id=f"train-{index}") for index, grid in enumerate(TRAIN_GRIDS)]
    rows.append(source(validation_grid, split="validation", task_id="validation"))
    input_path = tmp_path / "source.jsonl"
    input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    output = tmp_path / "export"
    report = export_jev([input_path], output, train_cell_budget=16, max_train_boards=2)

    assert report["split_counts"] == {"train": 2, "validation": 1}
    assert report["train_sampling"]["available_unique_boards"] == 3
    assert report["train_sampling"]["selected_unique_boards"] == 2
    records = [json.loads(line) for line in (output / "records.jsonl").read_text().splitlines()]
    train = [record for record in records if record["split"] == "train"]
    validation = [record for record in records if record["split"] == "validation"]
    assert all(len(record["input"]["questions"]) == 16 for record in train)
    assert len(validation[0]["input"]["questions"]) == 35

    training_data = BoardDataset(output / "records.jsonl", "train")
    validation_data = BoardDataset(output / "records.jsonl", "validation")
    assert len(training_data) == 2
    assert len(training_data[0].inputs.questions) == 16
    assert len(validation_data[0].inputs.questions) == 35
