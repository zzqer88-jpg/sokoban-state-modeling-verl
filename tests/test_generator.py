from collections import Counter
import json
import pytest

from sokoban_state_modeling.generator.audit import audit_manifest
from sokoban_state_modeling.generator.dataset_builder import DatasetBuilder
from sokoban_state_modeling.generator.sampler import MAIN_V1, QuotaSchedule


def test_joint_quota_schedule_is_exact():
    schedule = QuotaSchedule(100, MAIN_V1, 123)
    rows = [schedule.assignment(i) for i in range(100)]
    assert Counter(row["dim_room"] for row in rows) == {(6, 6): 60, (7, 7): 25, (8, 8): 15}
    assert Counter(row["num_boxes"] for row in rows) == {1: 90, 2: 10}
    assert Counter(row["player_distance"] for row in rows) == {"adjacent": 30, "near": 30, "medium": 25, "far": 15}
    assert sum(row["terminal"] for row in rows) == 10
    assert all(row["num_boxes"] == 2 for row in rows if row["primary_state_category"] == "box_on_target")
    assert all(row["num_boxes"] == 2 for row in rows if row["primary_state_category"] == "player_on_target")
    assert all(row["dim_room"] == (8, 8) for row in rows if row["player_distance"] == "far")
    assert all(row["action_length"] == 3 for row in rows if row["primary_state_category"] == "box_on_target")
    assert all(row["player_distance"] == "near" for row in rows if row["primary_state_category"] == "box_on_target")
    assert all(row["player_distance"] == "near" for row in rows if row["primary_state_category"] == "player_on_target")
    narrow = [row for row in rows if row["primary_state_category"] == "narrow_or_high_wall"]
    assert all(row["topology_multiplier"] == 0.75 for row in narrow)
    assert all(row["p_change_directions"] == 0.15 for row in narrow)
    assert all(row["reverse_progress"] == "near_goal" for row in narrow)
    far = [row for row in rows if row["player_distance"] == "far"]
    assert all(row["topology_multiplier"] in {1.25, 1.5} for row in far)
    assert all(row["p_change_directions"] == 0.35 for row in far)
    assert all(row["player_distance"] == "adjacent" for row in rows if row["primary_state_category"] == "player_near_box")
    assert all(row["primary_state_category"] == "near_completion" for row in rows if row["terminal"])


def test_eval_dimensions_are_equal_strata():
    validation = QuotaSchedule(2000, MAIN_V1, 3, split="validation")
    assert Counter(validation.assignment(i)["dim_room"] for i in range(2000)) == {
        (6, 6): 667, (7, 7): 667, (8, 8): 666,
    }
    rows = [validation.assignment(i) for i in range(2000)]
    assert all(row["player_distance"] == "near" for row in rows if row["primary_state_category"] == "player_on_target")
    test = QuotaSchedule(10_000, MAIN_V1, 4, split="test")
    assert Counter(test.assignment(i)["dim_room"] for i in range(10_000)) == {
        (6, 6): 3_334, (7, 7): 3_333, (8, 8): 3_333,
    }


def test_generation_resumes_only_committed_chunks(tmp_path):
    interrupted = DatasetBuilder(tmp_path, "train", 20, chunk_size=5)
    original = interrupted._make_row

    def fail_in_second_chunk(index):
        if index == 7:
            raise RuntimeError("simulated interruption")
        return original(index)

    interrupted._make_row = fail_in_second_chunk
    with pytest.raises(RuntimeError, match="simulated interruption"):
        interrupted.build()
    chunks = list((tmp_path / "generation_chunks" / "train").glob("chunk_*.jsonl"))
    assert len(chunks) == 1
    assert len(chunks[0].read_text(encoding="utf-8").splitlines()) == 5

    manifest, index = DatasetBuilder(tmp_path, "train", 20, chunk_size=5).build(resume=True)
    report = audit_manifest(manifest, index)
    assert report["passed"] and report["num_tasks"] == 20
    assert report["provenance"]["bit_generator"] == "PCG64"
    first_row = json.loads(manifest.read_text(encoding="utf-8").splitlines()[0])
    assert "image_path" not in first_row and "image_sha256" not in first_row
    assert not list(tmp_path.rglob("*.png"))
    assert not (tmp_path / "generation_chunks").exists()

    clean_manifest, _ = DatasetBuilder(tmp_path / "clean", "train", 20, chunk_size=5).build()
    assert manifest.read_bytes() == clean_manifest.read_bytes()
