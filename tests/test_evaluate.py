import json

import pytest

from sokoban_state_modeling.scripts.evaluate import load_unique_responses


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_response_loader_requires_exact_unique_coverage(tmp_path):
    path = tmp_path / "responses.jsonl"
    _write(path, [{"task_index": 1, "response": "b"}, {"task_index": 0, "response": "a"}])
    assert [row["task_index"] for row in load_unique_responses(path, 2)] == [0, 1]

    _write(path, [{"task_index": 0, "response": "a"}])
    with pytest.raises(ValueError, match="cover 1/2"):
        load_unique_responses(path, 2)
    assert len(load_unique_responses(path, 2, allow_partial=True)) == 1

    _write(path, [{"task_index": 0, "response": "a"}, {"task_index": 0, "response": "b"}])
    with pytest.raises(ValueError, match="duplicate"):
        load_unique_responses(path, 2, allow_partial=True)
