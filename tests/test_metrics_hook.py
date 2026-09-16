from sokoban_state_modeling.reward.verl_reward import compute_score
from sokoban_state_modeling.state.codec import grid_to_state, state_to_grid
from sokoban_state_modeling.state.transition import apply_actions
from sokoban_state_modeling.reward.verl_reward import SLICE_INDICATOR_LABELS
from sokoban_state_modeling.training.metrics_hook import (
    MetricAccumulator,
    aggregate_reward_columns,
    reward_columns_from_extra_fields,
)


def _ground_truth():
    current = grid_to_state(["######", "#____#", "#_PXO#", "#____#", "#____#", "######"])
    nxt, _, _, _ = apply_actions(current, ["Right"])
    return current, nxt, {
        "current_state": current.to_dict(),
        "next_state": nxt.to_dict(),
        "query_actions": ["Right"],
        "num_boxes": 1,
        "reward_profile": "dense",
        "reasoning_mode": "non_thinking",
    }


def _response(a, b):
    return "<perception>\n" + "\n".join(state_to_grid(a)) + "\n</perception>\n<prediction>\n" + "\n".join(state_to_grid(b)) + "\n</prediction>"


def test_accumulator_uses_ratio_of_global_sums_and_slices():
    current, nxt, truth = _ground_truth()
    perfect = compute_score("test", _response(current, nxt), truth)
    wrong = compute_score("test", _response(current, current), truth)
    labels = {"query_length": 1, "size": "6x6", "boxes": 1, "terminated": True, "events": ["box_enters_target"]}
    left = MetricAccumulator.from_rows([(perfect, labels)])
    right = MetricAccumulator.from_rows([(wrong, labels)])
    left.merge(right)
    result = left.result()
    assert result["overall"]["conditional_prediction_success_numerator"] == 1.0
    assert result["overall"]["conditional_prediction_success_denominator"] == 2.0
    assert result["overall"]["conditional_prediction_success_rate"] == 0.5
    assert result["slices"]["event=box_enters_target"]["samples"] == 2


def test_accumulator_rejects_sparse_reward_keysets():
    accumulator = MetricAccumulator()
    try:
        accumulator.update({"score": 1.0}, {"query_length": 1, "size": "6x6", "boxes": 1, "terminated": False})
    except ValueError as exc:
        assert "missing fixed keys" in str(exc)
    else:
        raise AssertionError("sparse reward metrics were accepted")


def test_reward_uses_fixed_slice_schema_and_no_per_sample_conditional_ratio():
    current, nxt, truth = _ground_truth()
    labels = {
        "query_length": 1,
        "size": "6x6",
        "boxes": 1,
        "terminated": True,
        "events": ["box_enters_target"],
        "coverage": {"task_index": 7, "cycle": 2, "position": 8, "is_padding": True},
    }
    result = compute_score("test", _response(current, nxt), truth, labels)
    assert "conditional_prediction_success_rate" not in result
    assert set(SLICE_INDICATOR_LABELS) <= result.keys()
    assert result["slice_event_box_enters_target"] == 1.0
    assert result["slice_event_wall_noop"] == 0.0
    assert result["coverage_is_padding"] == 1.0


def test_column_aggregation_derives_ratio_and_slice_metrics():
    current, nxt, truth = _ground_truth()
    labels = {
        "query_length": 1,
        "size": "6x6",
        "boxes": 1,
        "terminated": True,
        "events": ["box_enters_target"],
        "coverage": {"task_index": 0, "cycle": 0, "position": 0, "is_padding": False},
    }
    rows = [
        compute_score("test", _response(current, nxt), truth, labels),
        compute_score("test", _response(current, current), truth, labels),
    ]
    columns = {key: [row[key] for row in rows] for key in rows[0]}
    values = aggregate_reward_columns(columns, "train/sokoban")
    assert values["train/sokoban/overall/conditional_prediction_success_rate"] == 0.5
    assert values["train/sokoban/slice/k/1/conditional_prediction_success_rate"] == 0.5
    assert values["train/sokoban/coverage/unique_in_batch"] == 2


def test_transfer_queue_extra_fields_filter_padding_before_domain_aggregation():
    current, nxt, truth = _ground_truth()
    base = {
        "query_length": 1,
        "size": "6x6",
        "boxes": 1,
        "terminated": True,
        "events": ["box_enters_target"],
    }
    unique = compute_score(
        "test",
        _response(current, nxt),
        truth,
        base | {"coverage": {"task_index": 0, "cycle": 0, "position": 0, "is_padding": False}},
    )
    padding = compute_score(
        "test",
        _response(current, current),
        truth,
        base | {"coverage": {"task_index": 0, "cycle": 0, "position": 1, "is_padding": True}},
    )
    fields = [{"reward_extra_info": unique}, {"reward_extra_info": padding}]
    columns = reward_columns_from_extra_fields(fields, [True, False])
    values = aggregate_reward_columns(columns, "train-sokoban")
    assert values["train-sokoban/overall/samples"] == 1
    assert values["train-sokoban/overall/prediction_success_rate"] == 1.0
