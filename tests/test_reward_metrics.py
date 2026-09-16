import math

from sokoban_state_modeling.reward.metrics import aggregate_metrics
from sokoban_state_modeling.reward.reward import score_response
from sokoban_state_modeling.state.codec import grid_to_state, state_to_grid
from sokoban_state_modeling.state.transition import apply_actions


def format_response(a, b):
    return "<perception>\n" + "\n".join(state_to_grid(a)) + "\n</perception>\n<prediction>\n" + "\n".join(state_to_grid(b)) + "\n</prediction>"


def test_golden_response_scores_one_in_both_profiles():
    current = grid_to_state(["######", "#____#", "#_PXO#", "#____#", "#____#", "######"])
    next_state, _, _, _ = apply_actions(current, ["Right"])
    text = format_response(current, next_state)
    for profile in ("dense", "exact_only"):
        result = score_response(text, current, next_state, ["Right"], 1, profile)
        assert result.reward == 1.0
        assert result.metrics["joint_success_rate"] == 1.0
        assert result.metrics["format_exact_rate"] == 1.0


def test_exact_only_keeps_three_components_separate():
    current = grid_to_state(["######", "#____#", "#_PXO#", "#____#", "#____#", "######"])
    next_state, _, _, _ = apply_actions(current, ["Right"])
    result = score_response(format_response(current, current), current, next_state, ["Right"], 1, "exact_only")
    assert result.reward == 0.5  # format 0.1 + perception 0.4
    assert result.metrics["prediction_success_rate"] == 0.0


def test_conditional_metric_is_ratio_of_sums():
    rows = [
        {"conditional_prediction_success_numerator": 1, "conditional_prediction_success_denominator": 1},
        {"conditional_prediction_success_numerator": 0, "conditional_prediction_success_denominator": 1},
        {"conditional_prediction_success_numerator": 0, "conditional_prediction_success_denominator": 0},
    ]
    result = aggregate_metrics(rows)
    assert result["conditional_prediction_success_rate"] == 0.5


def test_no_change_delta_does_not_reward_copying_a_wrong_perception():
    current = grid_to_state(["######", "#_P__#", "#_XO_#", "#____#", "#____#", "######"])
    wrong = grid_to_state(["######", "#P___#", "#__XO#", "#____#", "#____#", "######"])
    # Up is a wall no-op in the true state. Both model sections consistently copy an
    # incorrect board, which must not earn the no-change delta point.
    result = score_response(format_response(wrong, wrong), current, current, ["Up"], 1, "dense")
    assert result.components["delta_score"] == 0.0
