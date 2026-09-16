"""Deterministic dense and exact-only RLVR rewards."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sokoban_state_modeling.prompt.parser import ParsedResponse, parse_response
from sokoban_state_modeling.state.codec import StateSnapshot
from sokoban_state_modeling.state.validation import is_valid_state
from .metrics import changed_cell_f1, entity_macro_f1, exact_match, fixed_macro_f1, task_metrics


@dataclass(frozen=True)
class RewardResult:
    reward: float
    profile: str
    components: dict[str, float]
    metrics: dict[str, float]
    parsed: ParsedResponse


def _full_score(truth: StateSnapshot, predicted: StateSnapshot | None) -> float:
    if predicted is None:
        return 0.0
    return 0.4 * fixed_macro_f1(truth, predicted) + 0.6 * entity_macro_f1(truth, predicted)


def score_response(
    response: str,
    current: StateSnapshot,
    next_state: StateSnapshot,
    actions: list[str],
    num_boxes: int,
    profile: str = "dense",
) -> RewardResult:
    if profile not in {"dense", "exact_only"}:
        raise ValueError("reward profile must be 'dense' or 'exact_only'")
    parsed = parse_response(response, *current.shape)
    perceived, predicted = parsed.perception.state, parsed.prediction.state
    perception_exact = float(exact_match(current, perceived, num_boxes))
    prediction_exact = float(exact_match(next_state, predicted, num_boxes))
    format_score = float(parsed.format_exact)
    perception_full = _full_score(current, perceived)
    prediction_full = _full_score(next_state, predicted)
    delta_score = changed_cell_f1(current, next_state, perceived, predicted)
    exact_all = perception_exact * prediction_exact * format_score
    if profile == "dense":
        perception_score = 0.5 * perception_full + 0.5 * perception_exact
        prediction_score = 0.25 * prediction_full + 0.25 * delta_score + 0.5 * prediction_exact
        reward = 0.4 * perception_score + 0.5 * prediction_score + 0.05 * format_score + 0.05 * exact_all
    else:
        perception_score = perception_exact
        prediction_score = prediction_exact
        reward = 0.1 * format_score + 0.4 * perception_exact + 0.5 * prediction_exact
    components = {
        "format_score": format_score,
        "perception_fixed_macro_f1": fixed_macro_f1(current, perceived) if perceived is not None else 0.0,
        "perception_entity_macro_f1": entity_macro_f1(current, perceived) if perceived is not None else 0.0,
        "perception_full": perception_full,
        "perception_exact": perception_exact,
        "perception_score": perception_score,
        "prediction_fixed_macro_f1": fixed_macro_f1(next_state, predicted) if predicted is not None else 0.0,
        "prediction_entity_macro_f1": entity_macro_f1(next_state, predicted) if predicted is not None else 0.0,
        "prediction_full": prediction_full,
        "delta_score": delta_score,
        "prediction_exact": prediction_exact,
        "prediction_score": prediction_score,
        "exact_all": exact_all,
    }
    metrics = task_metrics(parsed, current, next_state, actions, num_boxes)
    return RewardResult(float(np.clip(reward, 0.0, 1.0)), profile, components, metrics, parsed)
