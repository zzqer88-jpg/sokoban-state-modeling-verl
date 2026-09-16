"""Per-task and dataset-level state-modeling metrics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from typing import Any

import numpy as np

from sokoban_state_modeling.prompt.parser import ParsedResponse
from sokoban_state_modeling.state.codec import StateSnapshot, state_to_grid
from sokoban_state_modeling.state.transition import apply_actions
from sokoban_state_modeling.state.validation import is_valid_state

METRIC_NAMES = (
    "perception_success_rate",
    "prediction_success_rate",
    "joint_success_rate",
    "conditional_prediction_success_rate",
    "format_exact_rate",
    "perception_cell_accuracy",
    "prediction_cell_accuracy",
    "perception_entity_macro_f1",
    "prediction_entity_macro_f1",
    "changed_cell_f1",
    "transition_consistency_rate",
)


def binary_f1(truth: np.ndarray, predicted: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    tp = int(np.sum(truth & predicted))
    fp = int(np.sum(~truth & predicted))
    fn = int(np.sum(truth & ~predicted))
    denominator = 2 * tp + fp + fn
    return 1.0 if denominator == 0 else (2.0 * tp) / denominator


def fixed_macro_f1(truth: StateSnapshot, predicted: StateSnapshot) -> float:
    return float(np.mean([binary_f1(truth.fixed == value, predicted.fixed == value) for value in (0, 1, 2)]))


def entity_macro_f1(truth: StateSnapshot, predicted: StateSnapshot) -> float:
    masks = (
        (truth.entity == 1, predicted.entity == 1),
        (truth.entity == 2, predicted.entity == 2),
        (truth.fixed == 2, predicted.fixed == 2),
    )
    return float(np.mean([binary_f1(a, b) for a, b in masks]))


def cell_accuracy(truth: StateSnapshot, predicted: StateSnapshot) -> float:
    return float(np.mean((truth.fixed == predicted.fixed) & (truth.entity == predicted.entity)))


def exact_match(truth: StateSnapshot, predicted: StateSnapshot | None, num_boxes: int) -> bool:
    return bool(
        predicted is not None
        and is_valid_state(predicted, num_boxes)
        and np.array_equal(truth.fixed, predicted.fixed)
        and np.array_equal(truth.entity, predicted.entity)
    )


def _change_atoms(before: StateSnapshot, after: StateSnapshot) -> set[tuple[int, int, str, str]]:
    old_grid, new_grid = state_to_grid(before), state_to_grid(after)
    return {
        (r, c, old_grid[r][c], new_grid[r][c])
        for r in range(before.shape[0])
        for c in range(before.shape[1])
        if old_grid[r][c] != new_grid[r][c]
    }


def changed_cell_f1(
    current: StateSnapshot,
    next_state: StateSnapshot,
    perceived: StateSnapshot | None,
    predicted: StateSnapshot | None,
) -> float:
    if perceived is None or predicted is None:
        return 0.0
    truth = _change_atoms(current, next_state)
    guess = _change_atoms(perceived, predicted)
    if not truth:
        return float(
            np.array_equal(current.fixed, predicted.fixed)
            and np.array_equal(current.entity, predicted.entity)
        )
    if not guess:
        return 0.0
    tp = len(truth & guess)
    return (2.0 * tp) / (len(truth) + len(guess))


def task_metrics(
    parsed: ParsedResponse,
    current: StateSnapshot,
    next_state: StateSnapshot,
    actions: list[str],
    num_boxes: int,
) -> dict[str, float]:
    perceived = parsed.perception.state
    predicted = parsed.prediction.state
    perception_exact = exact_match(current, perceived, num_boxes)
    prediction_exact = exact_match(next_state, predicted, num_boxes)
    consistency = 0.0
    if perceived is not None and predicted is not None and is_valid_state(perceived, num_boxes) and is_valid_state(predicted, num_boxes):
        try:
            replayed, _, _, _ = apply_actions(perceived, actions)
            consistency = float(
                np.array_equal(replayed.fixed, predicted.fixed)
                and np.array_equal(replayed.entity, predicted.entity)
            )
        except ValueError:
            consistency = 0.0
    return {
        "perception_success_rate": float(perception_exact),
        "prediction_success_rate": float(prediction_exact),
        "joint_success_rate": float(perception_exact and prediction_exact),
        "conditional_prediction_success_rate": float(prediction_exact) if perception_exact else float("nan"),
        "conditional_prediction_success_numerator": float(perception_exact and prediction_exact),
        "conditional_prediction_success_denominator": float(perception_exact),
        # This is deliberately the same predicate used by the reward.  The looser
        # ParsedResponse.format_valid remains available to parser diagnostics, but
        # reporting it here made the dashboard disagree with format_score.
        "format_exact_rate": float(parsed.format_exact),
        "perception_cell_accuracy": cell_accuracy(current, perceived) if perceived is not None else 0.0,
        "prediction_cell_accuracy": cell_accuracy(next_state, predicted) if predicted is not None else 0.0,
        "perception_entity_macro_f1": entity_macro_f1(current, perceived) if perceived is not None else 0.0,
        "prediction_entity_macro_f1": entity_macro_f1(next_state, predicted) if predicted is not None else 0.0,
        "changed_cell_f1": changed_cell_f1(current, next_state, perceived, predicted),
        "transition_consistency_rate": consistency,
    }


def aggregate_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    rows = list(rows)
    if not rows:
        return {name: float("nan") for name in METRIC_NAMES}
    result: dict[str, float] = {}
    for name in METRIC_NAMES:
        if name == "conditional_prediction_success_rate":
            numerator = sum(float(row.get("conditional_prediction_success_numerator", 0.0)) for row in rows)
            denominator = sum(float(row.get("conditional_prediction_success_denominator", 0.0)) for row in rows)
            result[name] = numerator / denominator if denominator else float("nan")
        else:
            values = [float(row[name]) for row in rows if name in row and not math.isnan(float(row[name]))]
            result[name] = sum(values) / len(values) if values else float("nan")
    return result
