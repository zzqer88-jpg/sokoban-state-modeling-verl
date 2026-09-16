"""Pure VERL custom reward entry point with a stable numeric extra-info schema."""

from __future__ import annotations

import math
from typing import Any

from sokoban_state_modeling.reward.reasoning_adapter import split_reasoning
from sokoban_state_modeling.reward.reward import score_response
from sokoban_state_modeling.state.codec import grid_to_state


DIAGNOSTIC_KEYS = (
    "reasoning_boundary_valid",
    "reasoning_present",
    "reasoning_mode_violation",
    "thinking_truncated",
    "scoring_error",
)

EVENT_VALUES = (
    "player_move",
    "box_push",
    "wall_noop",
    "blocked_box_noop",
    "box_enters_target",
    "box_leaves_target",
    "player_target_transition",
)

# These names are intentionally numeric and fixed across every result.  VERL's
# reward loop assumes that every sample has the same reward-extra schema.
SLICE_INDICATOR_LABELS = {
    **{f"slice_k_{value}": f"k={value}" for value in (1, 2, 3)},
    **{f"slice_event_{value}": f"event={value}" for value in EVENT_VALUES},
    **{f"slice_size_{value.replace('x', '_')}": f"size={value}" for value in ("6x6", "7x7", "8x8")},
    **{f"slice_boxes_{value}": f"boxes={value}" for value in (1, 2)},
    "slice_terminated_false": "terminated=false",
    "slice_terminated_true": "terminated=true",
}

COVERAGE_KEYS = (
    "coverage_task_index",
    "coverage_cycle",
    "coverage_position",
    "coverage_is_padding",
)


def _metadata_metrics(extra_info: dict[str, Any] | None) -> dict[str, float]:
    """Encode audit metadata as a fixed numeric keyset."""
    indicators = {name: 0.0 for name in SLICE_INDICATOR_LABELS}
    coverage_values = {name: math.nan for name in COVERAGE_KEYS}
    if not extra_info:
        return indicators | coverage_values

    labels = {
        f"k={int(extra_info['query_length'])}",
        f"size={extra_info['size']}",
        f"boxes={int(extra_info['boxes'])}",
        f"terminated={str(bool(extra_info['terminated'])).lower()}",
    }
    labels.update(f"event={event}" for event in set(extra_info.get("events", ())))
    unknown = labels - set(SLICE_INDICATOR_LABELS.values())
    if unknown:
        raise ValueError(f"unsupported metric slice labels: {sorted(unknown)}")
    for name, label in SLICE_INDICATOR_LABELS.items():
        indicators[name] = float(label in labels)

    coverage = extra_info.get("coverage", {})
    if coverage:
        coverage_values = {
            "coverage_task_index": float(coverage["task_index"]),
            "coverage_cycle": float(coverage["cycle"]),
            "coverage_position": float(coverage["position"]),
            "coverage_is_padding": float(bool(coverage["is_padding"])),
        }
    return indicators | coverage_values


def _failure_result(extra_info: dict[str, Any] | None = None) -> dict[str, float]:
    from sokoban_state_modeling.reward.metrics import METRIC_NAMES

    result = {"score": 0.0}
    result.update({name: math.nan for name in METRIC_NAMES if name != "conditional_prediction_success_rate"})
    result.update(
        {
            "conditional_prediction_success_numerator": math.nan,
            "conditional_prediction_success_denominator": math.nan,
            **{name: 0.0 for name in DIAGNOSTIC_KEYS},
            "scoring_error": 1.0,
        }
    )
    try:
        result.update(_metadata_metrics(extra_info))
    except Exception:
        result.update({name: 0.0 for name in SLICE_INDICATOR_LABELS})
        result.update({name: math.nan for name in COVERAGE_KEYS})
    return result


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Score one decoded response using VERL's custom reward signature."""
    del data_source
    try:
        mode = str(ground_truth["reasoning_mode"])
        prompt_open = bool(ground_truth.get("thinking_prompt_open", mode == "thinking"))
        separated = split_reasoning(solution_str, mode, prompt_open=prompt_open)
        current = grid_to_state(ground_truth["current_state"]["grid"])
        next_state = grid_to_state(ground_truth["next_state"]["grid"])
        result = score_response(
            separated.answer_payload if separated.boundary_valid else "",
            current,
            next_state,
            list(ground_truth["query_actions"]),
            int(ground_truth["num_boxes"]),
            str(ground_truth["reward_profile"]),
        )
        output = {"score": float(result.reward)}
        output.update(
            {
                name: float(value)
                for name, value in result.metrics.items()
                if name != "conditional_prediction_success_rate"
            }
        )
        output.update(
            {
                "reasoning_boundary_valid": float(separated.boundary_valid),
                "reasoning_present": float(separated.reasoning_present),
                "reasoning_mode_violation": float(separated.mode_violation),
                "thinking_truncated": float(mode == "thinking" and not separated.boundary_valid),
                "scoring_error": 0.0,
            }
        )
        output.update(_metadata_metrics(extra_info))
        return output
    except Exception:  # Infrastructure/data contract errors must remain observable.
        return _failure_result(extra_info)
