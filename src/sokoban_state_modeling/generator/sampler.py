"""Auditable sampler profile and exact marginal quota schedules."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Sequence

import numpy as np


def largest_remainder(total: int, weights: Sequence[float]) -> list[int]:
    weights = np.asarray(weights, dtype=float)
    if total < 0 or len(weights) == 0 or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights must be non-negative and sum to one")
    raw = weights * total
    counts = np.floor(raw).astype(int)
    order = np.argsort(-(raw - counts), kind="stable")
    for index in order[: total - int(counts.sum())]:
        counts[index] += 1
    return counts.tolist()


def round_half_up(value: float) -> int:
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


@dataclass(frozen=True)
class WeightedChoices:
    values: tuple[Any, ...]
    weights: tuple[float, ...]


@dataclass(frozen=True)
class SamplerProfile:
    name: str = "main-v1"
    dataset_seed: int = 18427
    dim_room: WeightedChoices = WeightedChoices(((6, 6), (7, 7), (8, 8)), (0.60, 0.25, 0.15))
    num_boxes: WeightedChoices = WeightedChoices((1, 2), (0.90, 0.10))
    topology_multiplier: WeightedChoices = WeightedChoices((0.75, 1.0, 1.25, 1.5), (0.15, 0.50, 0.25, 0.10))
    p_change_directions: WeightedChoices = WeightedChoices((0.15, 0.35, 0.60, 0.80), (0.15, 0.50, 0.25, 0.10))
    reverse_search_depth: WeightedChoices = WeightedChoices((100, 300, 500), (0.20, 0.60, 0.20))
    reverse_progress: WeightedChoices = WeightedChoices(("near_goal", "middle", "far"), (0.20, 0.55, 0.25))
    player_distance: WeightedChoices = WeightedChoices(("adjacent", "near", "medium", "far"), (0.30, 0.30, 0.25, 0.15))
    primary_state_category: WeightedChoices = WeightedChoices(
        ("ordinary_solvable", "player_near_box", "near_completion", "box_on_target", "player_on_target", "narrow_or_high_wall"),
        (0.50, 0.20, 0.15, 0.05, 0.05, 0.05),
    )
    action_length: WeightedChoices = WeightedChoices((1, 2, 3), (0.40, 0.35, 0.25))
    terminal: WeightedChoices = WeightedChoices((False, True), (0.90, 0.10))
    max_task_attempts: int = 500
    max_reverse_nodes: int = 200_000
    max_forward_bfs_nodes: int = 500_000
    max_snapshot_candidates_per_depth: int = 16
    max_snapshot_candidates_per_bucket: int = 256


MAIN_V1 = SamplerProfile()


class QuotaSchedule:
    """Deterministic independently shuffled exact marginal assignments."""

    def __init__(self, total: int, profile: SamplerProfile, seed: int, split: str = "train"):
        self.total = int(total)
        self.profile = profile
        self.rng = np.random.default_rng(seed)
        self._columns: dict[str, list[Any]] = {}
        for name in (
            "dim_room", "topology_multiplier", "p_change_directions",
            "reverse_search_depth", "primary_state_category", "action_length",
        ):
            choices: WeightedChoices = getattr(profile, name)
            if name == "dim_room" and split in {"validation", "test"}:
                choices = WeightedChoices(choices.values, (1 / 3, 1 / 3, 1 / 3))
            values: list[Any] = []
            for value, count in zip(choices.values, largest_remainder(total, choices.weights)):
                values.extend([value] * count)
            self.rng.shuffle(values)
            self._columns[name] = values

        # Coupled axes are built after primary categories. Every box_on_target route
        # needs two boxes, and every player_near_box route is adjacent. Fill only the
        # remaining quota from other rows so both exact marginals remain true.
        categories = self._columns["primary_state_category"]

        def rebuild_with_forced(name: str, forced: dict[int, Any]) -> None:
            choices: WeightedChoices = getattr(profile, name)
            counts = {
                value: count
                for value, count in zip(choices.values, largest_remainder(total, choices.weights))
            }
            values: list[Any] = [None] * total
            for row_index, value in forced.items():
                values[row_index] = value
                counts[value] -= 1
            if min(counts.values()) < 0:
                raise ValueError(f"forced {name} assignments exceed its marginal quota")
            remaining = [value for value in choices.values for _ in range(counts[value])]
            self.rng.shuffle(remaining)
            for row_index, value in zip(
                (i for i, current in enumerate(values) if current is None), remaining,
            ):
                values[row_index] = value
            self._columns[name] = values

        narrow_rows = [
            i for i, value in enumerate(categories) if value == "narrow_or_high_wall"
        ]

        # A box-on-target query must actually exercise leaving a target. Reserve
        # length-three blocks for those rows, then fill the untouched rows while
        # preserving the global 40/35/25 action-length marginal.
        action_counts = {
            value: count
            for value, count in zip(profile.action_length.values, largest_remainder(total, profile.action_length.weights))
        }
        action_values: list[Any] = [None] * total
        for i, value in enumerate(categories):
            if value == "box_on_target":
                action_values[i] = 3
                action_counts[3] -= 1
        if min(action_counts.values()) < 0:
            raise ValueError("not enough length-three action slots for box_on_target rows")
        remaining_actions = [value for value in profile.action_length.values for _ in range(action_counts[value])]
        self.rng.shuffle(remaining_actions)
        for i, value in zip((i for i, value in enumerate(action_values) if value is None), remaining_actions):
            action_values[i] = value
        self._columns["action_length"] = action_values

        near_rows = [i for i, value in enumerate(categories) if value == "near_completion"]
        self.rng.shuffle(near_rows)
        terminal_count = largest_remainder(total, profile.terminal.weights)[1]
        terminal_values = [False] * total
        for i in near_rows[:terminal_count]:
            terminal_values[i] = True
        self._columns["terminal"] = terminal_values

        # Terminal rows must come from the near-goal reverse third. Allocate the rest
        # of the exact 20/55/25 marginal around that forced subset.
        progress_counts = largest_remainder(total, profile.reverse_progress.weights)
        progress_values: list[Any] = [None] * total
        forced_near = [
            i for i, value in enumerate(categories)
            if value in {"near_completion", "narrow_or_high_wall"}
        ]
        for i in forced_near:
            progress_values[i] = "near_goal"
        eligible_progress = [i for i, value in enumerate(progress_values) if value is None]
        remaining_progress = ["near_goal"] * (progress_counts[0] - len(forced_near))
        for value, count in zip(profile.reverse_progress.values[1:], progress_counts[1:]):
            remaining_progress.extend([value] * count)
        self.rng.shuffle(eligible_progress)
        self.rng.shuffle(remaining_progress)
        for i, value in zip(eligible_progress, remaining_progress):
            progress_values[i] = value
        self._columns["reverse_progress"] = progress_values

        box_values = [1] * total
        forced_boxes = [
            i for i, value in enumerate(categories)
            if value in {"box_on_target", "player_on_target"}
        ]
        desired_two = largest_remainder(total, profile.num_boxes.weights)[1]
        eligible = [i for i in range(total) if i not in set(forced_boxes)]
        self.rng.shuffle(eligible)
        for i in forced_boxes + eligible[: desired_two - len(forced_boxes)]:
            box_values[i] = 2
        self._columns["num_boxes"] = box_values

        distance_values: list[Any] = [None] * total
        forced_adjacent = [i for i, value in enumerate(categories) if value == "player_near_box"]
        desired_counts = largest_remainder(total, profile.player_distance.weights)
        for i in forced_adjacent:
            distance_values[i] = "adjacent"
        for i, is_terminal_task in enumerate(terminal_values):
            if is_terminal_task:
                distance_values[i] = "adjacent"
        for i, category in enumerate(categories):
            if category == "near_completion" and not terminal_values[i]:
                distance_values[i] = "near"
            elif category == "box_on_target":
                distance_values[i] = "near"
            elif category == "player_on_target":
                distance_values[i] = "near"
        desired_far = desired_counts[3]
        forced_far = distance_values.count("far")
        ordinary_rows = [
            i for i, category in enumerate(categories)
            if category == "ordinary_solvable" and distance_values[i] is None
        ]
        self.rng.shuffle(ordinary_rows)
        if desired_far - forced_far > len(ordinary_rows):
            raise ValueError("not enough ordinary rows for the coupled far-distance quota")
        for i in ordinary_rows[: desired_far - forced_far]:
            distance_values[i] = "far"
        forced_distance_counts = {
            value: distance_values.count(value) for value in profile.player_distance.values
        }
        extra_adjacent = desired_counts[0] - forced_distance_counts["adjacent"]
        eligible_adjacent = [
            i for i, value in enumerate(categories)
            if distance_values[i] is None and value != "ordinary_solvable"
        ]
        if extra_adjacent > len(eligible_adjacent):
            # Tiny developer smoke sets can have no special-category row available for
            # the residual adjacent quota. Shift only that rounding residue to `near`;
            # formal 1k/50k builds have ample capacity and preserve the exact profile.
            shortage = extra_adjacent - len(eligible_adjacent)
            desired_counts[0] -= shortage
            desired_counts[1] += shortage
            extra_adjacent = len(eligible_adjacent)
        self.rng.shuffle(eligible_adjacent)
        for i in eligible_adjacent[:extra_adjacent]:
            distance_values[i] = "adjacent"
        eligible = [i for i, value in enumerate(distance_values) if value is None]
        self.rng.shuffle(eligible)
        remaining_values: list[str] = []
        for value, count in zip(profile.player_distance.values[1:], desired_counts[1:]):
            remaining_values.extend([value] * (count - forced_distance_counts[value]))
        self.rng.shuffle(remaining_values)
        for i, value in zip(eligible, remaining_values):
            distance_values[i] = value
        self._columns["player_distance"] = distance_values
        far_rows = [i for i, value in enumerate(distance_values) if value == "far"]
        self.rng.shuffle(far_rows)
        topology_forced = {i: 0.75 for i in narrow_rows}
        topology_forced.update({i: 1.5 for i in far_rows[: largest_remainder(total, profile.topology_multiplier.weights)[3]]})
        topology_forced.update({i: 1.25 for i in far_rows[largest_remainder(total, profile.topology_multiplier.weights)[3]:]})
        rebuild_with_forced("topology_multiplier", topology_forced)
        direction_forced = {i: 0.15 for i in narrow_rows}
        direction_forced.update({i: 0.35 for i in far_rows})
        rebuild_with_forced("p_change_directions", direction_forced)
        # Rebuild the dimension column jointly. The 15% far-distance quota is paired
        # exactly with the 15% 8x8 quota; smaller interiors cannot produce distance
        # >= 6 reliably once walls, boxes and reachability constraints are applied.
        original_dimensions = self._columns["dim_room"]
        dimension_counts = {value: original_dimensions.count(value) for value in self.profile.dim_room.values}
        dimensions: list[Any] = [None] * total
        for i in far_rows:
            if dimension_counts.get((8, 8), 0) <= 0:
                raise ValueError("the 8x8 quota is too small for the far-distance quota")
            dimensions[i] = (8, 8)
            dimension_counts[(8, 8)] -= 1
        remaining_dimensions = [
            value for value in self.profile.dim_room.values
            for _ in range(dimension_counts.get(value, 0))
        ]
        self.rng.shuffle(remaining_dimensions)
        for i, value in zip((i for i, value in enumerate(dimensions) if value is None), remaining_dimensions):
            dimensions[i] = value
        self._columns["dim_room"] = dimensions
        adjacent_rows = [i for i, value in enumerate(distance_values) if value == "adjacent"]
        self.rng.shuffle(adjacent_rows)
        pushable_count = largest_remainder(len(adjacent_rows), (0.70, 0.30))[0]
        adjacent_type = [None] * total
        for rank, i in enumerate(adjacent_rows):
            adjacent_type[i] = "pushable_adjacent" if rank < pushable_count else "blocked_adjacent"
        self._columns["adjacent_type"] = adjacent_type

    def assignment(self, index: int) -> dict[str, Any]:
        if not 0 <= index < self.total:
            raise IndexError(index)
        result = {name: values[index] for name, values in self._columns.items()}
        height, width = result["dim_room"]
        auto = int(1.7 * (height + width))
        result["topology_steps"] = round_half_up(auto * result["topology_multiplier"])
        return result


DISTANCE_RANGES = {
    "adjacent": (0, 0), "near": (1, 2), "medium": (3, 5), "far": (6, None),
}


def reverse_bucket(depth: int, d_max: int) -> str:
    first = int(np.ceil(d_max / 3))
    second = int(np.ceil(2 * d_max / 3))
    return "near_goal" if depth <= first else ("middle" if depth <= second else "far")
