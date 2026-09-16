"""Domain-correct sum/count aggregation independent of VERL logging defaults."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Mapping

from sokoban_state_modeling.reward.metrics import METRIC_NAMES
from sokoban_state_modeling.reward.verl_reward import (
    COVERAGE_KEYS,
    DIAGNOSTIC_KEYS,
    SLICE_INDICATOR_LABELS,
)

CONDITIONAL_RATE = "conditional_prediction_success_rate"
CONDITIONAL_NUMERATOR = "conditional_prediction_success_numerator"
CONDITIONAL_DENOMINATOR = "conditional_prediction_success_denominator"
MEAN_KEYS = ("score",) + tuple(name for name in METRIC_NAMES if name != CONDITIONAL_RATE) + DIAGNOSTIC_KEYS
REQUIRED_KEYS = frozenset(MEAN_KEYS + (CONDITIONAL_NUMERATOR, CONDITIONAL_DENOMINATOR))
REWARD_EXTRA_KEYS = tuple(MEAN_KEYS) + (
    CONDITIONAL_NUMERATOR,
    CONDITIONAL_DENOMINATOR,
) + tuple(SLICE_INDICATOR_LABELS) + COVERAGE_KEYS


@dataclass
class _Bucket:
    sums: dict[str, float] = field(default_factory=lambda: {name: 0.0 for name in MEAN_KEYS})
    counts: dict[str, int] = field(default_factory=lambda: {name: 0 for name in MEAN_KEYS})
    conditional_numerator: float = 0.0
    conditional_denominator: float = 0.0
    samples: int = 0

    def update(self, metrics: Mapping[str, Any]) -> None:
        missing = REQUIRED_KEYS - metrics.keys()
        if missing:
            raise ValueError(f"reward extra-info is missing fixed keys: {sorted(missing)}")
        self.samples += 1
        for name in MEAN_KEYS:
            value = float(metrics[name])
            if not math.isnan(value):
                self.sums[name] += value
                self.counts[name] += 1
        numerator = float(metrics[CONDITIONAL_NUMERATOR])
        denominator = float(metrics[CONDITIONAL_DENOMINATOR])
        if not math.isnan(numerator) and not math.isnan(denominator):
            self.conditional_numerator += numerator
            self.conditional_denominator += denominator

    def merge(self, other: "_Bucket") -> None:
        self.samples += other.samples
        self.conditional_numerator += other.conditional_numerator
        self.conditional_denominator += other.conditional_denominator
        for name in MEAN_KEYS:
            self.sums[name] += other.sums[name]
            self.counts[name] += other.counts[name]

    def result(self) -> dict[str, float | int | None]:
        output: dict[str, float | int | None] = {"samples": self.samples}
        for name in MEAN_KEYS:
            output[name] = self.sums[name] / self.counts[name] if self.counts[name] else None
        output[CONDITIONAL_NUMERATOR] = self.conditional_numerator
        output[CONDITIONAL_DENOMINATOR] = self.conditional_denominator
        output[CONDITIONAL_RATE] = (
            self.conditional_numerator / self.conditional_denominator
            if self.conditional_denominator
            else None
        )
        return output


def slice_labels(extra_info: Mapping[str, Any]) -> tuple[str, ...]:
    labels = {
        f"k={int(extra_info['query_length'])}",
        f"size={extra_info['size']}",
        f"boxes={int(extra_info['boxes'])}",
        f"terminated={str(bool(extra_info['terminated'])).lower()}",
    }
    labels.update(f"event={event}" for event in extra_info.get("events", ()))
    return tuple(sorted(labels))


def encoded_slice_labels(metrics: Mapping[str, Any]) -> tuple[str, ...]:
    missing = set(SLICE_INDICATOR_LABELS) - metrics.keys()
    if missing:
        raise ValueError(f"reward extra-info is missing slice keys: {sorted(missing)}")
    return tuple(
        label
        for name, label in SLICE_INDICATOR_LABELS.items()
        if float(metrics[name]) == 1.0
    )


class MetricAccumulator:
    """Accumulate raw values so ranks can merge before deriving any ratio."""

    def __init__(self):
        self.overall = _Bucket()
        self.slices: dict[str, _Bucket] = {}

    def update(self, metrics: Mapping[str, Any], extra_info: Mapping[str, Any]) -> None:
        self.overall.update(metrics)
        for label in slice_labels(extra_info):
            self.slices.setdefault(label, _Bucket()).update(metrics)

    def update_encoded(self, metrics: Mapping[str, Any]) -> None:
        """Update from the fixed numeric schema returned through VERL."""
        self.overall.update(metrics)
        for label in encoded_slice_labels(metrics):
            self.slices.setdefault(label, _Bucket()).update(metrics)

    def merge(self, other: "MetricAccumulator") -> None:
        self.overall.merge(other.overall)
        for label, bucket in other.slices.items():
            self.slices.setdefault(label, _Bucket()).merge(bucket)

    def result(self) -> dict[str, Any]:
        return {
            "overall": self.overall.result(),
            "slices": {label: bucket.result() for label, bucket in sorted(self.slices.items())},
        }

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]],
    ) -> "MetricAccumulator":
        value = cls()
        for metrics, extra_info in rows:
            value.update(metrics, extra_info)
        return value


def _column_length(columns: Mapping[str, Any]) -> int:
    for key in ("score", "reward", *MEAN_KEYS):
        if key in columns:
            return len(columns[key])
    return 0


def aggregate_reward_columns(columns: Mapping[str, Any], prefix: str) -> dict[str, float | int]:
    """Aggregate VERL reward-extra columns into stable logger keys.

    V1's central trainer has already gathered the complete global batch, so the
    sums and counts here are global rather than rank-local.
    """
    length = _column_length(columns)
    accumulator = MetricAccumulator()
    for index in range(length):
        row = {key: values[index] for key, values in columns.items() if index < len(values)}
        accumulator.update_encoded(row)

    nested = accumulator.result()
    output: dict[str, float | int] = {}
    for metric, value in nested["overall"].items():
        if value is not None:
            output[f"{prefix}/overall/{metric}"] = value
    for label, values in nested["slices"].items():
        dimension, value_name = label.split("=", 1)
        for metric, value in values.items():
            if value is not None:
                output[f"{prefix}/slice/{dimension}/{value_name}/{metric}"] = value

    if length and all(key in columns for key in COVERAGE_KEYS):
        refs = [
            {
                key: float(columns[key][index])
                for key in COVERAGE_KEYS
            }
            for index in range(length)
            if not math.isnan(float(columns["coverage_cycle"][index]))
        ]
        if refs:
            latest = max(refs, key=lambda ref: (ref["coverage_cycle"], ref["coverage_position"]))
            cycle_size = length  # replaced below when dataset-length metadata is present
            output[f"{prefix}/coverage/cycle"] = int(latest["coverage_cycle"])
            output[f"{prefix}/coverage/position"] = int(latest["coverage_position"]) + 1
            output[f"{prefix}/coverage/padding_in_batch"] = int(
                sum(ref["coverage_is_padding"] for ref in refs)
            )
            output[f"{prefix}/coverage/unique_in_batch"] = len(refs) - output[
                f"{prefix}/coverage/padding_in_batch"
            ]
            del cycle_size
    return output


def reward_columns_from_extra_fields(
    extra_fields: Iterable[Any],
    include: Iterable[bool] | None = None,
) -> dict[str, list[float]]:
    """Extract the fixed reward dictionary nested by VERL's V1 AgentLoop."""
    fields = list(extra_fields)
    mask = list(include) if include is not None else [True] * len(fields)
    if len(mask) != len(fields):
        raise ValueError("extra_fields and include mask have different lengths")
    columns = {key: [] for key in REWARD_EXTRA_KEYS}
    for keep, extra_field in zip(mask, fields, strict=True):
        if not keep:
            continue
        if not isinstance(extra_field, Mapping):
            raise ValueError("VERL extra_fields row is not a mapping")
        reward_info = extra_field.get("reward_extra_info")
        if not isinstance(reward_info, Mapping):
            raise ValueError("VERL extra_fields row has no reward_extra_info mapping")
        missing = set(REWARD_EXTRA_KEYS) - reward_info.keys()
        if missing:
            raise ValueError(f"reward extra-info is missing fixed keys: {sorted(missing)}")
        for key in REWARD_EXTRA_KEYS:
            columns[key].append(float(reward_info[key]))
    return columns
