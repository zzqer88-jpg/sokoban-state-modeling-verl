"""Canonical two-layer Sokoban state encoding."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable

import numpy as np
from sokoban_state_modeling import SCHEMA_VERSION

SYMBOL_TO_LAYERS = {
    "#": (0, 0), "_": (1, 0), "O": (2, 0),
    "X": (1, 2), "P": (1, 1), "*": (2, 2), "+": (2, 1),
}
LAYERS_TO_SYMBOL = {value: key for key, value in SYMBOL_TO_LAYERS.items()}


@dataclass(frozen=True)
class StateSnapshot:
    fixed: np.ndarray
    entity: np.ndarray

    def __post_init__(self) -> None:
        fixed = np.asarray(self.fixed, dtype=np.uint8)
        entity = np.asarray(self.entity, dtype=np.uint8)
        if fixed.ndim != 2 or entity.ndim != 2 or fixed.shape != entity.shape:
            raise ValueError("fixed and entity must be equally shaped 2-D arrays")
        object.__setattr__(self, "fixed", np.ascontiguousarray(fixed))
        object.__setattr__(self, "entity", np.ascontiguousarray(entity))

    @property
    def shape(self) -> tuple[int, int]:
        return self.fixed.shape

    def copy(self) -> "StateSnapshot":
        return StateSnapshot(self.fixed.copy(), self.entity.copy())

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixed": self.fixed.tolist(),
            "entity": self.entity.tolist(),
            "grid": state_to_grid(self),
        }


def state_to_grid(state: StateSnapshot) -> list[str]:
    rows: list[str] = []
    for r in range(state.shape[0]):
        chars = []
        for c in range(state.shape[1]):
            pair = (int(state.fixed[r, c]), int(state.entity[r, c]))
            if pair not in LAYERS_TO_SYMBOL:
                raise ValueError(f"unrepresentable cell {pair} at {(r, c)}")
            chars.append(LAYERS_TO_SYMBOL[pair])
        rows.append("".join(chars))
    return rows


def grid_to_state(rows: Iterable[str]) -> StateSnapshot:
    rows = list(rows)
    if not rows or not rows[0]:
        raise ValueError("grid cannot be empty")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("grid rows must have equal width")
    fixed = np.empty((len(rows), width), dtype=np.uint8)
    entity = np.empty_like(fixed)
    for r, row in enumerate(rows):
        for c, symbol in enumerate(row):
            try:
                fixed[r, c], entity[r, c] = SYMBOL_TO_LAYERS[symbol]
            except KeyError as exc:
                raise ValueError(f"invalid grid symbol {symbol!r} at {(r, c)}") from exc
    return StateSnapshot(fixed, entity)


def room_to_state(room_fixed: np.ndarray, room_state: np.ndarray) -> StateSnapshot:
    """Convert gym_sokoban's runtime encoding (0..6) to two layers."""
    room_fixed = np.asarray(room_fixed)
    room_state = np.asarray(room_state)
    if room_fixed.shape != room_state.shape:
        raise ValueError("room_fixed and room_state shapes differ")
    if np.any(~np.isin(room_state, np.arange(7))):
        raise ValueError("room_state contains values outside 0..6")
    fixed = np.asarray(room_fixed, dtype=np.uint8).copy()
    entity = np.zeros(room_state.shape, dtype=np.uint8)
    entity[np.isin(room_state, (5, 6))] = 1
    entity[np.isin(room_state, (3, 4))] = 2
    # Runtime code 6 is a compatibility rendering value; fixed remains target.
    fixed[room_state == 6] = 2
    return StateSnapshot(fixed, entity)


def state_to_room(state: StateSnapshot) -> tuple[np.ndarray, np.ndarray]:
    fixed = state.fixed.astype(np.int64, copy=True)
    room = fixed.copy()
    player = state.entity == 1
    boxes = state.entity == 2
    room[player] = 5
    room[boxes & (fixed == 1)] = 4
    room[boxes & (fixed == 2)] = 3
    return fixed, room


def canonical_hash(kind: str, state: StateSnapshot, extra: Any = None) -> str:
    payload = {
        "schema": SCHEMA_VERSION,
        "kind": kind,
        "shape": list(state.shape),
        "fixed": state.fixed.tolist(),
        "entity": state.entity.tolist(),
        "extra": extra,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def map_hash(state: StateSnapshot) -> str:
    empty = StateSnapshot(state.fixed, np.zeros_like(state.entity))
    return canonical_hash("map", empty)


def state_hash(state: StateSnapshot) -> str:
    return canonical_hash("state", state)


def task_hash(state: StateSnapshot, actions: Iterable[str]) -> str:
    return canonical_hash("task", state, list(actions))
