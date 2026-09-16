"""Pure Sokoban state machine shared by generation, reward, and rollout."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from sokoban_state_modeling.engine import render_state
from sokoban_state_modeling.state.codec import StateSnapshot, state_hash
from sokoban_state_modeling.state.transition import TransitionResult, apply_action, is_terminal
from sokoban_state_modeling.state.validation import validation_errors


def rgb_sha256(rgb: np.ndarray) -> str:
    """Hash raw RGB together with shape and dtype to prevent ambiguous bytes."""
    value = np.ascontiguousarray(rgb)
    digest = hashlib.sha256()
    digest.update(b"sokoban-rgb-v1\0")
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray(value.shape, dtype=">u8").tobytes())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class SokobanObservation:
    state: StateSnapshot
    rgb: np.ndarray
    state_hash: str
    rgb_hash: str
    terminated: bool

    def __post_init__(self) -> None:
        rgb = np.ascontiguousarray(self.rgb, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("observation RGB must have shape HxWx3")
        rgb.setflags(write=False)
        object.__setattr__(self, "rgb", rgb)


class SokobanCore:
    """Small mutable facade over the single pure transition implementation."""

    def __init__(self, state: StateSnapshot | None = None, num_boxes: int | None = None):
        self._state: StateSnapshot | None = None
        self._num_boxes = num_boxes
        if state is not None:
            self.reset(state, num_boxes=num_boxes)

    @property
    def state(self) -> StateSnapshot:
        if self._state is None:
            raise RuntimeError("core has not been reset")
        return self._state.copy()

    def reset(self, state: StateSnapshot, num_boxes: int | None = None) -> SokobanObservation:
        expected = self._num_boxes if num_boxes is None else num_boxes
        errors = validation_errors(state, expected)
        if errors:
            raise ValueError("invalid Sokoban state: " + "; ".join(errors))
        self._num_boxes = expected
        self._state = state.copy()
        return self.observe()

    def observe(self) -> SokobanObservation:
        state = self.state
        rgb = render_state(state)
        return SokobanObservation(
            state=state,
            rgb=rgb,
            state_hash=state_hash(state),
            rgb_hash=rgb_sha256(rgb),
            terminated=is_terminal(state),
        )

    def step(self, action: str) -> tuple[TransitionResult, SokobanObservation]:
        if self._state is None:
            raise RuntimeError("core has not been reset")
        result = apply_action(self._state, action)
        self._state = result.state.copy()
        return result, self.observe()

