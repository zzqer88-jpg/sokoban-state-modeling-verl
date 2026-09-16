"""Per-trajectory lifecycle and deterministic replay for multi-turn rollout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sokoban_state_modeling.state.codec import StateSnapshot, grid_to_state, state_to_grid

from .core import SokobanCore, SokobanObservation


@dataclass(frozen=True)
class SessionStep:
    turn: int
    action: str
    observation: SokobanObservation
    terminated: bool
    truncated: bool
    event: str
    event_subtype: str | None


@dataclass(frozen=True)
class SessionSnapshot:
    initial_grid: tuple[str, ...]
    current_grid: tuple[str, ...]
    actions: tuple[str, ...]
    state_hashes: tuple[str, ...]
    rgb_hashes: tuple[str, ...]
    events: tuple[str, ...]
    turn: int
    max_turns: int
    terminated: bool
    truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_grid": list(self.initial_grid),
            "current_grid": list(self.current_grid),
            "actions": list(self.actions),
            "state_hashes": list(self.state_hashes),
            "rgb_hashes": list(self.rgb_hashes),
            "events": list(self.events),
            "turn": self.turn,
            "max_turns": self.max_turns,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SessionSnapshot":
        return cls(
            initial_grid=tuple(value["initial_grid"]),
            current_grid=tuple(value["current_grid"]),
            actions=tuple(value["actions"]),
            state_hashes=tuple(value["state_hashes"]),
            rgb_hashes=tuple(value["rgb_hashes"]),
            events=tuple(value["events"]),
            turn=int(value["turn"]),
            max_turns=int(value["max_turns"]),
            terminated=bool(value["terminated"]),
            truncated=bool(value["truncated"]),
        )


class SokobanSession:
    """One isolated episode; environment events remain hidden from the model."""

    def __init__(self, max_turns: int = 128):
        if int(max_turns) <= 0:
            raise ValueError("max_turns must be positive")
        self.max_turns = int(max_turns)
        self.core = SokobanCore()
        self._initial: StateSnapshot | None = None
        self._observations: list[SokobanObservation] = []
        self._actions: list[str] = []
        self._events: list[str] = []
        self.terminated = False
        self.truncated = False
        self.closed = False

    @property
    def turn(self) -> int:
        return len(self._actions)

    @property
    def actions(self) -> tuple[str, ...]:
        return tuple(self._actions)

    @property
    def observations(self) -> tuple[SokobanObservation, ...]:
        return tuple(self._observations)

    def reset(self, state: StateSnapshot, num_boxes: int | None = None) -> SokobanObservation:
        if self.closed:
            raise RuntimeError("closed session cannot be reset")
        observation = self.core.reset(state, num_boxes=num_boxes)
        self._initial = state.copy()
        self._observations = [observation]
        self._actions = []
        self._events = []
        self.terminated = observation.terminated
        self.truncated = False
        return observation

    def step(self, action: str) -> SessionStep:
        if self.closed:
            raise RuntimeError("session is closed")
        if not self._observations:
            raise RuntimeError("reset must be called before step")
        if self.terminated or self.truncated:
            raise RuntimeError("episode is already finished")
        result, observation = self.core.step(action)
        self._actions.append(action)
        self._events.append(result.event)
        self._observations.append(observation)
        self.terminated = result.terminated
        self.truncated = not self.terminated and self.turn >= self.max_turns
        return SessionStep(
            turn=self.turn,
            action=action,
            observation=observation,
            terminated=self.terminated,
            truncated=self.truncated,
            event=result.event,
            event_subtype=result.event_subtype,
        )

    def snapshot(self) -> SessionSnapshot:
        if self._initial is None or not self._observations:
            raise RuntimeError("reset must be called before snapshot")
        return SessionSnapshot(
            initial_grid=tuple(state_to_grid(self._initial)),
            current_grid=tuple(state_to_grid(self.core.state)),
            actions=self.actions,
            state_hashes=tuple(item.state_hash for item in self._observations),
            rgb_hashes=tuple(item.rgb_hash for item in self._observations),
            events=tuple(self._events),
            turn=self.turn,
            max_turns=self.max_turns,
            terminated=self.terminated,
            truncated=self.truncated,
        )

    @classmethod
    def restore(cls, snapshot: SessionSnapshot | dict[str, Any]) -> "SokobanSession":
        saved = SessionSnapshot.from_dict(snapshot) if isinstance(snapshot, dict) else snapshot
        session = cls(max_turns=saved.max_turns)
        session.reset(grid_to_state(saved.initial_grid))
        for action in saved.actions:
            session.step(action)
        replayed = session.snapshot()
        expected = saved.to_dict()
        actual = replayed.to_dict()
        if actual != expected:
            raise ValueError(f"session snapshot replay mismatch: actual={actual!r}, expected={expected!r}")
        if tuple(state_to_grid(session.core.state)) != saved.current_grid:
            raise ValueError("restored current state does not match snapshot")
        return session

    def close(self) -> None:
        self.closed = True
        self._initial = None
        self._observations.clear()
        self._actions.clear()
        self._events.clear()

