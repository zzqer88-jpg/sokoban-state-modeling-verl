"""Pure Sokoban dynamics used by generation, reward checks and audits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from .codec import StateSnapshot

ACTIONS = ("Up", "Down", "Left", "Right")
DELTAS = {"Up": (-1, 0), "Down": (1, 0), "Left": (0, -1), "Right": (0, 1)}


@dataclass(frozen=True)
class TransitionResult:
    state: StateSnapshot
    event: str
    event_subtype: Optional[str]
    terminated: bool


def is_terminal(state: StateSnapshot) -> bool:
    boxes = state.entity == 2
    return bool(np.any(boxes) and np.all(state.fixed[boxes] == 2))


def _inside(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def apply_action(state: StateSnapshot, action: str) -> TransitionResult:
    if action not in DELTAS:
        raise ValueError(f"unknown action {action!r}; expected one of {ACTIONS}")
    players = np.argwhere(state.entity == 1)
    if len(players) != 1:
        raise ValueError("state must contain exactly one player")
    pr, pc = map(int, players[0])
    dr, dc = DELTAS[action]
    nr, nc = pr + dr, pc + dc
    if not _inside(state.shape, nr, nc) or state.fixed[nr, nc] == 0:
        return TransitionResult(state.copy(), "wall_noop", None, is_terminal(state))

    fixed, entity = state.fixed.copy(), state.entity.copy()
    if entity[nr, nc] == 2:
        br, bc = nr + dr, nc + dc
        blocked = (
            not _inside(state.shape, br, bc)
            or fixed[br, bc] == 0
            or entity[br, bc] != 0
        )
        if blocked:
            return TransitionResult(state.copy(), "blocked_box_noop", None, is_terminal(state))
        left_target = fixed[nr, nc] == 2
        entered_target = fixed[br, bc] == 2
        entity[br, bc] = 2
        event = "box_leaves_target" if left_target else ("box_enters_target" if entered_target else "box_push")
    else:
        event = "player_move"
    entity[pr, pc] = 0
    entity[nr, nc] = 1
    subtype = None
    if event == "player_move" and fixed[pr, pc] != fixed[nr, nc] and 2 in (fixed[pr, pc], fixed[nr, nc]):
        event = "player_target_transition"
        subtype = "enters_target" if fixed[nr, nc] == 2 else "leaves_target"
    next_state = StateSnapshot(fixed, entity)
    return TransitionResult(next_state, event, subtype, is_terminal(next_state))


def apply_actions(state: StateSnapshot, actions: Iterable[str]) -> tuple[StateSnapshot, list[str], list[Optional[str]], bool]:
    actions = list(actions)
    if not 1 <= len(actions) <= 3:
        raise ValueError("an action block must contain 1 to 3 actions")
    current = state.copy()
    events: list[str] = []
    subtypes: list[Optional[str]] = []
    for index, action in enumerate(actions):
        result = apply_action(current, action)
        current = result.state
        events.append(result.event)
        subtypes.append(result.event_subtype)
        if result.terminated and index != len(actions) - 1:
            raise ValueError("action block contains actions after the puzzle is solved")
    return current, events, subtypes, is_terminal(current)
