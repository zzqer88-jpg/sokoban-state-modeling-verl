"""Action-block construction with explicit terminal semantics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import product

import numpy as np

from sokoban_state_modeling.state.codec import StateSnapshot
from sokoban_state_modeling.state.transition import ACTIONS, apply_action, is_terminal

EVENT_WEIGHTS = {
    "player_move": 0.25,
    "box_push": 0.20,
    "wall_noop": 0.15,
    "blocked_box_noop": 0.15,
    "box_enters_target": 0.10,
    "box_leaves_target": 0.05,
    "player_target_transition": 0.10,
}


@dataclass(frozen=True)
class ActionBlock:
    actions: list[str]
    next_state: StateSnapshot
    events: list[str]
    event_subtypes: list[str | None]
    terminated: bool


class ActionSampler:
    def __init__(self):
        self.event_counts: Counter[str] = Counter()
        self.direction_counts: dict[str, Counter[str]] = {event: Counter() for event in EVENT_WEIGHTS}

    def _score(self, event: str, action: str) -> tuple[float, int, str]:
        total = sum(self.event_counts.values()) + 1
        event_deficit = EVENT_WEIGHTS[event] * total - self.event_counts[event]
        direction_deficit = (sum(self.direction_counts[event].values()) + 1) / 4 - self.direction_counts[event][action]
        return event_deficit, direction_deficit, action

    def sample(
        self,
        state: StateSnapshot,
        length: int,
        terminal: bool,
        rng: np.random.Generator,
        required_events: set[str] | None = None,
    ) -> ActionBlock | None:
        candidates = []
        for actions_tuple in product(ACTIONS, repeat=length):
            current = state.copy()
            events: list[str] = []
            subtypes: list[str | None] = []
            valid = True
            for index, action in enumerate(actions_tuple):
                result = apply_action(current, action)
                if result.terminated and index != length - 1:
                    valid = False
                    break
                current = result.state
                events.append(result.event)
                subtypes.append(result.event_subtype)
            if not valid or is_terminal(current) != terminal:
                continue
            if required_events and not any(event in required_events for event in events):
                continue
            if required_events:
                # Required/rare events are determined largely by room geometry. Merely
                # tie-breaking candidate action blocks cannot repair a directional bias
                # when every block in one room points the same way, so reject that room
                # unless its anchor event advances one of the currently least-used
                # directions. DatasetBuilder will try another reverse snapshot/map.
                balanced_anchor = any(
                    event in required_events
                    and self.direction_counts[event][action]
                    == min(self.direction_counts[event][direction] for direction in ACTIONS)
                    for event, action in zip(events, actions_tuple)
                )
                if not balanced_anchor:
                    continue
            scores = [self._score(event, action) for event, action in zip(events, actions_tuple)]
            candidates.append((sum(score[0] for score in scores), sum(score[1] for score in scores), actions_tuple, current, events, subtypes))
        if not candidates:
            return None
        best_prefix = max((item[0], item[1]) for item in candidates)
        tied = [item for item in candidates if (item[0], item[1]) == best_prefix]
        _, _, actions_tuple, current, events, subtypes = tied[int(rng.integers(len(tied)))]
        actions = list(actions_tuple)
        staged = list(zip(events, actions))
        for event, action in staged:
            self.event_counts[event] += 1
            self.direction_counts[event][action] += 1
        return ActionBlock(actions, current, events, subtypes, terminal)
