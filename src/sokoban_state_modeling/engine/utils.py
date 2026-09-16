"""Experiment-local RAGEN room generation with bounded reverse snapshots.

Topology placement and reverse moves originate from
``vagen.envs.sokoban.ragen_engine.utils`` (RAGEN-2, d97bb32).  Snapshot
collection is local to this experiment and avoids the original module globals.
"""

from __future__ import annotations

from dataclasses import dataclass
import marshal
import random
from typing import Iterator

import numpy as np

CHANGE_COORDINATES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def room_topology_generation(dim: tuple[int, int], p_change_directions: float, num_steps: int) -> np.ndarray:
    """Copied from RAGEN/gym_sokoban and kept deterministic under caller seeding."""
    masks = np.asarray((
        ((0, 0, 0), (1, 1, 1), (0, 0, 0)),
        ((0, 1, 0), (0, 1, 0), (0, 1, 0)),
        ((0, 0, 0), (1, 1, 0), (0, 1, 0)),
        ((0, 0, 0), (1, 1, 0), (1, 1, 0)),
        ((0, 0, 0), (0, 1, 1), (0, 1, 0)),
    ))
    directions = ((1, 0), (0, 1), (-1, 0), (0, -1))
    direction = random.choice(directions)
    position = np.array((random.randint(1, dim[0] - 1), random.randint(1, dim[1] - 1)))
    level = np.zeros(dim, dtype=np.int64)
    for _ in range(num_steps):
        if random.random() < p_change_directions:
            direction = random.choice(directions)
        position += direction
        position[0] = max(min(position[0], dim[0] - 2), 1)
        position[1] = max(min(position[1], dim[1] - 2), 1)
        start = position - 1
        level[start[0]:start[0] + 3, start[1]:start[1] + 3] += random.choice(masks)
    level[level > 0] = 1
    level[:, (0, dim[1] - 1)] = 0
    level[(0, dim[0] - 1), :] = 0
    return level


def place_boxes_and_player(room: np.ndarray, num_boxes: int, second_player: bool = False) -> np.ndarray:
    result = room.copy()
    positions = np.argwhere(result == 1)
    count = num_boxes + (2 if second_player else 1)
    if len(positions) <= count:
        raise RuntimeError("not enough free cells for player and boxes")
    chosen = np.random.choice(len(positions), size=count, replace=False)
    for position in positions[chosen[: 2 if second_player else 1]]:
        result[tuple(position)] = 5
    for position in positions[chosen[2 if second_player else 1:]]:
        result[tuple(position)] = 2
    return result


def box_displacement_score(mapping: dict[tuple[int, int], tuple[int, int]]) -> int:
    return int(sum(np.abs(np.asarray(target) - np.asarray(position)).sum() for target, position in mapping.items()))


def reverse_move(room_state: np.ndarray, room_structure: np.ndarray, box_mapping: dict, last_pull, action: int):
    player = np.argwhere(room_state == 5)[0]
    change = np.asarray(CHANGE_COORDINATES[action % 4])
    next_position = player + change
    if room_state[tuple(next_position)] in (1, 2):
        room_state[tuple(player)] = room_structure[tuple(player)]
        room_state[tuple(next_position)] = 5
        possible_box = player - change
        if action < 4 and room_state[tuple(possible_box)] in (3, 4):
            room_state[tuple(player)] = 3
            room_state[tuple(possible_box)] = room_structure[tuple(possible_box)]
            old_position = tuple(map(int, possible_box))
            for target, position in box_mapping.items():
                if position == old_position:
                    box_mapping[target] = tuple(map(int, player))
                    last_pull = target
                    break
    return room_state, box_mapping, last_pull


@dataclass(frozen=True)
class ReverseSnapshot:
    room_fixed: np.ndarray
    room_state: np.ndarray
    box_mapping: dict[tuple[int, int], tuple[int, int]]
    reverse_actions: tuple[int, ...]
    reverse_depth: int
    box_swaps: int
    box_displacement: int


def _normalise_runtime_state(room_state: np.ndarray, room_fixed: np.ndarray) -> np.ndarray:
    result = np.asarray(room_state, dtype=np.int64).copy()
    box_mask = np.isin(result, (3, 4))
    result[box_mask & (room_fixed == 2)] = 3
    result[box_mask & (room_fixed != 2)] = 4
    return result


def _reverse_states(
    initial: np.ndarray,
    fixed: np.ndarray,
    box_mapping: dict[tuple[int, int], tuple[int, int]],
    max_depth: int,
    max_nodes: int,
) -> Iterator[tuple[np.ndarray, dict, tuple[int, ...], int, int]]:
    stack = [(initial.copy(), box_mapping.copy(), 0, (-1, -1), ())]
    seen: set[bytes] = set()
    while stack and len(seen) < max_nodes:
        state, mapping, swaps, last_pull, actions = stack.pop()
        key = marshal.dumps(state)
        if key in seen:
            continue
        seen.add(key)
        yield state, mapping, actions, swaps, len(seen)
        if len(actions) >= max_depth:
            continue
        # Reversed push order keeps the original DFS expansion order 0,1,2,3.
        for action in reversed(range(4)):
            nxt, nxt_mapping, nxt_pull = reverse_move(
                state.copy(), fixed, mapping.copy(), last_pull, action
            )
            nxt_swaps = swaps + int(nxt_pull != last_pull)
            stack.append((nxt, nxt_mapping, nxt_swaps, nxt_pull, actions + (action,)))


def generate_room_snapshots(
    *,
    dim: tuple[int, int],
    topology_steps: int,
    p_change_directions: float,
    num_boxes: int,
    reverse_search_depth: int,
    max_reverse_nodes: int,
    max_per_depth: int,
    rng: np.random.Generator,
    tries: int = 4,
) -> tuple[np.ndarray, list[ReverseSnapshot], int]:
    """Generate one topology and bounded reservoirs of solvable reverse states."""
    for _ in range(tries):
        room = room_topology_generation(dim, p_change_directions, topology_steps)
        room = place_boxes_and_player(room, num_boxes=num_boxes, second_player=False)
        fixed = room.copy()
        fixed[fixed == 5] = 1
        initial = room.copy()
        initial[initial == 2] = 4
        targets = np.argwhere(fixed == 2)
        mapping = {tuple(map(int, target)): tuple(map(int, target)) for target in targets}
        reservoirs: dict[int, list[ReverseSnapshot]] = {}
        counts: dict[int, int] = {}
        max_seen_depth = 0
        for state, current_mapping, actions, swaps, _nodes in _reverse_states(
            initial, fixed, mapping, reverse_search_depth, max_reverse_nodes
        ):
            depth = len(actions)
            max_seen_depth = max(max_seen_depth, depth)
            displacement = int(box_displacement_score(current_mapping))
            if depth == 0 or displacement <= 0:
                continue
            snapshot = ReverseSnapshot(
                fixed.copy(), _normalise_runtime_state(state, fixed), current_mapping.copy(),
                actions, depth, swaps, displacement,
            )
            bucket = reservoirs.setdefault(depth, [])
            counts[depth] = counts.get(depth, 0) + 1
            if len(bucket) < max_per_depth:
                bucket.append(snapshot)
            else:
                replace = int(rng.integers(0, counts[depth]))
                if replace < max_per_depth:
                    bucket[replace] = snapshot
        snapshots = [item for depth in sorted(reservoirs) for item in reservoirs[depth]]
        if snapshots:
            return fixed, snapshots, max_seen_depth
    raise RuntimeError("failed to generate a room with displaced-box reverse snapshots")
