"""State legality and deterministic forward solving."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .codec import StateSnapshot
from .transition import ACTIONS, apply_action, is_terminal


def validation_errors(state: StateSnapshot, num_boxes: int | None = None) -> list[str]:
    errors: list[str] = []
    if np.any(~np.isin(state.fixed, (0, 1, 2))):
        errors.append("fixed values must be 0, 1, or 2")
    if np.any(~np.isin(state.entity, (0, 1, 2))):
        errors.append("entity values must be 0, 1, or 2")
    if np.any((state.fixed == 0) & (state.entity != 0)):
        errors.append("entities cannot occupy walls")
    border = np.concatenate((state.fixed[0], state.fixed[-1], state.fixed[:, 0], state.fixed[:, -1]))
    if np.any(border != 0):
        errors.append("outer border must contain only walls")
    if int(np.sum(state.entity == 1)) != 1:
        errors.append("state must contain exactly one player")
    boxes = int(np.sum(state.entity == 2))
    targets = int(np.sum(state.fixed == 2))
    if boxes != targets:
        errors.append("box and target counts must match")
    if num_boxes is not None and (boxes != int(num_boxes) or targets != int(num_boxes)):
        errors.append(f"expected exactly {num_boxes} boxes and targets")
    return errors


def is_valid_state(state: StateSnapshot, num_boxes: int | None = None) -> bool:
    return not validation_errors(state, num_boxes)


@dataclass(frozen=True)
class SolverResult:
    status: str
    shortest_steps: int | None
    actions: tuple[str, ...] = ()
    expanded_nodes: int = 0


def _solve_forward_numpy_reference(state: StateSnapshot, max_nodes: int = 500_000) -> SolverResult:
    """Original NumPy solver retained as a differential-test oracle."""
    if validation_errors(state):
        return SolverResult("invalid", None)
    def key_of(item: StateSnapshot) -> bytes:
        # fixed is constant throughout one search; only the compact entity layer is
        # needed for identity. This avoids hashing/JSON-serialising the walls at every
        # expanded node.
        return item.entity.tobytes(order="C")

    start_key = key_of(state)
    queue = deque([state])
    parents: dict[bytes, tuple[bytes | None, str | None]] = {start_key: (None, None)}
    expanded = 0
    while queue:
        current = queue.popleft()
        current_key = key_of(current)
        if is_terminal(current):
            actions: list[str] = []
            cursor = current_key
            while parents[cursor][0] is not None:
                previous, action = parents[cursor]
                actions.append(str(action))
                cursor = previous  # type: ignore[assignment]
            actions.reverse()
            return SolverResult("solvable", len(actions), tuple(actions), expanded)
        if expanded >= max_nodes:
            return SolverResult("solver_unknown", None, expanded_nodes=expanded)
        expanded += 1
        for action in ACTIONS:
            nxt = apply_action(current, action).state
            key = key_of(nxt)
            if key not in parents:
                parents[key] = (current_key, action)
                queue.append(nxt)
    return SolverResult("unsolvable", None, expanded_nodes=expanded)


def _bitboard_neighbours(state: StateSnapshot) -> tuple[tuple[int, int, int, int], ...]:
    """Return wall-aware U/D/L/R destinations for each row-major cell."""
    rows, cols = state.shape
    if rows * cols > 64:
        raise ValueError("bitboard solver supports at most 64 cells")
    neighbours: list[tuple[int, int, int, int]] = []
    deltas = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for cell in range(rows * cols):
        row, col = divmod(cell, cols)
        destinations = []
        for delta_row, delta_col in deltas:
            next_row, next_col = row + delta_row, col + delta_col
            if (
                0 <= next_row < rows
                and 0 <= next_col < cols
                and state.fixed[next_row, next_col] != 0
            ):
                destinations.append(next_row * cols + next_col)
            else:
                destinations.append(-1)
        neighbours.append(tuple(destinations))
    return tuple(neighbours)


def solve_forward(state: StateSnapshot, max_nodes: int = 500_000) -> SolverResult:
    """Deterministic shortest-path BFS using one integer per dynamic state.

    The public semantics intentionally match the retained NumPy reference: the
    same U/D/L/R expansion order, max-node boundary, shortest action sequence,
    terminal test and expanded-node count are preserved.
    """
    if validation_errors(state):
        return SolverResult("invalid", None)
    if state.fixed.size > 64:
        return _solve_forward_numpy_reference(state, max_nodes)

    neighbours = _bitboard_neighbours(state)
    targets = 0
    boxes = 0
    player = -1
    rows, cols = state.shape
    for cell in range(rows * cols):
        row, col = divmod(cell, cols)
        if state.fixed[row, col] == 2:
            targets |= 1 << cell
        entity = int(state.entity[row, col])
        if entity == 2:
            boxes |= 1 << cell
        elif entity == 1:
            player = cell

    def pack(current_boxes: int, current_player: int) -> int:
        return current_boxes | (current_player << 64)

    start_key = pack(boxes, player)
    queue = deque([(boxes, player)])
    parents: dict[int, tuple[int | None, int | None]] = {start_key: (None, None)}
    expanded = 0
    while queue:
        current_boxes, current_player = queue.popleft()
        current_key = pack(current_boxes, current_player)
        if current_boxes and (current_boxes & ~targets) == 0:
            action_indices: list[int] = []
            cursor = current_key
            while parents[cursor][0] is not None:
                previous, action_index = parents[cursor]
                action_indices.append(int(action_index))
                cursor = int(previous)
            action_indices.reverse()
            return SolverResult(
                "solvable",
                len(action_indices),
                tuple(ACTIONS[index] for index in action_indices),
                expanded,
            )
        if expanded >= max_nodes:
            return SolverResult("solver_unknown", None, expanded_nodes=expanded)
        expanded += 1

        for action_index in range(4):
            destination = neighbours[current_player][action_index]
            if destination < 0:
                continue
            next_boxes = current_boxes
            destination_bit = 1 << destination
            if current_boxes & destination_bit:
                beyond = neighbours[destination][action_index]
                if beyond < 0 or current_boxes & (1 << beyond):
                    continue
                next_boxes = current_boxes ^ destination_bit ^ (1 << beyond)
            next_player = destination
            key = pack(next_boxes, next_player)
            if key not in parents:
                parents[key] = (current_key, action_index)
                queue.append((next_boxes, next_player))
    return SolverResult("unsolvable", None, expanded_nodes=expanded)
