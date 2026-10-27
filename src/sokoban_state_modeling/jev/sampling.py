"""Deterministic, semantics-aware cell sampling for Jev training records."""
from __future__ import annotations

import hashlib

from sokoban_state_modeling.state.codec import StateSnapshot, state_hash


SAMPLER_VERSION = "sokoban-semantic-cells-v1"
Coordinate = tuple[int, int]


def _rank(seed: str, group: str, coordinate: Coordinate) -> str:
    row, column = coordinate
    return hashlib.sha256(f"{seed}|{SAMPLER_VERSION}|{group}|{row}|{column}".encode()).hexdigest()


def _ordered(candidates: set[Coordinate], seed: str, group: str) -> list[Coordinate]:
    return sorted(candidates, key=lambda coordinate: (_rank(seed, group, coordinate), coordinate))


def select_training_cells(current: StateSnapshot, budget: int = 16) -> tuple[Coordinate, ...]:
    """Select semantic cells, local context, background types, and board regions."""
    if type(budget) is not int or budget < 1:
        raise ValueError("cell budget must be a positive integer")
    height, width = current.shape
    all_cells = {(row, column) for row in range(height) for column in range(width)}
    if budget >= len(all_cells):
        return tuple(sorted(all_cells))

    seed = state_hash(current)
    selected: set[Coordinate] = set()

    def add(candidates: set[Coordinate], group: str, limit: int | None = None) -> None:
        available = candidates - selected
        count = len(available) if limit is None else min(limit, len(available))
        count = min(count, budget - len(selected))
        selected.update(_ordered(available, seed, group)[:count])

    players = {(r, c) for r, c in all_cells if current.entity[r, c] == 1}
    boxes = {(r, c) for r, c in all_cells if current.entity[r, c] == 2}
    goals = {(r, c) for r, c in all_cells if current.fixed[r, c] == 2}
    add(players, "players")
    add(boxes, "boxes")
    add(goals, "goals")

    semantic_cells = players | boxes | goals
    neighbors: set[Coordinate] = set()
    for row, column in semantic_cells:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            candidate = row + dr, column + dc
            if candidate in all_cells:
                neighbors.add(candidate)
    add(neighbors, "local_context", limit=5)

    boundary_walls, interior_walls, near_floors, far_floors = set(), set(), set(), set()
    for row, column in all_cells:
        fixed = int(current.fixed[row, column])
        entity = int(current.entity[row, column])
        if fixed == 0:
            if row in {0, height - 1} or column in {0, width - 1}:
                boundary_walls.add((row, column))
            else:
                interior_walls.add((row, column))
        elif fixed == 1 and entity == 0:
            distance = min((abs(row - sr) + abs(column - sc) for sr, sc in semantic_cells),
                           default=height + width)
            (near_floors if distance <= 2 else far_floors).add((row, column))
    for name, candidates in (("boundary_wall", boundary_walls),
                             ("interior_wall", interior_walls),
                             ("near_floor", near_floors), ("far_floor", far_floors)):
        add(candidates, name, limit=1)

    remaining = all_cells - selected
    quadrants = {
        index: {coordinate for coordinate in remaining
                if min(1, 2 * coordinate[0] // height) * 2
                + min(1, 2 * coordinate[1] // width) == index}
        for index in range(4)
    }
    while len(selected) < budget:
        available = [index for index, cells in quadrants.items() if cells]
        if not available:
            break
        counts = {
            index: sum(min(1, 2 * row // height) * 2 + min(1, 2 * column // width) == index
                       for row, column in selected)
            for index in available
        }
        quadrant = min(available, key=lambda index: (counts[index], index))
        coordinate = _ordered(quadrants[quadrant], seed, f"quadrant_{quadrant}")[0]
        selected.add(coordinate)
        quadrants[quadrant].remove(coordinate)

    if len(selected) != budget:
        raise RuntimeError("cell sampler failed to fill its requested budget")
    return tuple(sorted(selected))
