"""Shared input/output contract; no neural-network or training dependencies."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
from typing import Mapping

CLASSES = ("wall", "floor", "goal", "box", "box_on_goal", "player", "player_on_goal")
SYMBOLS = "#_OX*P+"
DESCRIPTIONS = (
    "A wall.", "Empty floor with no goal, box, or player.",
    "An unoccupied goal square.", "A box on a non-goal floor square.",
    "A box on a goal square.", "The player on a non-goal floor square.",
    "The player on a goal square.",
)
SCHEMA = "sokoban-jev-perception-v1"
SAMPLED_SCHEMA = "sokoban-jev-perception-sampled-v1"


def probability_vector(values: Mapping[str, float]) -> tuple[float, ...]:
    if set(values) != set(CLASSES):
        raise ValueError("Probability keys must match the seven classes")
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) for v in values.values()):
        raise ValueError("Probabilities must be numbers")
    vector = tuple(float(values[key]) for key in CLASSES)
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in vector):
        raise ValueError("Probabilities must be finite and in [0,1]")
    if not math.isclose(sum(vector), 1.0, abs_tol=1e-6):
        raise ValueError("Probabilities must sum to one")
    return vector


def board_state(height: int, width: int) -> str:
    if type(height) is not int or type(width) is not int or min(height, width) < 1:
        raise ValueError("Board dimensions must be positive integers")
    return (f"The board has {height} rows and {width} columns. "
            "Coordinates are zero-based (row, column), with (0, 0) at the top-left. "
            "Row indices increase downward. Column indices increase to the right.")


def cell_question(row: int, column: int) -> str:
    return f"What is the current state of the cell at (row={row}, column={column})?"


@dataclass(frozen=True)
class CellQuestion:
    key: str
    row: int
    column: int
    instructions: str
    criteria: tuple[str, ...]


@dataclass(frozen=True)
class BoardInput:
    image_path: Path
    height: int
    width: int
    state: str
    questions: tuple[CellQuestion, ...]

    @classmethod
    def create(cls, image_path: Path, height: int, width: int) -> BoardInput:
        return cls(image_path, height, width, board_state(height, width), tuple(
            CellQuestion(f"cell_{r}_{c}", r, c, cell_question(r, c), DESCRIPTIONS)
            for r in range(height) for c in range(width)
        ))
