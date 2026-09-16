import numpy as np
import pytest

from sokoban_state_modeling.state.codec import grid_to_state, room_to_state, state_to_grid, state_to_room
from sokoban_state_modeling.state.validation import validation_errors


def test_grid_and_runtime_round_trip():
    grid = ["######", "#_+__#", "#_*__#", "#_X__#", "#____#", "######"]
    state = grid_to_state(grid)
    assert state_to_grid(state) == grid
    fixed, room = state_to_room(state)
    assert room[1, 2] == 5 and fixed[1, 2] == 2
    assert room[2, 2] == 3
    assert state_to_grid(room_to_state(fixed, room)) == grid
    assert validation_errors(state) == []


def test_bad_symbol_and_shape_are_rejected():
    with pytest.raises(ValueError):
        grid_to_state(["###", "#?#", "###"])
    with pytest.raises(ValueError):
        grid_to_state(["###", "##"])
