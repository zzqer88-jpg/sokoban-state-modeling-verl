from collections import deque

from sokoban_state_modeling.state.codec import grid_to_state
from sokoban_state_modeling.state.transition import ACTIONS, apply_action
from sokoban_state_modeling.state.validation import (
    _solve_forward_numpy_reference,
    solve_forward,
)


FIXTURES = (
    ["######", "#____#", "#_PXO#", "#____#", "#____#", "######"],
    ["#######", "#_____#", "#_P_XO#", "#_____#", "#_____#", "#_____#", "#######"],
    ["########", "#______#", "#_P_XO_#", "#___X_O#", "#______#", "#______#", "#______#", "########"],
)


def _reachable_states(rows, depth=4):
    initial = grid_to_state(rows)
    queue = deque([(initial, 0)])
    seen = {initial.entity.tobytes()}
    while queue:
        state, current_depth = queue.popleft()
        yield state
        if current_depth == depth:
            continue
        for action in ACTIONS:
            nxt = apply_action(state, action).state
            key = nxt.entity.tobytes()
            if key not in seen:
                seen.add(key)
                queue.append((nxt, current_depth + 1))


def test_bitboard_solver_matches_numpy_reference_on_reachable_states():
    for rows in FIXTURES:
        for state in _reachable_states(rows):
            assert solve_forward(state) == _solve_forward_numpy_reference(state)


def test_bitboard_solver_preserves_max_node_boundary():
    state = grid_to_state(FIXTURES[1])
    for max_nodes in (0, 1, 2, 5, 20, 100):
        assert solve_forward(state, max_nodes) == _solve_forward_numpy_reference(state, max_nodes)


def test_bitboard_solver_rejects_invalid_state_like_reference():
    state = grid_to_state(FIXTURES[0])
    state.entity[state.entity == 1] = 0
    assert solve_forward(state) == _solve_forward_numpy_reference(state)
