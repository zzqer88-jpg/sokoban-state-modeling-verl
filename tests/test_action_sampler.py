import numpy as np

from sokoban_state_modeling.generator.action_sampler import ActionSampler
from sokoban_state_modeling.state.codec import grid_to_state, state_to_grid
from sokoban_state_modeling.state.transition import apply_actions


def test_only_final_action_can_terminate():
    state = grid_to_state(["######", "#____#", "#_PXO#", "#____#", "#____#", "######"])
    block = ActionSampler().sample(state, 1, True, np.random.default_rng(0))
    assert block is not None
    assert block.actions == ["Right"]
    assert block.terminated
    replayed, events, _, terminal = apply_actions(state, block.actions)
    assert state_to_grid(replayed) == state_to_grid(block.next_state)
    assert events == ["box_enters_target"] and terminal


def test_action_suffix_after_completion_is_rejected():
    state = grid_to_state(["######", "#____#", "#_PXO#", "#____#", "#____#", "######"])
    try:
        apply_actions(state, ["Right", "Left"])
    except ValueError as exc:
        assert "after the puzzle is solved" in str(exc)
    else:
        raise AssertionError("completed prefix was accepted")


def test_required_event_filters_action_blocks():
    state = grid_to_state(["######", "#____#", "#_PXO#", "#____#", "#____#", "######"])
    assert ActionSampler().sample(
        state, 1, True, np.random.default_rng(0), required_events={"wall_noop"}
    ) is None
