from sokoban_state_modeling.engine.env import _load_surfaces, renderer_provenance
from sokoban_state_modeling.env import ObservableTrajectory, SokobanCore, SokobanSession
from sokoban_state_modeling.state.codec import grid_to_state, state_to_grid


def _state():
    return grid_to_state(["######", "#_#__#", "#_PXO#", "#____#", "#____#", "######"])


def test_core_renders_from_bundled_assets_and_steps():
    _load_surfaces.cache_clear()
    core = SokobanCore(_state(), num_boxes=1)
    initial = core.observe()
    assert initial.rgb.shape == (96, 96, 3)
    assert not initial.rgb.flags.writeable
    result, final = core.step("Right")
    assert result.event == "box_enters_target"
    assert final.terminated
    assert initial.rgb_hash != final.rgb_hash
    provenance = renderer_provenance()
    assert provenance["renderer"] == "bundled-gym-sokoban-compatible-sprites"
    assert len(provenance["sprite_sha256"]) == len(_load_surfaces()) == 7


def test_session_preserves_noop_history_and_restores_exactly():
    session = SokobanSession(max_turns=3)
    first = session.reset(_state(), num_boxes=1)
    noop = session.step("Up")
    assert noop.event == "wall_noop"
    assert noop.observation.rgb_hash == first.rgb_hash
    moved = session.step("Right")
    assert moved.terminated and not moved.truncated
    saved = session.snapshot()
    assert saved.actions == ("Up", "Right")
    assert len(saved.rgb_hashes) == 3
    restored = SokobanSession.restore(saved.to_dict())
    assert restored.snapshot() == saved
    assert state_to_grid(restored.core.state) == list(saved.current_grid)


def test_session_turn_limit_and_close():
    session = SokobanSession(max_turns=1)
    session.reset(_state(), num_boxes=1)
    step = session.step("Left")
    assert step.truncated and not step.terminated
    session.close()
    try:
        session.step("Right")
    except RuntimeError as exc:
        assert "closed" in str(exc)
    else:
        raise AssertionError("closed session accepted a step")


def test_observable_trajectory_keeps_complete_history_and_duplicate_noop_frame():
    session = SokobanSession(max_turns=3)
    initial = session.reset(_state(), num_boxes=1)
    history = ObservableTrajectory("Choose exactly one Sokoban action.", initial)
    noop = session.step("Up")
    history.append("<think>wall above</think><action>Up</action>", "Up", noop.observation)
    assert history.trace()["rgb_hashes"] == [initial.rgb_hash, initial.rgb_hash]
    moved = session.step("Right")
    history.append("<action>Right</action>", "Right", moved.observation)
    messages = history.model_messages()
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert len([message for message in messages if message["role"] == "user"]) == 3
    assert history.trace()["actions"] == ["Up", "Right"]
