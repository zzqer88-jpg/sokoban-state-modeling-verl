from sokoban_state_modeling.reward.reasoning_adapter import (
    decode_preserving_reasoning_markers,
    split_reasoning,
)


class _FakeTokenizer:
    pieces = {
        1: "reason",
        2: "<think>",
        3: "</think>",
        4: "answer",
        5: "<eos>",
    }

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"<think>": [2], "</think>": [3]}[text]

    def decode(self, ids, skip_special_tokens=True):
        values = [self.pieces[int(value)] for value in ids]
        if skip_special_tokens:
            values = [value for value in values if value not in {"<think>", "</think>", "<eos>"}]
        return "".join(values)


def test_decode_preserves_thinking_boundaries_but_drops_other_special_tokens():
    tokenizer = _FakeTokenizer()
    assert decode_preserving_reasoning_markers(tokenizer, [2, 1, 3, 4, 5]) == (
        "<think>reason</think>answer"
    )
    assert decode_preserving_reasoning_markers(tokenizer, [1, 3, 4, 5]) == "reason</think>answer"
from sokoban_state_modeling.reward.verl_reward import compute_score
from sokoban_state_modeling.state.codec import grid_to_state, state_to_grid
from sokoban_state_modeling.state.transition import apply_actions


def _response(current, nxt):
    return "<perception>\n" + "\n".join(state_to_grid(current)) + "\n</perception>\n<prediction>\n" + "\n".join(state_to_grid(nxt)) + "\n</prediction>"


def test_prompt_open_thinking_boundary_and_truncation():
    split = split_reasoning("reasoning</think><action>Up</action>", "thinking", prompt_open=True)
    assert split.boundary_valid and split.reasoning_content == "reasoning"
    assert split.answer_payload == "<action>Up</action>"
    truncated = split_reasoning("unfinished reasoning", "thinking", prompt_open=True)
    assert not truncated.boundary_valid and truncated.reasoning_present


def test_non_thinking_marker_is_reported_as_violation():
    split = split_reasoning("<think>x</think><action>Up</action>", "non_thinking")
    assert split.boundary_valid and split.mode_violation
    assert split.answer_payload == "<action>Up</action>"


def test_verl_reward_has_fixed_numeric_keyset_for_both_modes():
    current = grid_to_state(["######", "#____#", "#_PXO#", "#____#", "#____#", "######"])
    nxt, _, _, _ = apply_actions(current, ["Right"])
    base = {
        "current_state": current.to_dict(),
        "next_state": nxt.to_dict(),
        "query_actions": ["Right"],
        "num_boxes": 1,
        "reward_profile": "dense",
    }
    answer = _response(current, nxt)
    direct = compute_score("test", answer, {**base, "reasoning_mode": "non_thinking"})
    thinking = compute_score(
        "test", "checked the board</think>" + answer,
        {**base, "reasoning_mode": "thinking", "thinking_prompt_open": True},
    )
    assert direct.keys() == thinking.keys()
    assert direct["score"] == thinking["score"] == 1.0
    assert thinking["reasoning_boundary_valid"] == 1.0
