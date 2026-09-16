"""Strict action payload parsing for future multi-turn profiles."""

from __future__ import annotations

import re

from sokoban_state_modeling.state.transition import ACTIONS

_ACTION_PATTERN = re.compile(r"\A<action>(Up|Down|Left|Right)</action>\n?\Z")


def parse_action_payload(answer_payload: str) -> str:
    match = _ACTION_PATTERN.fullmatch(answer_payload.replace("\r\n", "\n").replace("\r", "\n"))
    if match is None:
        raise ValueError("action answer must be exactly <action>Up|Down|Left|Right</action>")
    action = match.group(1)
    assert action in ACTIONS
    return action

