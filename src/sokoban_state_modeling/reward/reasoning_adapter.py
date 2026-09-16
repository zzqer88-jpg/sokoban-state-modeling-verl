"""Separate model-native thinking text from the strict domain answer payload."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ReasoningSplit:
    reasoning_content: str
    answer_payload: str
    boundary_valid: bool
    reasoning_present: bool
    mode_violation: bool
    prompt_open: bool


def decode_preserving_reasoning_markers(tokenizer: Any, token_ids: Iterable[int]) -> str:
    """Decode generated tokens while retaining ``<think>`` boundaries only."""
    ids = [int(value) for value in token_ids]
    markers: list[tuple[str, tuple[int, ...]]] = []
    for marker in ("<think>", "</think>"):
        encoded = tuple(int(value) for value in tokenizer.encode(marker, add_special_tokens=False))
        if encoded:
            markers.append((marker, encoded))

    def decode_span(span: list[int]) -> str:
        return tokenizer.decode(span, skip_special_tokens=True) if span else ""

    output: list[str] = []
    cursor = 0
    while cursor < len(ids):
        nearest: tuple[int, str, tuple[int, ...]] | None = None
        for marker, pattern in markers:
            limit = len(ids) - len(pattern) + 1
            for start in range(cursor, max(cursor, limit)):
                if tuple(ids[start : start + len(pattern)]) == pattern:
                    candidate = (start, marker, pattern)
                    if nearest is None or start < nearest[0]:
                        nearest = candidate
                    break
        if nearest is None:
            output.append(decode_span(ids[cursor:]))
            break
        start, marker, pattern = nearest
        output.append(decode_span(ids[cursor:start]))
        output.append(marker)
        cursor = start + len(pattern)
    return "".join(output)


def split_reasoning(response: str, mode: str, *, prompt_open: bool = False) -> ReasoningSplit:
    if mode not in {"thinking", "non_thinking"}:
        raise ValueError("mode must be thinking or non_thinking")
    opening_count = response.count("<think>")
    closing_count = response.count("</think>")
    has_marker = bool(opening_count or closing_count)

    if mode == "non_thinking":
        if not has_marker:
            return ReasoningSplit("", response, True, False, False, False)
        # Preserve a uniquely delimited final answer for scoring while reporting a
        # mode violation. Ambiguous tags fail closed.
        if opening_count == 1 and closing_count == 1:
            opening = response.find("<think>")
            closing = response.find("</think>")
            if opening == 0 and opening < closing:
                reasoning = response[len("<think>"):closing]
                answer = response[closing + len("</think>"):]
                return ReasoningSplit(reasoning, answer, True, bool(reasoning), True, False)
        return ReasoningSplit("", "", False, has_marker, True, False)

    if prompt_open:
        if opening_count != 0 or closing_count != 1:
            return ReasoningSplit("", "", False, bool(response), False, True)
        closing = response.find("</think>")
        reasoning = response[:closing]
        return ReasoningSplit(reasoning, response[closing + len("</think>"):], True, bool(reasoning), False, True)

    if opening_count != 1 or closing_count != 1:
        return ReasoningSplit("", "", False, has_marker, False, False)
    opening = response.find("<think>")
    closing = response.find("</think>")
    if opening != 0 or opening >= closing:
        return ReasoningSplit("", "", False, True, False, False)
    reasoning = response[len("<think>"):closing]
    return ReasoningSplit(
        reasoning,
        response[closing + len("</think>"):],
        True,
        bool(reasoning),
        False,
        False,
    )
