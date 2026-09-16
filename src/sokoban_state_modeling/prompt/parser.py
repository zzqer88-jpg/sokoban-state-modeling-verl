"""Strict, independently decodable two-section response parser."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional

from sokoban_state_modeling.state.codec import StateSnapshot, grid_to_state

ALLOWED = frozenset("#_OXP*+")


@dataclass(frozen=True)
class ParsedSection:
    syntax_valid: bool
    rows: tuple[str, ...] = ()
    state: Optional[StateSnapshot] = None


@dataclass(frozen=True)
class ParsedResponse:
    perception: ParsedSection
    prediction: ParsedSection
    format_valid: bool
    format_exact: bool


def _section(text: str, name: str, height: int, width: int) -> ParsedSection:
    opening, closing = f"<{name}>", f"</{name}>"
    if text.count(opening) != 1 or text.count(closing) != 1:
        return ParsedSection(False)
    start = text.find(opening) + len(opening)
    end = text.find(closing)
    if start > end:
        return ParsedSection(False)
    body = text[start:end]
    if not body.startswith("\n") or not body.endswith("\n"):
        return ParsedSection(False)
    rows = tuple(body[1:-1].split("\n"))
    valid = (
        len(rows) == height
        and all(len(row) == width for row in rows)
        and all(set(row) <= ALLOWED for row in rows)
    )
    if not valid:
        return ParsedSection(False, rows)
    return ParsedSection(True, rows, grid_to_state(rows))


def parse_response(text: str, rows: int, cols: int) -> ParsedResponse:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    perception = _section(text, "perception", rows, cols)
    prediction = _section(text, "prediction", rows, cols)
    p_open, p_close = text.find("<perception>"), text.find("</perception>")
    n_open, n_close = text.find("<prediction>"), text.find("</prediction>")
    ordered = -1 not in (p_open, p_close, n_open, n_close) and p_open < p_close < n_open < n_close
    format_valid = perception.syntax_valid and prediction.syntax_valid and ordered
    exact_pattern = re.compile(
        rf"\A<perception>\n(?:[\#_OXP*+]{{{cols}}}\n){{{rows}}}</perception>\n"
        rf"<prediction>\n(?:[\#_OXP*+]{{{cols}}}\n){{{rows}}}</prediction>\n?\Z"
    )
    return ParsedResponse(perception, prediction, format_valid, bool(exact_pattern.fullmatch(text)))
