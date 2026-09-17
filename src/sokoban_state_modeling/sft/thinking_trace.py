"""Build deterministic, replay-checked scratchpads for prompt-open thinking SFT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from sokoban_state_modeling.prompt.template import build_task_prompt, system_prompt
from sokoban_state_modeling.reward.reasoning_adapter import split_reasoning
from sokoban_state_modeling.reward.reward import score_response
from sokoban_state_modeling.state.codec import StateSnapshot, grid_to_state, state_to_grid
from sokoban_state_modeling.state.transition import DELTAS, apply_action

SFT_SCHEMA_VERSION = "sokoban-thinking-sft-v1"


@dataclass(frozen=True)
class ThinkingTrace:
    reasoning: str
    next_state: StateSnapshot
    events: tuple[str, ...]
    event_subtypes: tuple[str | None, ...]
    terminated: bool


def _position(mask: np.ndarray) -> tuple[int, int]:
    values = np.argwhere(mask)
    if len(values) != 1:
        raise ValueError(f"expected exactly one position, found {len(values)}")
    row, column = map(int, values[0])
    return row, column


def _positions(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple(tuple(map(int, value)) for value in np.argwhere(mask))


def _coord(position: tuple[int, int]) -> str:
    return f"({position[0] + 1},{position[1] + 1})"


def _coord_list(positions: Iterable[tuple[int, int]]) -> str:
    return "[" + ",".join(_coord(position) for position in positions) + "]"


def _inside(state: StateSnapshot, position: tuple[int, int]) -> bool:
    row, column = position
    return 0 <= row < state.shape[0] and 0 <= column < state.shape[1]


def _cell_label(state: StateSnapshot, position: tuple[int, int]) -> str:
    if not _inside(state, position):
        return "outside"
    row, column = position
    fixed, entity = int(state.fixed[row, column]), int(state.entity[row, column])
    if fixed == 0:
        return "wall"
    if entity == 1:
        return "player_on_target" if fixed == 2 else "player"
    if entity == 2:
        return "box_on_target" if fixed == 2 else "box"
    return "target" if fixed == 2 else "floor"


def _same_state(left: StateSnapshot, right: StateSnapshot) -> bool:
    return bool(np.array_equal(left.fixed, right.fixed) and np.array_equal(left.entity, right.entity))


def _initial_line(state: StateSnapshot) -> str:
    player = _position(state.entity == 1)
    boxes = _positions(state.entity == 2)
    targets = _positions(state.fixed == 2)
    return (
        f"Initial: player={_coord(player)}; boxes={_coord_list(boxes)}; "
        f"targets={_coord_list(targets)}."
    )


def _step_line(
    index: int,
    total_actions: int,
    state: StateSnapshot,
    action: str,
) -> tuple[str, StateSnapshot, str, str | None, bool]:
    if action not in DELTAS:
        raise ValueError(f"unknown action {action!r}")
    player = _position(state.entity == 1)
    delta = DELTAS[action]
    front = (player[0] + delta[0], player[1] + delta[1])
    front_label = _cell_label(state, front)
    result = apply_action(state, action)
    prefix = (
        f"Given action {index}/{total_actions} = {action}: "
        f"front={_coord(front)}={front_label};"
    )

    if result.event == "wall_noop":
        line = f"{prefix} result=noop; event={result.event}."
    elif result.event == "blocked_box_noop":
        beyond = (front[0] + delta[0], front[1] + delta[1])
        line = (
            f"{prefix} beyond={_coord(beyond)}={_cell_label(state, beyond)}; "
            f"result=noop; event={result.event}."
        )
    elif result.event in {"box_push", "box_enters_target", "box_leaves_target"}:
        beyond = (front[0] + delta[0], front[1] + delta[1])
        line = (
            f"{prefix} beyond={_coord(beyond)}={_cell_label(state, beyond)}; result=push; "
            f"player={_coord(player)}->{_coord(front)}; box={_coord(front)}->{_coord(beyond)}; "
            f"event={result.event}."
        )
    elif result.event in {"player_move", "player_target_transition"}:
        subtype = f"; subtype={result.event_subtype}" if result.event_subtype else ""
        line = (
            f"{prefix} result=move; player={_coord(player)}->{_coord(front)}; "
            f"event={result.event}{subtype}."
        )
    else:  # pragma: no cover - protects this schema from unreviewed new events.
        raise ValueError(f"unsupported transition event {result.event!r}")
    return line, result.state, result.event, result.event_subtype, result.terminated


def build_reasoning_trace(current: StateSnapshot, actions: Iterable[str]) -> ThinkingTrace:
    actions = tuple(actions)
    if not 1 <= len(actions) <= 3:
        raise ValueError("a reasoning trace requires one to three actions")
    state = current.copy()
    lines = [_initial_line(state)]
    events: list[str] = []
    subtypes: list[str | None] = []
    terminated = False
    for index, action in enumerate(actions, 1):
        if terminated:
            raise ValueError("action block contains actions after the puzzle is solved")
        line, state, event, subtype, terminated = _step_line(
            index,
            len(actions),
            state,
            action,
        )
        lines.append(line)
        events.append(event)
        subtypes.append(subtype)
    final_player = _position(state.entity == 1)
    final_boxes = _positions(state.entity == 2)
    lines.append(
        f"Final: player={_coord(final_player)}; boxes={_coord_list(final_boxes)}; "
        f"solved={str(terminated).lower()}."
    )
    return ThinkingTrace("\n".join(lines), state, tuple(events), tuple(subtypes), terminated)


def build_answer(current: StateSnapshot, next_state: StateSnapshot) -> str:
    return (
        "<perception>\n"
        + "\n".join(state_to_grid(current))
        + "\n</perception>\n<prediction>\n"
        + "\n".join(state_to_grid(next_state))
        + "\n</prediction>"
    )


def build_thinking_sft_record(task: dict[str, Any]) -> dict[str, Any]:
    """Convert one verified manifest row to a prompt-open thinking SFT record."""
    current = grid_to_state(task["current_state"]["grid"])
    expected_next = grid_to_state(task["next_state"]["grid"])
    actions = tuple(str(value) for value in task["query_actions"])
    trace = build_reasoning_trace(current, actions)
    if not _same_state(trace.next_state, expected_next):
        raise ValueError(f"task {task.get('task_id')} next_state disagrees with action replay")
    if list(trace.events) != task.get("event_sequence"):
        raise ValueError(f"task {task.get('task_id')} event_sequence disagrees with action replay")
    if list(trace.event_subtypes) != task.get("event_subtype_sequence"):
        raise ValueError(f"task {task.get('task_id')} event_subtype_sequence disagrees with action replay")
    if trace.terminated != bool(task.get("terminated")):
        raise ValueError(f"task {task.get('task_id')} terminated disagrees with action replay")

    rows, columns = current.shape
    answer = build_answer(current, trace.next_state)
    target = trace.reasoning + "</think>" + answer
    separated = split_reasoning(target, "thinking", prompt_open=True)
    if not separated.boundary_valid or separated.reasoning_content != trace.reasoning or separated.answer_payload != answer:
        raise ValueError(f"task {task.get('task_id')} produced an invalid thinking boundary")
    num_boxes = int(task["resolved_generator_config"]["num_boxes"])
    scored = score_response(answer, current, trace.next_state, list(actions), num_boxes, "exact_only")
    if scored.reward != 1.0:
        raise ValueError(f"task {task.get('task_id')} generated a non-exact answer")
    prompt = build_task_prompt(rows, columns, actions, "<image>")
    return {
        "sft_schema_version": SFT_SCHEMA_VERSION,
        "task_id": task["task_id"],
        "task_index": int(task["task_index"]),
        "split": task["split"],
        "messages": [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": prompt},
        ],
        "enable_thinking": True,
        "thinking_prompt_open": True,
        "current_state": task["current_state"],
        "reasoning": trace.reasoning,
        "answer": answer,
        "target": target,
        "query_actions": list(actions),
        "event_sequence": list(trace.events),
        "event_subtype_sequence": list(trace.event_subtypes),
        "terminated": trace.terminated,
        "map_hash": task["map_hash"],
        "current_state_hash": task["current_state_hash"],
        "next_state_hash": task["next_state_hash"],
        "task_hash": task["task_hash"],
        "metadata": {
            "size": f"{rows}x{columns}",
            "num_boxes": num_boxes,
            "query_length": len(actions),
        },
    }
