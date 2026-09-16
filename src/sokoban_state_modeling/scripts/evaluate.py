#!/usr/bin/env python3
"""Score saved model responses without an LLM judge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sokoban_state_modeling.generator.manifest_store import ManifestStore
from sokoban_state_modeling.reward.metrics import aggregate_metrics
from sokoban_state_modeling.reward.reward import score_response
from sokoban_state_modeling.state.codec import grid_to_state


def load_unique_responses(path: Path, expected_count: int, allow_partial: bool = False) -> list[dict]:
    """Load one response per task and reject biased accidental subsets."""
    by_index: dict[int, dict] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if "task_index" not in item or "response" not in item:
                raise ValueError(f"response line {line_number} needs task_index and response")
            task_index = int(item["task_index"])
            if not 0 <= task_index < expected_count:
                raise ValueError(f"response line {line_number} has out-of-range task_index={task_index}")
            if task_index in by_index:
                raise ValueError(f"duplicate response for task_index={task_index}")
            by_index[task_index] = item
    if not allow_partial and len(by_index) != expected_count:
        missing = sorted(set(range(expected_count)) - set(by_index))
        preview = missing[:10]
        suffix = "..." if len(missing) > len(preview) else ""
        raise ValueError(
            f"responses cover {len(by_index)}/{expected_count} tasks; missing {preview}{suffix}. "
            "Use --allow-partial only for an explicitly labelled development report."
        )
    return [by_index[index] for index in sorted(by_index)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--responses", type=Path, required=True, help="JSONL rows with task_index and response")
    parser.add_argument("--profile", choices=("dense", "exact_only"), default="dense")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    store = ManifestStore(args.manifest, args.index)
    metrics, rewards = [], []
    slices: dict[str, list[tuple[dict, float]]] = {}
    response_rows = load_unique_responses(args.responses, len(store), args.allow_partial)
    for item in response_rows:
        task = store.read(int(item["task_index"]))
        result = score_response(
            item["response"], grid_to_state(task["current_state"]["grid"]),
            grid_to_state(task["next_state"]["grid"]), task["query_actions"],
            task["resolved_generator_config"]["num_boxes"], args.profile,
        )
        rewards.append(result.reward)
        metrics.append(result.metrics)
        labels = (
            f"dim_room={task['resolved_generator_config']['dim_room'][0]}x{task['resolved_generator_config']['dim_room'][1]}",
            f"num_boxes={task['resolved_generator_config']['num_boxes']}",
            f"query_length={len(task['query_actions'])}",
            f"terminated={str(bool(task['terminated'])).lower()}",
        )
        for label in labels:
            slices.setdefault(label, []).append((result.metrics, result.reward))
        for event in set(task["event_sequence"]):
            slices.setdefault(f"contains_event={event}", []).append((result.metrics, result.reward))
    report = {
        "num_responses": len(rewards),
        "expected_tasks": len(store),
        "complete_coverage": len(rewards) == len(store),
        "mean_reward": sum(rewards) / len(rewards) if rewards else None,
        "metrics": aggregate_metrics(metrics),
        "slices": {
            label: {
                "count": len(values),
                "mean_reward": sum(value[1] for value in values) / len(values),
                "metrics": aggregate_metrics(value[0] for value in values),
            }
            for label, values in sorted(slices.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
