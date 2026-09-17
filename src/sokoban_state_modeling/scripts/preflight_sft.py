#!/usr/bin/env python3
"""Fail-closed CPU preflight for the formal thinking SFT exports."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from sokoban_state_modeling.sft.verl_dataset import prepare_thinking_sft_record


FORMAL_COUNTS = {
    "train": {
        "records": 12_000,
        "sizes": {"6x6": 7_200, "7x7": 3_000, "8x8": 1_800},
        "query_lengths": {"1": 4_800, "2": 4_200, "3": 3_000},
        "boxes": {"1": 10_800, "2": 1_200},
    },
    "validation": {
        "records": 1_200,
        "sizes": {"6x6": 400, "7x7": 400, "8x8": 400},
        "query_lengths": {"1": 480, "2": 420, "3": 300},
        "boxes": {"1": 1_080, "2": 120},
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_export(path: str | Path, expected_split: str) -> dict[str, Any]:
    source_path = Path(path).expanduser().resolve()
    counts = {
        "sizes": Counter(),
        "query_lengths": Counter(),
        "boxes": Counter(),
        "events": Counter(),
    }
    task_hashes: set[str] = set()
    map_hashes: set[str] = set()
    records = 0
    with source_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                prepare_thinking_sft_record(record)
            except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"invalid SFT record at {source_path}:{line_number}: {exc}") from exc
            if record.get("split") != expected_split:
                raise ValueError(
                    f"{source_path}:{line_number} has split={record.get('split')!r}, "
                    f"expected {expected_split!r}"
                )
            task_hash = str(record.get("task_hash", ""))
            if not task_hash or task_hash in task_hashes:
                raise ValueError(f"missing or duplicate task_hash at {source_path}:{line_number}")
            task_hashes.add(task_hash)
            map_hashes.add(str(record.get("map_hash", "")))
            metadata = record["metadata"]
            counts["sizes"][str(metadata["size"])] += 1
            counts["query_lengths"][str(metadata["query_length"])] += 1
            counts["boxes"][str(metadata["num_boxes"])] += 1
            counts["events"].update(str(value) for value in record["event_sequence"])
            records += 1
    if records == 0:
        raise ValueError(f"SFT export is empty: {source_path}")
    return {
        "path": str(source_path),
        "sha256": _sha256(source_path),
        "records": records,
        **{name: dict(sorted(counter.items())) for name, counter in counts.items()},
        "task_hashes": task_hashes,
        "map_hashes": map_hashes,
    }


def validate_formal_exports(train_path: str | Path, validation_path: str | Path) -> dict[str, Any]:
    train = inspect_export(train_path, "train")
    validation = inspect_export(validation_path, "validation")
    failures: list[str] = []
    for split, report in (("train", train), ("validation", validation)):
        expected = FORMAL_COUNTS[split]
        for key in ("records", "sizes", "query_lengths", "boxes"):
            if report[key] != expected[key]:
                failures.append(f"{split}.{key}: expected {expected[key]!r}, got {report[key]!r}")
        if len(report["events"]) != 7:
            failures.append(f"{split}.events: expected all 7 event classes, got {report['events']!r}")
    shared_tasks = train["task_hashes"] & validation["task_hashes"]
    shared_maps = train["map_hashes"] & validation["map_hashes"]
    if shared_tasks:
        failures.append(f"train/validation share {len(shared_tasks)} task hashes")
    if shared_maps:
        failures.append(f"train/validation share {len(shared_maps)} map hashes")
    for report in (train, validation):
        report.pop("task_hashes")
        report.pop("map_hashes")
    result = {"passed": not failures, "failures": failures, "train": train, "validation": validation}
    if failures:
        raise ValueError("formal SFT preflight failed:\n- " + "\n- ".join(failures))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate formal Sokoban thinking SFT exports")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = validate_formal_exports(args.train, args.validation)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

