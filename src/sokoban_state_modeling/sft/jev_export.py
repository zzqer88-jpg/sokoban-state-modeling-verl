"""Export current-state SFT records to English visual choice distribution data.

The multimodal input envelope is a dataset contract, not a request accepted by
unmodified OpenJev (which currently supports text only).
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from sokoban_state_modeling.jev.dataset import file_sha256
from sokoban_state_modeling import RENDERER_VERSION
from sokoban_state_modeling.engine.env import render_state
from sokoban_state_modeling.state.codec import StateSnapshot, grid_to_state, map_hash, state_hash

from sokoban_state_modeling.jev.models import (
    CLASSES, SYMBOLS, DESCRIPTIONS, SCHEMA, board_state, cell_question,
)


def family_hash(state: StateSnapshot) -> str:
    """Group terrain-identical rotations and reflections across splits."""
    variants = []
    for array in (state.fixed, np.fliplr(state.fixed)):
        for turns in range(4):
            rotated = np.rot90(array, turns)
            variants.append(json.dumps(rotated.tolist(), separators=(",", ":")))
    return hashlib.sha256(min(variants).encode()).hexdigest()


def build_record(source: dict) -> tuple[dict, StateSnapshot]:
    grid = source["current_state"]["grid"]
    state = grid_to_state(grid)
    if int((state.entity == 1).sum()) != 1:
        raise ValueError("current state must contain exactly one player")
    if int((state.entity == 2).sum()) != int((state.fixed == 2).sum()):
        raise ValueError("box and goal counts differ")
    for name, actual in (("current_state_hash", state_hash(state)), ("map_hash", map_hash(state))):
        if name in source and source[name] != actual:
            raise ValueError(f"{name} disagrees with current grid")
    for name in ("fixed", "entity"):
        if name in source["current_state"] and not np.array_equal(
            source["current_state"][name], getattr(state, name)
        ):
            raise ValueError(f"current_state.{name} disagrees with grid")
    split = source.get("split")
    if not isinstance(split, str) or not split:
        raise ValueError("source must provide a nonempty split")
    height, width = state.shape
    sid = state_hash(state)
    questions, targets = {}, {}
    for row in range(height):
        for col in range(width):
            key = f"cell_{row}_{col}"
            label = CLASSES[SYMBOLS.index(grid[row][col])]
            questions[key] = {
                "type": "choice",
                "instructions": cell_question(row, col),
                "criteria": dict(zip(CLASSES, DESCRIPTIONS)),
            }
            targets[key] = {"type": "choice", "probabilities": {
                name: float(name == label) for name in CLASSES
            }}
    keys = list(questions)
    return {
        "schema_version": SCHEMA, "sample_id": sid, "split": split,
        "class_order": list(CLASSES), "height": height, "width": width,
        "input": {
            "image_ref": f"images/{sid}.png",
            "state": board_state(height, width),
            "questions": questions,
            "request_groups": [keys[i:i + 64] for i in range(0, len(keys), 64)],
        },
        "target": {"answers": targets},
        "provenance": {"source_task_id": source.get("task_id"),
                       "state_hash": sid, "map_hash": map_hash(state),
                       "terrain_family_hash": family_hash(state),
                       "renderer_version": RENDERER_VERSION},
    }, state


def export_jev(inputs: list[Path], output: Path) -> dict:
    """Export atomically; fail on terrain-family leakage, deduplicate states."""
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    seen, families = set(), {}
    counts, classes = Counter(), Counter()
    sources, duplicates = [], 0
    try:
        (temporary / "images").mkdir()
        with (temporary / "records.jsonl").open("w", encoding="utf-8") as dest:
            for path in inputs:
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for line_no, raw in enumerate(handle, 1):
                        digest.update(raw)
                        if not raw.strip():
                            continue
                        try:
                            record, state = build_record(json.loads(raw))
                            family = record["provenance"]["terrain_family_hash"]
                            split = record["split"]
                            if family in families and families[family] != split:
                                raise ValueError("terrain family appears in multiple splits")
                            families[family] = split
                            sid = record["sample_id"]
                            if sid in seen:
                                duplicates += 1
                                continue
                            seen.add(sid)
                            image_path = temporary / record["input"]["image_ref"]
                            Image.fromarray(render_state(state)).save(image_path)
                            record["provenance"]["image_sha256"] = hashlib.sha256(image_path.read_bytes()).hexdigest()
                            record["provenance"].update(source_file=str(path.resolve()), source_line=line_no)
                            dest.write(json.dumps(record, ensure_ascii=True) + "\n")
                            counts[split] += 1
                            for answer in record["target"]["answers"].values():
                                classes.update([max(answer["probabilities"], key=answer["probabilities"].get)])
                        except (KeyError, ValueError, TypeError) as exc:
                            raise ValueError(f"{path}:{line_no}: {exc}") from exc
                sources.append({"path": str(path.resolve()), "sha256": digest.hexdigest()})
        if not seen:
            raise ValueError("no records found")
        report = {"schema_version": SCHEMA, "records": len(seen), "duplicates_removed": duplicates,
                  "split_counts": dict(counts), "cell_class_counts": dict(classes), "sources": sources,
                  "records_sha256": file_sha256(temporary / "records.jsonl"),
                  "target_source": "engine_one_hot", "unmodified_openjev_compatible": False}
        (temporary / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
        temporary.rename(output)
        return report
    except BaseException:
        shutil.rmtree(temporary)
        raise
