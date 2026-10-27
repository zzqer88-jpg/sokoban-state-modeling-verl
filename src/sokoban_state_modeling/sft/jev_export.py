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
    CLASSES, SYMBOLS, DESCRIPTIONS, SAMPLED_SCHEMA, SCHEMA, board_state, cell_question,
)
from sokoban_state_modeling.jev.sampling import SAMPLER_VERSION, select_training_cells


def family_hash(state: StateSnapshot) -> str:
    """Group terrain-identical rotations and reflections across splits."""
    variants = []
    for array in (state.fixed, np.fliplr(state.fixed)):
        for turns in range(4):
            rotated = np.rot90(array, turns)
            variants.append(json.dumps(rotated.tolist(), separators=(",", ":")))
    return hashlib.sha256(min(variants).encode()).hexdigest()


def build_record(source: dict, cell_budget: int | None = None) -> tuple[dict, StateSnapshot]:
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
    coordinates = select_training_cells(state, cell_budget) if cell_budget is not None else None
    image_id = state_hash(state)
    schema = SAMPLED_SCHEMA if coordinates is not None else SCHEMA
    sid = (hashlib.sha256(json.dumps([schema, image_id, coordinates]).encode()).hexdigest()
           if coordinates is not None else image_id)
    questions, targets = {}, {}
    selected = coordinates if coordinates is not None else tuple(
        (row, col) for row in range(height) for col in range(width)
    )
    for row, col in selected:
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
        "schema_version": schema, "sample_id": sid, "split": split,
        "class_order": list(CLASSES), "height": height, "width": width,
        "input": {
            "image_ref": f"images/{image_id}.png",
            "state": board_state(height, width),
            "questions": questions,
            "request_groups": [keys[i:i + 64] for i in range(0, len(keys), 64)],
            **({"sampling": {
                "version": SAMPLER_VERSION,
                "budget": cell_budget,
                "selected_coordinates": [list(coordinate) for coordinate in coordinates],
            }} if coordinates is not None else {}),
        },
        "target": {"answers": targets},
        "provenance": {"source_task_id": source.get("task_id"),
                       "state_hash": image_id, "map_hash": map_hash(state),
                       "terrain_family_hash": family_hash(state),
                       "renderer_version": RENDERER_VERSION},
    }, state


def _select_train_states(inputs: list[Path], limit: int | None) -> tuple[set[str] | None, int]:
    """Choose an order-independent subset of unique training boards."""
    if limit is None:
        return None, 0
    if type(limit) is not int or limit < 1:
        raise ValueError("max_train_boards must be a positive integer or None")
    candidates: set[str] = set()
    for path in inputs:
        with path.open("rb") as handle:
            for line_no, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                try:
                    source = json.loads(raw)
                    if source.get("split") == "train":
                        candidates.add(state_hash(grid_to_state(source["current_state"]["grid"])))
                except (KeyError, ValueError, TypeError) as exc:
                    raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return set(sorted(candidates)[:limit]), len(candidates)


def export_jev(inputs: list[Path], output: Path, train_cell_budget: int | None = None,
               max_train_boards: int | None = None) -> dict:
    """Export atomically; fail on terrain-family leakage, deduplicate states."""
    if output.exists():
        raise FileExistsError(output)
    if train_cell_budget is not None and (type(train_cell_budget) is not int or train_cell_budget < 1):
        raise ValueError("train_cell_budget must be a positive integer or None")
    selected_train_states, available_train_boards = _select_train_states(inputs, max_train_boards)
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
                            source = json.loads(raw)
                            source_split = source.get("split")
                            source_state_hash = state_hash(grid_to_state(source["current_state"]["grid"]))
                            if (source_split == "train" and selected_train_states is not None
                                    and source_state_hash not in selected_train_states):
                                continue
                            record, state = build_record(
                                source,
                                cell_budget=train_cell_budget if source_split == "train" else None,
                            )
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
                            if not image_path.exists():
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
                  "train_sampling": {
                      "sampler_version": SAMPLER_VERSION if train_cell_budget is not None else None,
                      "cells_per_board": train_cell_budget,
                      "max_boards": max_train_boards,
                      "available_unique_boards": available_train_boards,
                      "selected_unique_boards": (len(selected_train_states)
                                                  if selected_train_states is not None else None),
                  },
                  "records_sha256": file_sha256(temporary / "records.jsonl"),
                  "target_source": "engine_one_hot", "unmodified_openjev_compatible": False}
        (temporary / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
        temporary.rename(output)
        return report
    except BaseException:
        shutil.rmtree(temporary)
        raise
