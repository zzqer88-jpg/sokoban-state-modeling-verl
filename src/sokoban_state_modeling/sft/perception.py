"""Deterministic perception-only, prompt-open thinking SFT data."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from sokoban_state_modeling.prompt.template import system_prompt
from sokoban_state_modeling.state.codec import grid_to_state, map_hash, state_hash
from sokoban_state_modeling.state.validation import validation_errors

PERCEPTION_SCHEMA_VERSION = "sokoban-perception-thinking-sft-v1"


def build_perception_prompt(rows: int, cols: int) -> str:
    return f'''<image>
You are identifying the current Sokoban state from an RGB image.

The image shows the current {rows}x{cols} board. Use exactly one character per cell:
# = wall
_ = empty floor
O = empty target (uppercase letter O)
X = box on floor
P = player on floor
* = box on target
+ = player on target

Infer the current symbolic board from the image.
Use the thinking section to identify the board. After </think>, return only the perception grid without explanation. Do not output the literal placeholder text "...grid rows...". Output exactly {rows} grid rows between the tags, with exactly {cols} symbols per row:
<perception>
...grid rows...
</perception>'''


def build_perception_record(source: dict[str, Any]) -> dict[str, Any]:
    """Accept a manifest/SFT row; only current-state facts enter supervision."""
    state = grid_to_state(source["current_state"]["grid"])
    errors = validation_errors(state)
    if errors:
        raise ValueError("invalid current state: " + "; ".join(errors))
    canonical = state.to_dict()
    for key in ("fixed", "entity"):
        if key in source["current_state"] and source["current_state"][key] != canonical[key]:
            raise ValueError(f"current_state.{key} disagrees with grid")
    digest = state_hash(state)
    board_hash = map_hash(state)
    for key, expected in (("current_state_hash", digest), ("map_hash", board_hash)):
        if key in source and source[key] != expected:
            raise ValueError(f"{key} mismatch")

    def coords(mask: np.ndarray) -> list[str]:
        return [f"({int(r) + 1},{int(c) + 1})" for r, c in np.argwhere(mask)]

    player = coords(state.entity == 1)[0]
    boxes = ",".join(coords(state.entity == 2))
    targets = ",".join(coords(state.fixed == 2))
    reasoning = f"Initial: player={player}; boxes=[{boxes}]; targets=[{targets}]."
    answer = "<perception>\n" + "\n".join(canonical["grid"]) + "\n</perception>"
    rows, cols = state.shape
    return {
        "sft_schema_version": PERCEPTION_SCHEMA_VERSION,
        "task_id": f"perception_{digest}",
        "source_task_id": source.get("task_id"),
        "split": source.get("split", "unspecified"),
        "messages": [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": build_perception_prompt(rows, cols)},
        ],
        "enable_thinking": True,
        "thinking_prompt_open": True,
        "current_state": canonical,
        "current_state_hash": digest,
        "map_hash": board_hash,
        "reasoning": reasoning,
        "answer": answer,
        "target": reasoning + "</think>" + answer,
        "metadata": {"size": f"{rows}x{cols}", "num_boxes": int((state.entity == 2).sum())},
    }


def validate_perception_record(record: dict[str, Any]) -> None:
    if record.get("sft_schema_version") != PERCEPTION_SCHEMA_VERSION:
        raise ValueError("unsupported perception SFT schema")
    expected = build_perception_record(record)
    for key in ("messages", "reasoning", "answer", "target", "enable_thinking", "thinking_prompt_open"):
        if record.get(key) != expected[key]:
            raise ValueError(f"perception SFT {key} disagrees with current state contract")


def export_perception_sft(source_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Read JSONL without an index, deduplicate states, publish data plus audit.

    Source may be an existing transition manifest, thinking export, or rows with
    just current_state.grid and split. No actions or next-state labels are used.
    Each invocation accepts one split only. Existing files are never overwritten.
    """
    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    audit = output.with_suffix(output.suffix + ".audit.json")
    partial = output.with_name(output.name + ".partial")
    audit_partial = audit.with_name(audit.name + ".partial")
    if source in (output, audit, partial, audit_partial):
        raise ValueError("source and output paths must differ")
    for path in (output, audit, partial, audit_partial):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    splits: set[str] = set()
    sizes: Counter[str] = Counter()
    source_digest, output_digest = hashlib.sha256(), hashlib.sha256()
    total = 0
    with source.open("rb") as incoming, partial.open("xb") as destination:
        for line_number, line in enumerate(incoming, 1):
            source_digest.update(line)
            if not line.strip():
                continue
            try:
                record = build_perception_record(json.loads(line))
                validate_perception_record(record)
                splits.add(record["split"])
                if len(splits) > 1:
                    raise ValueError("source mixes splits; export each split separately")
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                raise ValueError(f"{source}:{line_number}: {exc}") from exc
            total += 1
            digest = record["current_state_hash"]
            if digest in seen:
                continue
            seen.add(digest)
            record["task_index"] = len(seen) - 1
            encoded = (json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
            destination.write(encoded)
            output_digest.update(encoded)
            sizes[record["metadata"]["size"]] += 1
        if not total:
            raise ValueError("source contains no records")
        destination.flush()
        os.fsync(destination.fileno())
    report = {
        "passed": True,
        "sft_schema_version": PERCEPTION_SCHEMA_VERSION,
        "source": str(source),
        "source_sha256": source_digest.hexdigest(),
        "source_records": total,
        "exported_records": len(seen),
        "duplicates_removed": total - len(seen),
        "split": next(iter(splits)),
        "size_counts": dict(sorted(sizes.items())),
        "output": str(output),
        "output_sha256": output_digest.hexdigest(),
    }
    with audit_partial.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    # Hard-link publication fails if another writer has created a final file.
    os.link(partial, output)
    os.link(audit_partial, audit)
    partial.unlink()
    audit_partial.unlink()
    return report
