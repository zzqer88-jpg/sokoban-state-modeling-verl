"""Export audited prompt-open thinking SFT records from an immutable manifest."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from sokoban_state_modeling.generator.manifest_store import ManifestStore

from .thinking_trace import SFT_SCHEMA_VERSION, build_thinking_sft_record


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_thinking_sft(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    index_path: str | Path | None = None,
    audit_path: str | Path | None = None,
    max_records: int = -1,
) -> dict[str, Any]:
    """Export deterministic JSONL plus a compact audit; never overwrite outputs."""
    manifest = Path(manifest_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    audit = (
        Path(audit_path).expanduser().resolve()
        if audit_path is not None
        else output.with_suffix(output.suffix + ".audit.json")
    )
    if max_records == 0 or max_records < -1:
        raise ValueError("max_records must be -1 or a positive integer")
    conflicts = [str(path) for path in (output, audit) if path.exists()]
    if conflicts:
        raise FileExistsError(f"refusing to overwrite SFT outputs: {conflicts}")

    store = ManifestStore(manifest, index_path)
    count = len(store) if max_records == -1 else min(len(store), int(max_records))
    output.parent.mkdir(parents=True, exist_ok=True)
    audit.parent.mkdir(parents=True, exist_ok=True)
    output_partial = output.with_name(output.name + ".partial")
    audit_partial = audit.with_name(audit.name + ".partial")
    if output_partial.exists() or audit_partial.exists():
        raise FileExistsError("unfinished SFT export exists; inspect or remove its .partial files")

    events: Counter[str] = Counter()
    sizes: Counter[str] = Counter()
    query_lengths: Counter[str] = Counter()
    with output_partial.open("w", encoding="utf-8", newline="\n") as destination:
        for index in range(count):
            record = build_thinking_sft_record(store.read(index, verify=True))
            destination.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            events.update(record["event_sequence"])
            sizes[record["metadata"]["size"]] += 1
            query_lengths[str(record["metadata"]["query_length"])] += 1
        destination.flush()
        os.fsync(destination.fileno())

    report = {
        "passed": True,
        "sft_schema_version": SFT_SCHEMA_VERSION,
        "source_manifest": str(manifest),
        "source_manifest_sha256": _sha256(manifest),
        "source_records": len(store),
        "exported_records": count,
        "complete_source_coverage": count == len(store),
        "output": str(output),
        "output_sha256": _sha256(output_partial),
        "event_counts": dict(sorted(events.items())),
        "size_counts": dict(sorted(sizes.items())),
        "query_length_counts": dict(sorted(query_lengths.items())),
    }
    with audit_partial.open("w", encoding="utf-8", newline="\n") as destination:
        destination.write(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        destination.flush()
        os.fsync(destination.fileno())
    output_partial.replace(output)
    audit_partial.replace(audit)
    return report
