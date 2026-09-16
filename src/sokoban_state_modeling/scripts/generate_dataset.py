#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sokoban_state_modeling.generator.audit import audit_manifest
from sokoban_state_modeling.generator.dataset_builder import DatasetBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic Sokoban state-modeling tasks")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--num-tasks", type=int, required=True)
    parser.add_argument(
        "--resume", action="store_true",
        help="resume the split's committed 100-row generation chunks",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    final_manifest = output_dir / "manifests" / f"{args.split}.jsonl"
    final_index = final_manifest.with_suffix(".idx")
    final_audit = output_dir / "audits" / f"{args.split}.json"
    conflicts = [path for path in (final_manifest, final_index, final_audit) if path.exists()]
    if conflicts:
        raise FileExistsError(f"refusing to overwrite published dataset paths: {conflicts}")

    output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir / f".{args.split}.staging"
    if staging_dir.exists() and not args.resume:
        raise FileExistsError(f"unfinished staging directory exists; inspect it or rerun with --resume: {staging_dir}")
    staging_dir.mkdir(parents=True, exist_ok=True)
    manifest, index = DatasetBuilder(staging_dir, args.split, args.num_tasks).build(resume=args.resume)
    report = audit_manifest(manifest, index)
    if not report["passed"]:
        failed_audit = staging_dir / "audit-failed.json"
        failed_audit.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"staging_dir": str(staging_dir), "audit": str(failed_audit), "passed": False}, indent=2))
        raise SystemExit(1)

    final_manifest.parent.mkdir(parents=True, exist_ok=True)
    final_audit.parent.mkdir(parents=True, exist_ok=True)
    index.replace(final_index)
    # Publish the manifest last so consumers cannot observe it without its index.
    manifest.replace(final_manifest)
    report["manifest"] = str(final_manifest)
    final_audit.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for empty in (staging_dir / "manifests", staging_dir):
        empty.rmdir()
    print(json.dumps({"manifest": str(final_manifest), "index": str(final_index), "audit": str(final_audit), "passed": True}, indent=2))


if __name__ == "__main__":
    main()
