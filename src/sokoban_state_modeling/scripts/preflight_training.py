#!/usr/bin/env python3
"""Fail fast on VAGEN coupling, dataset, and VERL lock mistakes."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re

from sokoban_state_modeling.compat.verl_contract import verify_lock, verify_source_lock
from sokoban_state_modeling.generator.manifest_store import ManifestStore


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def check_no_vagen_runtime(project_root: Path = PROJECT_ROOT) -> None:
    offenders: list[str] = []
    source_root = project_root / "src"
    for path in source_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(name == "vagen" or name.startswith("vagen.") for name in names):
                offenders.append(str(path.relative_to(project_root)))
    pyproject = (project_root / "pyproject.toml").read_text(encoding="utf-8")
    # Avoid requiring a TOML parser during the lightweight data-only preflight.
    # Dependency strings are quoted PEP 508 entries, so this boundary-aware check
    # catches vagen and vagen[...] without matching prose or this source file.
    if re.search(r"(?i)[\"']vagen(?:\[|\s|[<>=!~]|[\"'])", pyproject):
        offenders.append("pyproject.toml dependency")
    if offenders:
        raise RuntimeError(f"VAGEN runtime coupling found: {sorted(set(offenders))}")


def check_split(name: str, manifest: Path, index: Path, expected: int) -> None:
    store = ManifestStore(manifest, index)
    if len(store) != expected:
        raise RuntimeError(f"{name} manifest has {len(store)} rows; expected {expected}")
    first, last = store.read(0), store.read(expected - 1)
    if first["split"] != name or last["split"] != name:
        raise RuntimeError(f"{name} manifest contains the wrong split label")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-no-vagen-runtime", action="store_true")
    parser.add_argument("--verify-lock", action="store_true")
    parser.add_argument("--verify-source-lock", action="store_true")
    parser.add_argument("--verl-checkout", type=Path)
    parser.add_argument("--lock", type=Path, default=PROJECT_ROOT / "constraints" / "verl.lock.json")
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--train-index", type=Path)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--validation-index", type=Path)
    args = parser.parse_args()

    if args.check_no_vagen_runtime:
        check_no_vagen_runtime()
    if args.verify_lock:
        if args.verl_checkout is None:
            parser.error("--verify-lock requires --verl-checkout")
        verify_lock(args.verl_checkout, args.lock)
    if args.verify_source_lock:
        if args.verl_checkout is None:
            parser.error("--verify-source-lock requires --verl-checkout")
        verify_source_lock(args.verl_checkout, args.lock)
    dataset_values = (
        args.train_manifest,
        args.train_index,
        args.validation_manifest,
        args.validation_index,
    )
    if any(value is not None for value in dataset_values):
        if any(value is None for value in dataset_values):
            parser.error("all train/validation manifest and index arguments are required together")
        check_split("train", args.train_manifest, args.train_index, 50_000)
        check_split("validation", args.validation_manifest, args.validation_index, 2_000)
    if (
        not args.check_no_vagen_runtime
        and not args.verify_lock
        and not args.verify_source_lock
        and not any(value is not None for value in dataset_values)
    ):
        parser.error("select at least one preflight check")
    print("preflight passed")


if __name__ == "__main__":
    main()
