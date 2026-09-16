#!/usr/bin/env python3
"""Audit identity leakage across train, validation and test manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sokoban_state_modeling.generator.audit import audit_split_leakage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_split_leakage({
        "train": args.train,
        "validation": args.validation,
        "test": args.test,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
