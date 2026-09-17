#!/usr/bin/env python3
"""CLI for deterministic prompt-open thinking SFT export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sokoban_state_modeling.sft.thinking_export import export_thinking_sft


def main() -> None:
    parser = argparse.ArgumentParser(description="Export audited Sokoban thinking SFT JSONL")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument(
        "--max-records",
        type=int,
        default=-1,
        help="developer-only prefix limit; -1 exports the complete manifest",
    )
    args = parser.parse_args()
    report = export_thinking_sft(
        args.manifest,
        args.output,
        index_path=args.index,
        audit_path=args.audit,
        max_records=args.max_records,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
