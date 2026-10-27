"""Convert existing Sokoban SFT JSONL or manifests into visual Jev data."""
import argparse
import json
from pathlib import Path
from sokoban_state_modeling.sft.jev_export import export_jev


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True,
                        help="Pass all splits together to check cross-split leakage")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--train-cell-budget", type=int, default=16,
                        help="Sample this many cells from each training board (default: 16)")
    parser.add_argument("--max-train-boards", type=int, default=200,
                        help="Use at most this many unique training boards (default: 200)")
    args = parser.parse_args()
    print(json.dumps(export_jev(
        args.input, args.output,
        train_cell_budget=args.train_cell_budget,
        max_train_boards=args.max_train_boards,
    ), indent=2))


if __name__ == "__main__":
    main()
