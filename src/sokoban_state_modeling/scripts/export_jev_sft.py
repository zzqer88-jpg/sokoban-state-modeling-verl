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
    args = parser.parse_args()
    print(json.dumps(export_jev(args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
