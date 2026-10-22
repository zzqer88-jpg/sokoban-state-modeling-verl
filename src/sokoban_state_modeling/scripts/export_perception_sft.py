"""Export perception-only SFT with thinking from existing state JSONL."""

import argparse
import json
from pathlib import Path

from sokoban_state_modeling.sft.perception import export_perception_sft


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Manifest or thinking SFT JSONL")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_perception_sft(args.source, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
