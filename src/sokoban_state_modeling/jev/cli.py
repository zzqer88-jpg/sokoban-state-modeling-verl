"""Local visual choice commands. Training starts only with the train command."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit", help="Validate data and image hashes without loading Qwen")
    audit.add_argument("--data", type=Path, required=True)
    audit.add_argument("--split", default="train")
    fitting = sub.add_parser("train")
    fitting.add_argument("--config", type=Path, required=True)
    fitting.add_argument("--resume", type=Path)
    for name in ("evaluate", "predict"):
        command = sub.add_parser(name)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--device")
        command.add_argument("--cell-batch-size", type=int, default=4)
        command.add_argument("--output", type=Path, required=True)
        if name == "evaluate":
            command.add_argument("--data", type=Path, required=True)
            command.add_argument("--split", choices=("train", "validation", "test"), default="test")
        else:
            command.add_argument("--image", type=Path, required=True)
            command.add_argument("--height", type=int, required=True)
            command.add_argument("--width", type=int, required=True)
    args = parser.parse_args()
    if args.command == "audit":
        from .dataset import BoardDataset
        data = BoardDataset(args.data, args.split)
        for i in range(len(data)):
            data[i].image()
        result = {"split": args.split, "boards": len(data), "sha256": data.fingerprint}
    elif args.command == "train":
        from .config import Config
        from .training import train
        result = train(Config.load(args.config), args.resume)
    else:
        if args.output.exists():
            parser.error("Output already exists")
        from .training import load_checkpoint
        from .service import EvaluationService
        backend = load_checkpoint(args.checkpoint, device=args.device)
        service = EvaluationService(backend, args.cell_batch_size)
        if args.command == "evaluate":
            from .dataset import BoardDataset
            from .evaluation import evaluate_dataset
            data = BoardDataset(args.data, args.split)
            args.output.mkdir(parents=True)
            result = evaluate_dataset(service, data, args.output)
        else:
            from .models import BoardInput
            result = service.evaluate(BoardInput.create(args.image, args.height, args.width))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as handle:
                json.dump(result, handle, indent=2)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
