"""Indexed JSONL loading with strict separation of inputs and targets."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from PIL import Image

from .models import (BoardInput, CLASSES, SCHEMA, probability_vector)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class BoardExample:
    sample_id: str
    split: str
    inputs: BoardInput
    targets: tuple[tuple[float, ...], ...]
    image_hash: str
    family: str

    def image(self) -> Image.Image:
        if file_sha256(self.inputs.image_path) != self.image_hash:
            raise ValueError(f"Image hash mismatch: {self.inputs.image_path}")
        with Image.open(self.inputs.image_path) as image:
            return image.convert("RGB")


def parse_record(record: dict, root: Path) -> BoardExample:
    if record.get("schema_version") != SCHEMA or record.get("class_order") != list(CLASSES):
        raise ValueError("Unsupported schema or class order")
    raw = record["input"]
    path = (root / raw["image_ref"]).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Image must exist inside the export directory")
    board = BoardInput.create(path, record["height"], record["width"])
    if raw["state"] != board.state:
        raise ValueError("Unexpected state instructions")
    expected = {q.key for q in board.questions}
    answers = record["target"]["answers"]
    if set(raw["questions"]) != expected or set(answers) != expected:
        raise ValueError("Questions and targets must cover each coordinate exactly once")
    targets = []
    for question in board.questions:
        value = raw["questions"][question.key]
        if (value.get("type") != "choice" or value.get("instructions") != question.instructions
                or value.get("criteria") != dict(zip(CLASSES, question.criteria))):
            raise ValueError(f"Unexpected question/criteria: {question.key}")
        if answers[question.key].get("type") != "choice":
            raise ValueError("Expected choice targets")
        targets.append(probability_vector(answers[question.key]["probabilities"]))
    provenance = record["provenance"]
    if not record.get("sample_id") or not record.get("split") or not provenance.get("terrain_family_hash"):
        raise ValueError("Missing sample identity or split provenance")
    return BoardExample(record["sample_id"], record["split"], board, tuple(targets),
                        provenance["image_sha256"], provenance["terrain_family_hash"])


class BoardDataset:
    """Keep byte offsets, not hundreds of thousands of expanded questions in RAM."""
    def __init__(self, path: Path, split: str):
        self.path, self.split = path.resolve(), split
        self.offsets: list[int] = []
        families: dict[str, str] = {}
        samples: set[str] = set()
        with self.path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                example = parse_record(json.loads(line), self.path.parent)
                if example.sample_id in samples:
                    raise ValueError("Duplicate sample_id; export with deduplication first")
                samples.add(example.sample_id)
                if example.family in families and families[example.family] != example.split:
                    raise ValueError("Terrain family leaks across splits")
                families[example.family] = example.split
                if example.split == split:
                    self.offsets.append(offset)
        if not self.offsets:
            raise ValueError(f"No records for split {split!r}")
        self.fingerprint = file_sha256(self.path)

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> BoardExample:
        with self.path.open("rb") as stream:
            stream.seek(self.offsets[index])
            return parse_record(json.loads(stream.readline()), self.path.parent)
