"""Random-access JSONL manifests backed by uint64 byte-offset indexes."""

from __future__ import annotations

import json
from pathlib import Path
import struct
import threading
from typing import Any

from sokoban_state_modeling import GENERATOR_VERSION, RENDERER_VERSION, SCHEMA_VERSION
from sokoban_state_modeling.state.codec import grid_to_state, state_hash

_INDEX_CACHE: dict[tuple[str, str], tuple[int, ...]] = {}
_CACHE_LOCK = threading.Lock()


def build_index(manifest_path: str | Path, index_path: str | Path) -> int:
    manifest_path, index_path = Path(manifest_path), Path(index_path)
    offsets: list[int] = []
    with manifest_path.open("rb") as source:
        while True:
            offset = source.tell()
            line = source.readline()
            if not line:
                break
            if line.strip():
                offsets.append(offset)
    with index_path.open("wb") as output:
        for offset in offsets:
            output.write(struct.pack("<Q", offset))
    return len(offsets)


class ManifestStore:
    def __init__(self, manifest_path: str | Path, index_path: str | Path | None = None):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.index_path = Path(index_path).expanduser().resolve() if index_path else self.manifest_path.with_suffix(".idx")
        key = (str(self.manifest_path), str(self.index_path))
        with _CACHE_LOCK:
            offsets = _INDEX_CACHE.get(key)
            if offsets is None:
                raw = self.index_path.read_bytes()
                if len(raw) % 8:
                    raise ValueError("manifest index length is not a multiple of eight")
                offsets = tuple(value[0] for value in struct.iter_unpack("<Q", raw))
                if not offsets or offsets[0] != 0 or any(b <= a for a, b in zip(offsets, offsets[1:])):
                    raise ValueError("manifest offsets must start at zero and increase strictly")
                with self.manifest_path.open("rb") as source:
                    source.seek(offsets[-1])
                    last_line = source.readline()
                    if not last_line or source.read():
                        raise ValueError("manifest index count/last offset does not match JSONL")
                    try:
                        last_task_index = int(json.loads(last_line).get("task_index", -1))
                    except (json.JSONDecodeError, TypeError, ValueError) as exc:
                        raise ValueError("last manifest row is invalid") from exc
                    if last_task_index != len(offsets) - 1:
                        raise ValueError("manifest index count and final task_index disagree")
                _INDEX_CACHE[key] = offsets
        self.offsets = offsets

    def __len__(self) -> int:
        return len(self.offsets)

    def read(self, index: int, verify: bool = True) -> dict[str, Any]:
        if not 0 <= int(index) < len(self):
            raise IndexError(index)
        with self.manifest_path.open("rb") as source:
            source.seek(self.offsets[int(index)])
            row = json.loads(source.readline())
        if row.get("task_index") != int(index):
            raise ValueError(f"manifest row {index} has task_index={row.get('task_index')!r}")
        if row.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema {row.get('schema_version')!r}")
        if row.get("generator_version") != GENERATOR_VERSION:
            raise ValueError(f"unsupported generator version {row.get('generator_version')!r}")
        if row.get("renderer_version") != RENDERER_VERSION:
            raise ValueError(f"unsupported renderer version {row.get('renderer_version')!r}")
        accepted_attempt = row.get("accepted_attempt")
        if not isinstance(accepted_attempt, int) or isinstance(accepted_attempt, bool) or accepted_attempt < 0:
            raise ValueError(f"invalid accepted_attempt {accepted_attempt!r}")
        if verify:
            for key in ("current_state", "next_state"):
                snapshot = grid_to_state(row[key]["grid"])
                if snapshot.to_dict() != row[key]:
                    raise ValueError(f"{key} grid and arrays disagree")
                expected = row[f"{key}_hash"]
                if state_hash(snapshot) != expected:
                    raise ValueError(f"{key}_hash mismatch")
        return row
