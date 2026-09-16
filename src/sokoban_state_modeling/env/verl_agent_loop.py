"""Version-gated home for the future VERL Sokoban AgentLoop adapter.

The domain session is implemented now. The concrete VERL inheritance point is
deliberately unavailable until Stage 0 writes a verified interface lock.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def require_verified_verl_lock(lock_path: str | Path | None = None) -> dict:
    path = Path(lock_path or os.environ.get("SOKOBAN_VERL_LOCK", "constraints/verl.lock.json")).resolve()
    if not path.is_file():
        raise RuntimeError(f"VERL interface lock does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("verified") is not True or not value.get("commit"):
        raise RuntimeError(
            "VERL AgentLoop is disabled until Stage 0 verifies a concrete checkout "
            "and writes constraints/verl.lock.json"
        )
    return value


def build_sokoban_agent_loop_class():
    """Fail closed until code generation is bound to the verified VERL API."""
    lock = require_verified_verl_lock()
    raise NotImplementedError(
        "the verified lock exists but the Sokoban AgentLoop adapter has not been "
        f"generated for VERL commit {lock['commit']}"
    )

