"""Read-only checks for a materialized and explicitly locked VERL checkout."""

from __future__ import annotations

import hashlib
import ast
import json
from pathlib import Path
import subprocess
from typing import Any


REQUIRED_VERL_FILES = (
    "verl/trainer/main_ppo.py",
    "verl/trainer/ppo/v1/trainer_base.py",
    "verl/trainer/ppo/v1/trainer_sync.py",
    "verl/trainer/ppo/core_algos.py",
    "verl/trainer/ppo/utils.py",
    "verl/utils/dataset/rl_dataset.py",
    "verl/experimental/agent_loop/agent_loop.py",
    "verl/experimental/agent_loop/single_turn_agent_loop.py",
    "verl/trainer/ppo/reward.py",
    "verl/experimental/reward_loop/reward_manager/base.py",
    "verl/experimental/reward_loop/reward_manager/naive.py",
)

REQUIRED_VERL_SYMBOLS = {
    "verl/trainer/ppo/core_algos.py": ("compute_grpo_outcome_advantage",),
    "verl/trainer/ppo/utils.py": ("need_critic",),
    "verl/trainer/ppo/v1/trainer_base.py": (
        "PPOTrainer._init_dataloader",
        "PPOTrainer._val_metrics_update",
        "PPOTrainer._compute_metrics",
    ),
    "verl/utils/dataset/rl_dataset.py": ("RLHFDataset.__init__",),
    "verl/experimental/agent_loop/agent_loop.py": (
        "AgentLoopOutput",
        "AgentLoopBase.run",
        "AgentLoopWorker._compute_score",
    ),
    "verl/trainer/ppo/reward.py": (
        "resolve_reward_manager_cls",
        "load_reward_manager",
    ),
    "verl/experimental/reward_loop/reward_manager/base.py": (
        "RewardManagerBase.__init__",
        "RewardManagerBase.assemble_rm_scores",
    ),
    "verl/experimental/reward_loop/reward_manager/naive.py": ("NaiveRewardManager.run_single",),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(checkout: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={checkout.as_posix()}", "-C", str(checkout), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _symbol_sources(path: Path, names: tuple[str, ...]) -> dict[str, str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    found: dict[str, str] = {}

    def visit(body: list[ast.stmt], prefix: str = "") -> None:
        for node in body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = f"{prefix}.{node.name}" if prefix else node.name
                if qualified in names:
                    segment = ast.get_source_segment(source, node)
                    if segment is None:
                        raise RuntimeError(f"cannot extract source for {qualified} in {path}")
                    found[qualified] = hashlib.sha256(segment.encode("utf-8")).hexdigest()
                if isinstance(node, ast.ClassDef):
                    visit(node.body, qualified)

    visit(tree.body)
    missing = set(names) - found.keys()
    if missing:
        raise RuntimeError(f"missing required symbols in {path}: {sorted(missing)}")
    return found


def inspect_checkout(checkout: str | Path) -> dict[str, Any]:
    root = Path(checkout).expanduser().resolve()
    missing = [name for name in REQUIRED_VERL_FILES if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"incomplete VERL checkout {root}: missing {missing}")
    symbols: dict[str, dict[str, str]] = {}
    for name, required_symbols in REQUIRED_VERL_SYMBOLS.items():
        symbols[name] = _symbol_sources(root / name, required_symbols)
    return {
        "repository": git_value(root, "config", "--get", "remote.origin.url"),
        "commit": git_value(root, "rev-parse", "HEAD"),
        "dirty": bool(git_value(root, "status", "--porcelain")),
        "interfaces": {name: sha256_file(root / name) for name in REQUIRED_VERL_FILES},
        "symbols": symbols,
    }


def verify_source_lock(checkout: str | Path, lock_path: str | Path) -> dict[str, Any]:
    """Verify the pinned checkout while allowing the runtime gate to remain closed."""
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    observed = inspect_checkout(checkout)
    for name in ("repository", "commit", "interfaces", "symbols"):
        if observed[name] != lock.get(name):
            raise RuntimeError(f"VERL {name} differs from lock")
    if observed["dirty"]:
        raise RuntimeError("VERL checkout is dirty")
    return observed


def verify_lock(checkout: str | Path, lock_path: str | Path) -> dict[str, Any]:
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    if lock.get("verified") is not True:
        raise RuntimeError("VERL lock is not verified")
    return verify_source_lock(checkout, lock_path)
