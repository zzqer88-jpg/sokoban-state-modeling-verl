"""Full/LoRA training with board-normalized loss and an optional FSDP2 route."""
from __future__ import annotations
import json
import math
import os
import random
import shutil
import tempfile
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from .backend import QwenBackend
from .config import Config
from .dataset import BoardDataset, file_sha256
from .evaluation import evaluate_dataset
from .scoring import cell_kl
from .service import EvaluationService


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def versions() -> dict:
    return {name: version(name) for name in ("torch", "transformers", "peft", "Pillow")}


def save_checkpoint(backend: QwenBackend, optimizer: torch.optim.Optimizer,
                    scheduler: object, destination: Path, state: dict) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=destination.parent))
    try:
        weights_dir = "model" if backend.config.finetune_mode == "full" else "adapter"
        backend.model.save_pretrained(temporary / weights_dir)
        backend.processor.save_pretrained(temporary / "processor")
        metadata = {"format_version": 1, "config": backend.config.to_dict(),
                    "contract": backend.compiler.contract(), "versions": versions(),
                    "resolved_model_commit": getattr(backend.model.config, "_commit_hash", None),
                    **state}
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                   temporary / "training_state.pt")
        artifacts = {str(path.relative_to(temporary)): file_sha256(path)
                     for path in temporary.rglob("*") if path.is_file()}
        (temporary / "checksums.json").write_text(json.dumps(artifacts, indent=2) + "\n")
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise


def checkpoint_metadata(checkpoint: Path) -> dict:
    checksums = json.loads((checkpoint / "checksums.json").read_text())
    for name, expected in checksums.items():
        path = (checkpoint / name).resolve()
        if not path.is_relative_to(checkpoint.resolve()) or file_sha256(path) != expected:
            raise ValueError(f"Checkpoint artifact mismatch: {name}")
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    if metadata.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    return metadata


def load_checkpoint(checkpoint: Path, *, trainable: bool = False,
                    device: str | None = None) -> QwenBackend:
    metadata = checkpoint_metadata(checkpoint)
    config = Config.from_dict(metadata["config"])
    if device:
        config = replace(config, device=device)
    if metadata.get("resolved_model_commit"):
        config = replace(config, revision=metadata["resolved_model_commit"])
    backend = QwenBackend.load(config, trainable=trainable, checkpoint=checkpoint)
    if backend.compiler.contract() != metadata["contract"]:
        raise ValueError("Checkpoint prompt/token contract changed")
    return backend


def train(config: Config, resume: Path | None = None) -> dict:
    if config.distributed == "fsdp2":
        from .distributed import train_fsdp
        return train_fsdp(config, resume)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This reference trainer is single-process; do not launch with torchrun")
    train_data = BoardDataset(Path(config.data_path), "train")
    validation = BoardDataset(Path(config.data_path), "validation")
    root = Path(config.output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use a new output_dir (also when resuming)")
    metadata = checkpoint_metadata(resume) if resume else None
    if metadata:
        if metadata.get("trainer") == "fsdp2":
            raise ValueError("Resume this checkpoint with the FSDP2 trainer")
        if metadata["data_sha256"] != train_data.fingerprint:
            raise ValueError("Resume dataset changed")
        old, new = dict(metadata["config"]), config.to_dict()
        for name in ("device", "output_dir", "data_path"):
            old.pop(name, None)
            new.pop(name, None)
        if old != new or metadata["versions"] != versions():
            raise ValueError("Resume requires matching training configuration and libraries")
        if metadata["completed_epochs"] >= config.epochs:
            raise ValueError("Checkpoint already completed the configured epochs")
    seed_all(config.seed)
    backend = (load_checkpoint(resume, trainable=True, device=config.device)
               if resume else QwenBackend.load(config, trainable=True))
    # Keep the requested configuration separate from resolved_model_commit.
    if resume:
        backend.config = config
    parameters = [p for p in backend.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate)
    total_steps = math.ceil(len(train_data) / config.board_batch_size) * config.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0., 1-step/total_steps))
    start_epoch, step = 0, 0
    best = -1.
    best_checkpoint = None
    if resume:
        # Only load optimizer/RNG pickle from your own locally trusted checkpoint.
        state = torch.load(resume / "training_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None:
            if not torch.cuda.is_available():
                raise ValueError("CUDA training resume requires CUDA RNG support")
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        start_epoch, step, best = metadata["completed_epochs"], metadata["step"], metadata["best_board_exact"]
        best_checkpoint = metadata.get("best_checkpoint")
    if start_epoch >= config.epochs:
        raise ValueError("Checkpoint already completed the configured epochs")
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    history = []
    for epoch in range(start_epoch, config.epochs):
        backend.model.train()
        order = list(range(len(train_data)))
        random.shuffle(order)
        epoch_loss = 0.
        for start in range(0, len(order), config.board_batch_size):
            indices = order[start:start + config.board_batch_size]
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.
            for index in indices:
                example = train_data[index]
                image = example.image()
                count = len(example.inputs.questions)
                for offset in range(0, count, config.cell_batch_size):
                    questions = example.inputs.questions[offset:offset + config.cell_batch_size]
                    logits = backend.logits(example.inputs, questions, image)
                    target = torch.tensor(example.targets[offset:offset + len(questions)], device=logits.device)
                    # Each board contributes exactly 1 / len(indices), irrespective
                    # of cell count and the final partial microbatch size.
                    loss = cell_kl(logits, target).sum() / (count * len(indices))
                    loss.backward()
                    batch_loss += float(loss.detach())
            if not any(p.grad is not None for p in parameters):
                raise ValueError("No trainable gradients; graph may be detached")
            norm = torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            step += 1
            epoch_loss += batch_loss * len(indices)
            if step == 1 or step % 10 == 0:
                print(json.dumps({"epoch": epoch+1, "step": step, "kl": batch_loss,
                                  "grad_norm": float(norm)}), flush=True)
        report = evaluate_dataset(EvaluationService(backend, config.cell_batch_size), validation)
        improved = report["board_exact"] > best
        best = max(best, report["board_exact"])
        path = root / f"epoch_{epoch+1:03d}"
        if improved:
            best_checkpoint = str(path.resolve())
        entry = {"completed_epochs": epoch+1, "step": step, "train_kl": epoch_loss/len(train_data),
                 "validation": report, "best_board_exact": best, "best_checkpoint": best_checkpoint,
                 "data_sha256": train_data.fingerprint}
        save_checkpoint(backend, optimizer, scheduler, path, entry)
        (root / "best.json").write_text(json.dumps({"checkpoint": best_checkpoint, "board_exact": best}) + "\n")
        history.append(entry)
        (root / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return {"output_dir": str(root), "completed_epochs": config.epochs, "best_board_exact": best}
