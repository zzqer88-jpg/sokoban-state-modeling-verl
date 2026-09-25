"""Single-node FSDP2 full SFT. CUDA/NCCL execution is verified on the server."""
from __future__ import annotations

from datetime import timedelta
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist

from .backend import QwenBackend
from .config import Config
from .dataset import BoardDataset, file_sha256
from .models import CLASSES
from .scoring import cell_kl, log_probabilities


def step_assignments(order: list[int], world_size: int, boards_per_rank: int):
    """Yield [rank][slot] indices. None denotes zero-weight padding, never a repeat."""
    if world_size < 1 or boards_per_rank < 1 or not order:
        raise ValueError("Nonempty order and positive batch/world sizes required")
    size = world_size * boards_per_rank
    for start in range(0, len(order), size):
        batch = order[start:start + size]
        yield [[batch[slot * world_size + rank] if slot * world_size + rank < len(batch) else None
                for slot in range(boards_per_rank)] for rank in range(world_size)], len(batch)


def weighted_micro_loss(logits: torch.Tensor, target: torch.Tensor, *, board_cells: int,
                        real_boards: int, world_size: int, valid: bool) -> torch.Tensor:
    # FSDP averages gradients across ranks: multiply by world_size to recover
    # the mean over real boards, not padded slots or ranks.
    scale = world_size / (real_boards * board_cells) if valid else 0.
    return cell_kl(logits, target).sum() * scale


def maximum_rounds(local: int, device: torch.device) -> int:
    count = torch.tensor(local, device=device, dtype=torch.int64)
    dist.all_reduce(count, op=dist.ReduceOp.MAX)
    return int(count)


def rank0_call(function):
    """Broadcast rank-zero I/O errors before other ranks enter more collectives."""
    result = [None, None]
    if dist.get_rank() == 0:
        try:
            result[0] = function()
        except Exception as exc:
            result[1] = f"{type(exc).__name__}: {exc}"
    dist.broadcast_object_list(result, src=0)
    if result[1]:
        raise RuntimeError(result[1])
    return result[0]


def wrap_fsdp(backend: QwenBackend, device: torch.device) -> None:
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5VisionBlock
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    policy = MixedPrecisionPolicy(param_dtype=getattr(torch, backend.config.dtype), reduce_dtype=torch.float32)
    blocks = [module for module in backend.model.modules()
              if isinstance(module, (Qwen3_5DecoderLayer, Qwen3_5VisionBlock))]
    if not blocks:
        raise ValueError("No Qwen decoder/vision blocks found for FSDP wrapping")
    for module in blocks:
        fully_shard(module, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
    fully_shard(backend.model, mesh=mesh, mp_policy=policy, reshard_after_forward=True)


def clip_sharded_gradients(model: torch.nn.Module, max_norm: float, device: torch.device) -> float:
    from torch.distributed.tensor import DTensor, Shard
    total = torch.zeros((), device=device, dtype=torch.float64)
    shards = []
    for parameter in model.parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        if not isinstance(gradient, DTensor) or not any(isinstance(p, Shard) for p in gradient.placements):
            raise ValueError("Expected one-dimensional FSDP gradient shards")
        local = gradient.to_local()
        total += local.double().square().sum()
        shards.append(local)
    dist.all_reduce(total)
    norm = total.sqrt()
    if not torch.isfinite(norm):
        raise ValueError("Nonfinite distributed gradient norm")
    factor = (max_norm / (norm + 1e-6)).clamp(max=1.).float()
    for gradient in shards:
        gradient.mul_(factor.to(gradient.dtype))
    return float(norm)


def summarize_predictions(records: list[dict]) -> dict:
    matrix = np.zeros((7, 7), dtype=np.int64)
    exact = legal = 0
    entities = dict(player=0, box=0, goal=0)
    groups = {}
    for item in records:
        truth, predicted = np.asarray(item["truth"]), np.asarray(item["prediction"])
        np.add.at(matrix, (truth, predicted), 1)
        ok = int(np.array_equal(truth, predicted))
        exact += ok
        legal += int(np.isin(predicted, [5, 6]).sum() == 1 and
                     np.isin(predicted, [3, 4]).sum() == np.isin(predicted, [2, 4, 6]).sum())
        for key, values in (("player", [5, 6]), ("box", [3, 4]), ("goal", [2, 4, 6])):
            entities[key] += int(np.array_equal(np.isin(truth, values), np.isin(predicted, values)))
        group = groups.setdefault(item["size"], {"boards": 0, "exact": 0})
        group["boards"] += 1
        group["exact"] += ok
    per_class = {}
    for i, key in enumerate(CLASSES):
        support, selected = int(matrix[i].sum()), int(matrix[:, i].sum())
        precision = float(matrix[i, i] / selected) if selected else 0.
        recall = float(matrix[i, i] / support) if support else 0.
        per_class[key] = {"support": support, "precision": precision, "recall": recall,
                          "f1": 2*precision*recall/(precision+recall) if precision+recall else 0.}
    for group in groups.values():
        group["board_exact"] = group["exact"] / group["boards"]
    return {"boards": len(records), "board_exact": exact/len(records),
            "cell_accuracy": float(np.trace(matrix)/matrix.sum()), "confusion_matrix": matrix.tolist(),
            "structural_validity": legal/len(records), "per_class": per_class, "by_size": groups,
            "entity_exact": {k: v/len(records) for k, v in entities.items()}}


def validate_fsdp(backend: QwenBackend, data: BoardDataset, device: torch.device) -> dict:
    rank, world = dist.get_rank(), dist.get_world_size()
    backend.model.eval()
    records = []
    # All ranks must execute the same number of model forwards, including tails.
    with torch.no_grad():
        for assignments, _ in step_assignments(list(range(len(data))), world, 1):
            index = assignments[rank][0]
            example = data[index if index is not None else 0]
            image = example.image()
            questions = example.inputs.questions
            batches = math.ceil(len(questions)/backend.config.cell_batch_size) if index is not None else 0
            predictions = []
            for micro in range(maximum_rounds(batches, device)):
                start = micro * backend.config.cell_batch_size
                valid = index is not None and micro < batches
                selected = questions[start:start+backend.config.cell_batch_size] if valid else questions[:1]
                logits = backend.logits(example.inputs, selected, image)
                if valid:
                    predictions.extend(log_probabilities(logits).argmax(-1).cpu().tolist())
            if index is not None:
                records.append({"index": index, "size": f"{example.inputs.height}x{example.inputs.width}",
                                "truth": np.asarray(example.targets).argmax(-1).tolist(), "prediction": predictions})
    gathered = [None] * world if rank == 0 else None
    dist.gather_object(records, gathered, dst=0)
    def report():
        all_records = [item for subset in gathered for item in subset]
        if sorted(item["index"] for item in all_records) != list(range(len(data))):
            raise ValueError("Distributed validation coverage mismatch")
        return summarize_predictions(all_records)
    result = rank0_call(report)
    backend.model.train()
    return result


def save_fsdp(backend: QwenBackend, optimizer, scheduler, destination: Path, state: dict) -> None:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, get_model_state_dict, StateDictOptions
    from .training import versions
    rank = dist.get_rank()
    temporary = destination.parent / ("." + destination.name + ".partial")
    def prepare():
        if destination.exists() or temporary.exists():
            raise FileExistsError(f"Checkpoint path already exists: {destination} or {temporary}")
        temporary.mkdir()
    rank0_call(prepare)
    model_state, optim_state = get_state_dict(backend.model, optimizer)
    dcp.save({"model": model_state, "optimizer": optim_state}, checkpoint_id=temporary / "distributed")
    torch.save({"scheduler": scheduler.state_dict(), "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state()}, temporary / f"rank_{rank}.pt")
    # All ranks participate; only rank 0 receives full CPU tensors.
    full = get_model_state_dict(backend.model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    dist.barrier()
    def publish():
        backend.model.save_pretrained(temporary / "model", state_dict=full)
        backend.processor.save_pretrained(temporary / "processor")
        metadata = {"format_version": 1, "trainer": "fsdp2", "world_size": dist.get_world_size(),
                    "config": backend.config.to_dict(), "contract": backend.compiler.contract(),
                    "versions": versions(), "resolved_model_commit": getattr(backend.model.config, "_commit_hash", None),
                    **state}
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        checksums = {str(p.relative_to(temporary)): file_sha256(p) for p in temporary.rglob("*") if p.is_file()}
        (temporary / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")
        temporary.rename(destination)
    rank0_call(publish)


def restore_fsdp(backend: QwenBackend, optimizer, scheduler, checkpoint: Path) -> None:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
    model_state, optim_state = get_state_dict(backend.model, optimizer)
    state = {"model": model_state, "optimizer": optim_state}
    dcp.load(state, checkpoint_id=checkpoint / "distributed")
    set_state_dict(backend.model, optimizer, model_state_dict=state["model"], optim_state_dict=state["optimizer"])
    runtime = torch.load(checkpoint / f"rank_{dist.get_rank()}.pt", map_location="cpu", weights_only=False)
    scheduler.load_state_dict(runtime["scheduler"])
    random.setstate(runtime["python_rng"])
    np.random.set_state(runtime["numpy_rng"])
    torch.set_rng_state(runtime["torch_rng"])
    torch.cuda.set_rng_state(runtime["cuda_rng"])


def train_fsdp(config: Config, resume: Path | None = None) -> dict:
    from .training import seed_all, checkpoint_metadata, versions
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world < 2 or not torch.cuda.is_available() or config.finetune_mode != "full":
        raise ValueError("FSDP2 requires torchrun with >=2 CUDA GPUs and full SFT")
    if int(os.environ.get("LOCAL_WORLD_SIZE", str(world))) != world:
        raise ValueError("This launcher supports a single node only")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30), device_id=device)
    try:
        rank = dist.get_rank()
        train_data = BoardDataset(Path(config.data_path), "train")
        validation = BoardDataset(Path(config.data_path), "validation")
        root = Path(config.output_dir)
        metadata = rank0_call(lambda: checkpoint_metadata(resume)) if resume else None
        if metadata:
            old, new = dict(metadata["config"]), config.to_dict()
            for key in ("device", "output_dir", "data_path"):
                old.pop(key, None)
                new.pop(key, None)
            if (metadata.get("trainer") != "fsdp2" or metadata["world_size"] != world or old != new
                    or metadata["data_sha256"] != train_data.fingerprint or metadata["versions"] != versions()):
                raise ValueError("Resume requires same FSDP world size, config, data, and libraries")
            if metadata["completed_epochs"] >= config.epochs:
                raise ValueError("Checkpoint already completed this training schedule")
        def prepare_output():
            if root.exists() and any(root.iterdir()):
                raise FileExistsError("Use a new output directory")
            root.mkdir(parents=True, exist_ok=True)
            (root / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
        rank0_call(prepare_output)
        seed_all(config.seed)
        load_config = replace(config, device=str(device))
        if metadata and metadata.get("resolved_model_commit"):
            load_config = replace(load_config, revision=metadata["resolved_model_commit"])
        backend = QwenBackend.load(load_config, trainable=True, checkpoint=resume, initialize_on_cpu=True)
        backend.config = config
        if metadata and backend.compiler.contract() != metadata["contract"]:
            raise ValueError("Checkpoint compiler contract changed")
        wrap_fsdp(backend, device)
        optimizer = torch.optim.AdamW(backend.model.parameters(), lr=config.learning_rate)
        total = math.ceil(len(train_data)/(world*config.board_batch_size)) * config.epochs
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0., 1-step/total))
        start_epoch, step, best, best_path = 0, 0, -1., None
        seed_all(config.seed + rank)
        if resume:
            restore_fsdp(backend, optimizer, scheduler, resume)
            start_epoch, step = metadata["completed_epochs"], metadata["step"]
            best, best_path = metadata["best_board_exact"], metadata.get("best_checkpoint")
        for epoch in range(start_epoch, config.epochs):
            order = list(range(len(train_data)))
            random.Random(config.seed + epoch).shuffle(order)
            epoch_loss = 0.
            for assignments, real_boards in step_assignments(order, world, config.board_batch_size):
                optimizer.zero_grad(set_to_none=True)
                local_loss = torch.zeros((), device=device)
                for index in assignments[rank]:
                    example = train_data[index if index is not None else 0]
                    image = example.image()
                    count = len(example.inputs.questions)
                    rounds = math.ceil(count/config.cell_batch_size) if index is not None else 0
                    for micro in range(maximum_rounds(rounds, device)):
                        start = micro * config.cell_batch_size
                        valid = index is not None and micro < rounds
                        questions = (example.inputs.questions[start:start+config.cell_batch_size]
                                     if valid else example.inputs.questions[:1])
                        targets = example.targets[start:start+len(questions)] if valid else example.targets[:1]
                        logits = backend.logits(example.inputs, questions, image)
                        loss = weighted_micro_loss(logits, torch.tensor(targets, device=device),
                                                   board_cells=count, real_boards=real_boards,
                                                   world_size=world, valid=valid)
                        loss.backward()
                        local_loss += loss.detach()
                norm = clip_sharded_gradients(backend.model, config.max_grad_norm, device)
                optimizer.step()
                scheduler.step()
                dist.all_reduce(local_loss)
                step_loss = float(local_loss / world)
                epoch_loss += step_loss * real_boards
                step += 1
                if rank == 0 and (step == 1 or step % 10 == 0):
                    print(json.dumps({"epoch": epoch+1, "step": step, "kl": step_loss, "grad_norm": norm,
                                      "real_boards": real_boards}), flush=True)
            report = validate_fsdp(backend, validation, device)
            path = root / f"epoch_{epoch+1:03d}"
            if report["board_exact"] > best:
                best, best_path = report["board_exact"], str(path.resolve())
            state = {"completed_epochs": epoch+1, "step": step, "train_kl": epoch_loss/len(train_data),
                     "validation": report, "best_board_exact": best, "best_checkpoint": best_path,
                     "data_sha256": train_data.fingerprint}
            save_fsdp(backend, optimizer, scheduler, path, state)
            def publish_report():
                (root / "best.json").write_text(json.dumps({"checkpoint": best_path, "board_exact": best}) + "\n")
                (root / f"epoch_{epoch+1:03d}.json").write_text(json.dumps(state, indent=2) + "\n")
            rank0_call(publish_report)
        return {"output_dir": str(root), "completed_epochs": config.epochs, "best_board_exact": best}
    finally:
        dist.destroy_process_group()
