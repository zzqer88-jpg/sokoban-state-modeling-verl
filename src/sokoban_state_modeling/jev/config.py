"""Validated configuration for local and single-node FSDP2 training."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
from pathlib import Path


@dataclass(frozen=True)
class Config:
    model_path: str
    data_path: str = "data/jev_perception_v1/records.jsonl"
    output_dir: str = "outputs/jev_visual"
    revision: str | None = None
    device: str = "cuda"
    dtype: str = "bfloat16"
    finetune_mode: str = "full"
    distributed: str = "none"
    epochs: int = 1
    board_batch_size: int = 1
    cell_batch_size: int = 4
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    max_length: int = 4096
    min_pixels: int = 65536
    max_pixels: int = 262144
    seed: int = 42
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    train_merger: bool = False
    gradient_checkpointing: bool = True

    def __post_init__(self) -> None:
        for name in ("epochs", "board_batch_size", "cell_batch_size", "max_length", "min_pixels", "max_pixels", "lora_rank", "lora_alpha"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "max_grad_norm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not self.model_path or self.dtype not in {"float32", "bfloat16"}:
            raise ValueError("Provide a model path and dtype float32 or bfloat16")
        if self.finetune_mode not in {"full", "lora"}:
            raise ValueError("finetune_mode must be full or lora")
        if self.distributed not in {"none", "fsdp2"}:
            raise ValueError("distributed must be none or fsdp2")
        if self.distributed == "fsdp2" and self.finetune_mode != "full":
            raise ValueError("FSDP2 currently supports full SFT only")
        if self.min_pixels > self.max_pixels or not self.lora_targets:
            raise ValueError("Invalid pixel budget or empty LoRA targets")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in ("train_merger", "gradient_checkpointing"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")

    def to_dict(self) -> dict:
        values = asdict(self)
        values["lora_targets"] = list(self.lora_targets)
        return values

    @classmethod
    def from_dict(cls, values: dict) -> Config:
        values = dict(values)
        if "lora_targets" in values:
            values["lora_targets"] = tuple(values["lora_targets"])
        return cls(**values)

    @classmethod
    def load(cls, path: Path) -> Config:
        import yaml
        return cls.from_dict(yaml.safe_load(path.read_text()))
