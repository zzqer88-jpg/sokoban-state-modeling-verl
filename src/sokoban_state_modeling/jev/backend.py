"""Differentiable original-LM-head candidate readout, without SGLang."""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any
import torch
from PIL import Image
from .config import Config
from .models import BoardInput, CellQuestion
from .prompts import PromptCompiler


class QwenBackend:
    def __init__(self, model: Any, processor: Any, config: Config):
        self.model, self.processor, self.config = model, processor, config
        self.compiler = PromptCompiler(processor, config.max_length)

    @classmethod
    def load(cls, config: Config, *, trainable: bool = False,
             checkpoint: Path | None = None, initialize_on_cpu: bool = False) -> QwenBackend:
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        # FSDP keeps FP32 master shards/optimizer states and casts compute to BF16.
        dtype = torch.float32 if initialize_on_cpu else getattr(torch, config.dtype)
        processor = AutoProcessor.from_pretrained(
            str(checkpoint / "processor") if checkpoint else config.model_path,
            **({} if checkpoint else {"revision": config.revision}),
            min_pixels=config.min_pixels, max_pixels=config.max_pixels,
        )
        full_checkpoint = checkpoint and config.finetune_mode == "full"
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(checkpoint / "model") if full_checkpoint else config.model_path,
            revision=None if full_checkpoint else config.revision,
            dtype=dtype, attn_implementation="sdpa"
        )
        model.requires_grad_(False)
        if config.finetune_mode == "full":
            model.requires_grad_(trainable)
        elif checkpoint:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(checkpoint / "adapter"), is_trainable=trainable)
        elif trainable:
            from peft import LoraConfig, get_peft_model
            # Match language modules explicitly: never adapt the visual tower by accident.
            targets = [name for name, module in model.named_modules()
                       if ".language_model." in name and isinstance(module, torch.nn.Linear)
                       and name.rsplit(".", 1)[-1] in config.lora_targets]
            if not targets:
                raise ValueError("No matching language LoRA modules in this checkpoint")
            present = {name.rsplit(".", 1)[-1] for name in targets}
            if present != set(config.lora_targets):
                raise ValueError(f"Unmatched LoRA targets: {set(config.lora_targets) - present}")
            modules_to_save = None
            if config.train_merger:
                mergers = [name for name, _ in model.named_modules() if name.endswith("visual.merger")]
                if len(mergers) != 1:
                    raise ValueError("Cannot uniquely identify visual merger")
                modules_to_save = mergers
            model = get_peft_model(model, LoraConfig(
                r=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=0.0,
                target_modules="^(?:" + "|".join(re.escape(name) for name in targets) + ")$",
                modules_to_save=modules_to_save, bias="none",
            ))
        model.to("cpu" if initialize_on_cpu else config.device)
        if trainable and config.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train(trainable)
        if trainable and not any(p.requires_grad for p in model.parameters()):
            raise ValueError("No trainable parameters")
        return cls(model, processor, config)

    def logits(self, board: BoardInput, questions: tuple[CellQuestion, ...], image: Image.Image) -> torch.Tensor:
        inputs = self.compiler.batch(board, questions, image)
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                  for key, value in inputs.items()}
        # Left padding makes the last position valid for every branch. Restrict the
        # LM projection to this position to avoid [batch, sequence, vocabulary] RAM.
        output = self.model(**inputs, use_cache=False, logits_to_keep=1, return_dict=True)
        if output.logits.ndim != 3 or output.logits.shape[1] != 1:
            raise ValueError("Model must support last-position logits_to_keep=1")
        labels = torch.tensor([idx for _, idx in self.compiler.labels], device=device)
        return output.logits[:, 0].index_select(-1, labels).float()
