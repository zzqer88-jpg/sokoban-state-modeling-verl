"""Opt-in CPU integration: official processor + tiny random Qwen, no 4B weights.

JEV_QWEN_ASSETS=/path/to/tokenizer-and-processor-files pytest ...
No assets are downloaded by the test. This is a numerical smoke test, not SFT.
"""
import json
import os
from pathlib import Path

import pytest
import torch
from PIL import Image

from sokoban_state_modeling.jev.backend import QwenBackend
from sokoban_state_modeling.jev.config import Config
from sokoban_state_modeling.jev.models import BoardInput
from sokoban_state_modeling.jev.scoring import cell_kl
from sokoban_state_modeling.jev.training import save_checkpoint, load_checkpoint, checkpoint_metadata

ASSETS = os.environ.get("JEV_QWEN_ASSETS")
pytestmark = pytest.mark.skipif(not ASSETS, reason="Set JEV_QWEN_ASSETS for official processor integration")


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    from transformers import AutoProcessor, Qwen3_5Config, Qwen3_5ForConditionalGeneration
    root = tmp_path_factory.mktemp("tiny-qwen")
    processor = AutoProcessor.from_pretrained(ASSETS, min_pixels=1024, max_pixels=4096)
    config = Qwen3_5Config(
        text_config=dict(vocab_size=len(processor.tokenizer), hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                         layer_types=["linear_attention", "full_attention"],
                         linear_key_head_dim=8, linear_value_head_dim=8,
                         linear_num_key_heads=2, linear_num_value_heads=4,
                         max_position_embeddings=4096, pad_token_id=processor.tokenizer.pad_token_id,
                         rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                                          "partial_rotary_factor": 1., "mrope_section": [1, 1, 2]}),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
                           out_hidden_size=32, patch_size=16, temporal_patch_size=2,
                           spatial_merge_size=2, num_position_embeddings=16),
        image_token_id=processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        video_token_id=processor.tokenizer.convert_tokens_to_ids("<|video_pad|>"),
        vision_start_token_id=processor.tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        vision_end_token_id=processor.tokenizer.convert_tokens_to_ids("<|vision_end|>"),
    )
    torch.manual_seed(9)
    model = Qwen3_5ForConditionalGeneration(config)
    model.save_pretrained(root)
    processor.save_pretrained(root)
    return root


def test_real_processor_qwen_lora_gradients_and_checkpoint(tiny_checkpoint, tmp_path):
    config = Config(model_path=str(tiny_checkpoint), device="cpu", dtype="float32",
                    min_pixels=1024, max_pixels=4096, lora_rank=2, lora_alpha=4,
                    finetune_mode="lora", train_merger=True, gradient_checkpointing=True)
    backend = QwenBackend.load(config, trainable=True)
    board = BoardInput.create(Path("unused"), 2, 11)
    # Choose different token lengths to exercise actual left padding.
    questions = (board.questions[0], board.questions[10])
    image = Image.new("RGB", (80, 80), "white")
    backend.model.eval()
    batched = backend.logits(board, questions, image)
    single = backend.logits(board, questions[:1], image)
    torch.testing.assert_close(batched[0], single[0], atol=2e-5, rtol=2e-4)
    black = backend.logits(board, questions[:1], Image.new("RGB", (80, 80)))
    assert not torch.allclose(single, black, atol=1e-7, rtol=1e-7)
    backend.model.train()
    logits = backend.logits(board, questions, image)
    loss = cell_kl(logits, torch.eye(7)[:2]).mean()
    loss.backward()
    grads = {name: p.grad for name, p in backend.model.named_parameters() if p.requires_grad}
    assert any("lora_B" in name and grad is not None and grad.abs().sum() > 0 for name, grad in grads.items())
    assert any("merger" in name and grad is not None and grad.abs().sum() > 0 for name, grad in grads.items())
    assert all("lora_" in name or "modules_to_save" in name for name in grads)
    optimizer = torch.optim.AdamW([p for p in backend.model.parameters() if p.requires_grad], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
    # A single synthetic optimizer step checks serializing optimizer state. It does
    # not fit a dataset or use a pretrained model.
    optimizer.step()
    scheduler.step()
    backend.model.eval()
    reference = backend.logits(board, questions, image).detach()
    dest = tmp_path / "checkpoint"
    save_checkpoint(backend, optimizer, scheduler, dest, {"completed_epochs": 1})
    restored = load_checkpoint(dest, device="cpu")
    torch.testing.assert_close(restored.logits(board, questions, image), reference, atol=1e-6, rtol=1e-5)
    # Resume must retain optimizer/RNG state and trainable adapter/merger weights.
    resumed = load_checkpoint(dest, trainable=True, device="cpu")
    state = torch.load(dest / "training_state.pt", map_location="cpu", weights_only=False)
    resumed_opt = torch.optim.AdamW([p for p in resumed.model.parameters() if p.requires_grad], lr=1e-4)
    resumed_opt.load_state_dict(state["optimizer"])
    assert resumed_opt.state_dict()["state"]
    assert torch.equal(state["torch_rng"], state["torch_rng"].cpu())
    (dest / "metadata.json").write_text("{}")
    with pytest.raises(ValueError, match="mismatch"):
        checkpoint_metadata(dest)


def test_tiny_trainer_epoch_resume_matches_uninterrupted(tiny_checkpoint, tmp_path, monkeypatch):
    """Exercise training orchestration on random test weights, not a pretrained run."""
    from dataclasses import replace
    import sokoban_state_modeling.jev.training as training
    from sokoban_state_modeling.sft.jev_export import export_jev
    rows = [
        {"split": "train", "current_state": {"grid": ["P_"]}},
        {"split": "train", "current_state": {"grid": ["P__"]}},
        {"split": "validation", "current_state": {"grid": ["#P", "#_"]}},
    ]
    source = tmp_path / "source.jsonl"
    source.write_text("\n".join(map(json.dumps, rows)))
    export_jev([source], tmp_path / "data")
    cfg = Config(model_path=str(tiny_checkpoint), data_path=str(tmp_path / "data/records.jsonl"),
                 output_dir=str(tmp_path / "full"), device="cpu", dtype="float32", epochs=2,
                 min_pixels=1024, max_pixels=4096, lora_rank=2, lora_alpha=4,
                 finetune_mode="lora", gradient_checkpointing=False, cell_batch_size=2)
    training.train(cfg)
    original_save = training.save_checkpoint

    class SimulatedInterruption(Exception):
        pass

    def save_then_interrupt(*args, **kwargs):
        original_save(*args, **kwargs)
        raise SimulatedInterruption()

    monkeypatch.setattr(training, "save_checkpoint", save_then_interrupt)
    with pytest.raises(SimulatedInterruption):
        training.train(replace(cfg, output_dir=str(tmp_path / "partial")))
    monkeypatch.setattr(training, "save_checkpoint", original_save)
    training.train(replace(cfg, output_dir=str(tmp_path / "resumed")), tmp_path / "partial/epoch_001")
    from safetensors.torch import load_file
    full = load_file(tmp_path / "full/epoch_002/adapter/adapter_model.safetensors")
    resumed = load_file(tmp_path / "resumed/epoch_002/adapter/adapter_model.safetensors")
    assert set(full) == set(resumed)
    for key in full:
        torch.testing.assert_close(full[key], resumed[key], atol=1e-7, rtol=1e-6)
    assert (tmp_path / "resumed/best.json").exists()


def test_full_finetuning_preserves_all_weights_without_adapters(tiny_checkpoint, tmp_path):
    cfg = Config(model_path=str(tiny_checkpoint), device="cpu", dtype="float32",
                 finetune_mode="full", min_pixels=1024, max_pixels=4096,
                 gradient_checkpointing=True)
    backend = QwenBackend.load(cfg, trainable=True)
    assert all(p.requires_grad for p in backend.model.parameters())
    assert not any("lora_" in name for name, _ in backend.model.named_parameters())
    board = BoardInput.create(Path("unused"), 1, 1)
    image = Image.new("RGB", (64, 64), "red")
    logits = backend.logits(board, board.questions, image)
    cell_kl(logits, torch.eye(7)[3:4]).mean().backward()
    assert any("visual" in name and p.grad is not None and p.grad.abs().sum() > 0
               for name, p in backend.model.named_parameters())
    assert backend.model.lm_head.weight.grad is not None
    optimizer = torch.optim.AdamW(backend.model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
    backend.model.eval()
    reference = backend.logits(board, board.questions, image).detach()
    dest = tmp_path / "full_checkpoint"
    save_checkpoint(backend, optimizer, scheduler, dest, {"completed_epochs": 0})
    assert (dest / "model").is_dir() and not (dest / "adapter").exists()
    restored = load_checkpoint(dest, device="cpu")
    torch.testing.assert_close(restored.logits(board, board.questions, image), reference)
