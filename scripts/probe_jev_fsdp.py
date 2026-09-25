"""Opt-in multi-GPU numerical probe using random tiny Qwen, never 4B weights.

Run with torchrun on the server. --assets points to official processor/tokenizer
files; --work-dir must be a new shared local directory visible to all ranks.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from sokoban_state_modeling.jev.config import Config
from sokoban_state_modeling.jev.distributed import rank0_call
from sokoban_state_modeling.jev.training import train
from sokoban_state_modeling.sft.jev_export import export_jev


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Run this opt-in probe on the GPU server")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    try:
        def prepare():
            from transformers import AutoProcessor, Qwen3_5Config, Qwen3_5ForConditionalGeneration
            args.work_dir.mkdir(parents=True, exist_ok=False)
            processor = AutoProcessor.from_pretrained(args.assets, min_pixels=1024, max_pixels=4096)
            model_config = Qwen3_5Config(
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
            torch.manual_seed(7)
            Qwen3_5ForConditionalGeneration(model_config).save_pretrained(args.work_dir / "tiny_model")
            processor.save_pretrained(args.work_dir / "tiny_model")
            records = [{"split": "train", "current_state": {"grid": ["P" + "_"*i]}}
                       for i in range(1, 14)]
            records += [{"split": "validation", "current_state": {"grid": ["#P", "#_"]}},
                        {"split": "validation", "current_state": {"grid": ["##P", "___"]}}]
            source = args.work_dir / "source.jsonl"
            source.write_text("\n".join(map(json.dumps, records)))
            export_jev([source], args.work_dir / "data")
        rank0_call(prepare)
    finally:
        dist.destroy_process_group()
    cfg = Config(model_path=str(args.work_dir / "tiny_model"), data_path=str(args.work_dir / "data/records.jsonl"),
                 output_dir=str(args.work_dir / "run"), distributed="fsdp2", finetune_mode="full",
                 dtype="bfloat16", epochs=2, cell_batch_size=2, min_pixels=1024, max_pixels=4096)
    train(cfg)
    # Resume epoch one with the original two-epoch schedule into a separate output.
    from dataclasses import replace
    train(replace(cfg, output_dir=str(args.work_dir / "resumed")), args.work_dir / "run/epoch_001")
    if int(os.environ["RANK"]) == 0:
        from transformers import Qwen3_5ForConditionalGeneration
        reference = Qwen3_5ForConditionalGeneration.from_pretrained(args.work_dir / "run/epoch_002/model")
        restored = Qwen3_5ForConditionalGeneration.from_pretrained(args.work_dir / "resumed/epoch_002/model")
        max_difference = 0.
        for key, value in reference.state_dict().items():
            other = restored.state_dict()[key]
            torch.testing.assert_close(value, other, atol=1e-4, rtol=1e-3)
            max_difference = max(max_difference, float((value-other).abs().max()))
        report = {"status": "completed", "scope": "tiny random Qwen on actual GPUs, not 4B training",
                  "world_size": int(os.environ["WORLD_SIZE"]), "resume_completed": True,
                  "resume_max_weight_difference": max_difference}
        (args.work_dir / "probe_result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
