# Visual Jev-style probability training

This implementation lives in `src/sokoban_state_modeling/jev/`. It uses the full
Qwen3.5 multimodal model and its original LM head, reads seven candidate scores,
and trains with KL(target || prediction). All model-visible prompts are English.
It does not use SGLang, Pev's ChoicePolicy, PPO, or generated probability strings.

## Install

Use a separate server environment from the existing VERL environment:

```bash
python -m pip install -e '.[jev,test]'
```

The extra pins Transformers 5.17.0 and PEFT 0.21.0. Install a matched PyTorch /
torchvision pair for your server's CUDA runtime. The CPU integration checks used
PyTorch 2.14.0 and torchvision 0.29.0; CUDA and the actual 4B checkpoint have not
been validated here. No model weights are included in this repository.

## Convert the existing SFT dataset

Run from the repository root on the server containing the original data:

```bash
sokoban-export-jev-sft \
  --input data/sftcoldstartv11/sft/*.jsonl \
  --output data/jev_perception_sampled_200 \
  --train-cell-budget 16 \
  --max-train-boards 200
```

Pass all split files together. Only current states are reused. Original action,
reasoning and prediction targets are excluded. Output is `records.jsonl`, PNGs
under `images/`, and `audit.json`. Existing destinations are never overwritten.
The command defaults to 200 unique training boards and 16 questions per training
board, for at most 3,200 supervised training cells after deduplication. It first
includes the player, every box and every target, then samples up to five local
neighbors, boundary/interior walls, near/far floors, and spatially balanced
cells from all four board regions. Sampling is deterministic for each state and
its exact coordinates are stored in the record.

Validation and test boards retain every coordinate, so full-board exact accuracy
remains a valid evaluation metric. The 200-board cap applies only to the train
split. The audit records the available and selected unique training-board counts,
cell budget, and sampler version under `train_sampling`.
The converter checks rotations/reflections of terrain maps for cross-split
leakage; it does not infer every possible procedural-generation family.

One exported board contains one `input` and a separate `target.answers` mapping.
Each coordinate has seven probabilities in canonical order:

```text
wall, floor, goal, box, box_on_goal, player, player_on_goal
```

The engine provides one-hot target distributions. Neither targets nor provenance
are passed to the model. The loader aligns distributions by class key, validates
all coordinate questions and rejects mismatched image hashes.

```bash
sokoban-jev audit --data data/jev_perception_v1/records.jsonl --split train
sokoban-jev audit --data data/jev_perception_v1/records.jsonl --split validation
sokoban-jev audit --data data/jev_perception_v1/records.jsonl --split test
```

`audit` checks dataset contracts and image hashes without loading Qwen. It does
not measure recognition quality. Audit every split before training.

## Train (only when ready to start training)

The server default is **full SFT**, with a starting learning rate of `1e-5`
(a configurable baseline, not a tuned optimum). All multimodal parameters are
trainable.

Edit `configs/jev_visual_sft.yaml`: set `model_path` to the full multimodal
checkpoint, check the data path, and use a new output directory. Inspect image
preprocessing before choosing the final pixel budget.

```bash
sokoban-jev train --config configs/jev_visual_sft.yaml
```

This is a single-process reference trainer, not a distributed VERL trainer.
Do not launch it with torchrun. Set `finetune_mode: full` to train all parameters of the complete multimodal
model (vision encoder, merger, language backbone, and LM head). This mode does
not create LoRA adapters; its checkpoints store complete weights under `model/`.
LoRA-specific settings have no effect in full mode. Choose the mode based on the
server training plan, not the local CPU environment.

With `finetune_mode: lora`, it uses LoRA on the language model's full-attention
q/k/v/o projections; it does not adapt all hybrid linear-attention
projections. The visual tower remains present but frozen. `train_merger: true`
additionally trains and checkpoints `visual.merger` via PEFT modules_to_save;
the loader fails if that module cannot be identified uniquely.

`board_batch_size` is the number of boards per optimizer update;
`cell_batch_size` controls the number of coordinate branches per forward pass.
Every branch sees the whole image. Cell microbatches accumulate loss divided by
the complete board's cell count, then by the number of boards in the update.
A partial final batch uses its actual board count. There is no silent truncation.

Outputs include epoch checkpoints, training history and `best.json`, selected by
validation whole-board exact accuracy. Checkpoints save full model weights or the adapter, plus the processor,
configuration, label/template contract, software versions, optimizer, scheduler
and RNG states, with artifact checksums. For LoRA, the base model is referenced, not copied; keep local base weights
unchanged, or use an immutable Hub revision. Full-mode checkpoints contain the
complete model and can be loaded without the original base checkpoint.

Resume from a completed epoch into a **new** output directory:

```bash
sokoban-jev train --config configs/jev_visual_sft.yaml \
  --resume outputs/previous_run/epoch_001
```

For resume, update `output_dir` in the config. Dataset content, training settings
(including the original total epoch count) and recorded libraries must match.
Resume is at epoch boundaries; an interrupted incomplete epoch is rerun from the
last complete checkpoint. This is not an interface for extending a completed
one-epoch run into a different schedule. Load optimizer/RNG pickle files only
from your own trusted checkpoints. If the best checkpoint precedes the resume
point, keep its original directory; `best.json` can reference that prior run.

## Evaluate and predict

Use the epoch directory referenced by `best.json`:

```bash
sokoban-jev evaluate \
  --checkpoint outputs/jev_visual_run1/epoch_001 \
  --data data/jev_perception_v1/records.jsonl --split test \
  --output outputs/jev_test

sokoban-jev predict \
  --checkpoint outputs/jev_visual_run1/epoch_001 \
  --image /absolute/path/board.png --height 6 --width 6 \
  --output outputs/board_prediction.json
```

`--device cpu` and `--cell-batch-size` are available for evaluation/prediction.
Prediction returns a Jev-style `answers` mapping (choice + all seven probabilities)
and the assembled character grid. The caller provides full-board dimensions;
arbitrary screenshot detection/cropping is not implemented. Probabilities are
conditional on these seven options, not calibrated correctness estimates.

Evaluation reports raw board exact accuracy, cell accuracy, per-class metrics
with support, player/box/goal location-set exactness, structural validity and
size groups. It saves up to 20 error PNGs with incorrect cells marked in red.
No rule-based output correction is applied. For optional soft target datasets,
accuracy uses target argmax; the current engine targets are exact one-hot labels.

The local service uses ordinary branch batches and repeats image preprocessing /
encoding; it does not promise shared visual compute or the N+1 warmup behavior
of openjev-sglang. Exported request_groups preserve the future service's 64-question
limit but do not restrict local model batch sizes.

## Verification

```bash
python -m pytest
# Optional: official tokenizer/processor assets, no pretrained weights required.
JEV_QWEN_ASSETS=/path/to/Qwen3.5-4B-processor-files \
  python -m pytest tests/test_jev_qwen_integration.py
```

The integration test creates a tiny random Qwen with both linear and full
attention, uses the official multimodal processor, checks candidate probabilities,
LoRA/merger and full-model gradients, checkpoint round trips, and epoch-resume
equivalence using tiny synthetic fixtures. It is a numerical test, not a
Sokoban training experiment. No assets are automatically downloaded by tests.
Actual 4B GPU memory, throughput, training convergence and remote data conversion
remain to be checked on the server. The separate VERL lock is not marked verified
by these tests.

## Single node, 8 H800 GPUs: FSDP2 full SFT

Use `configs/jev_visual_sft_8xh800.yaml`, not the single-process example. Set
`model_path` and the data/output paths. The implementation wraps Qwen decoder
and vision blocks with PyTorch FSDP2 and shards remaining root parameters.
All multimodal parameters are trainable. Compute is BF16; parameter/optimizer
shards and gradient reduction are FP32. There is no SGLang dependency.

The current development environment uses PyTorch 2.14.0. The FSDP2/NCCL path
**has not run on this CPU-only development machine**. Use a matching CUDA
PyTorch/torchvision installation on the server and run the opt-in numerical
probe before the 4B experiment:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=8 \
  scripts/probe_jev_fsdp.py \
  --assets /absolute/path/to/Qwen3.5-4B \
  --work-dir outputs/jev_fsdp_probe
```

The probe loads only the tokenizer/processor from `--assets`, creates a tiny
random model, and checks a distributed run plus epoch resume against the
uninterrupted final weights. It includes variable cell counts, fewer validation
boards than ranks, and a partial training batch. It does not load the 4B weights
or train the actual Sokoban dataset. Success writes `probe_result.json`.

When ready for the actual full SFT run:

```bash
bash scripts/train_jev_8xh800.sh configs/jev_visual_sft_8xh800.yaml
```

Defaults: 1 board per rank per optimizer update (8 boards globally), 4 coordinate
questions per forward per rank, learning rate `1e-5`, gradient checkpointing on,
and 1 epoch. These are starting settings, not measured throughput/VRAM optima.
Increasing `cell_batch_size` increases concurrent image/question branches;
increasing `board_batch_size` increases accumulation and the global board batch.

All ranks execute equal forward/backward counts. Tail boards and extra cell
rounds use zero-weight dummy computation; they do not enter loss or metrics.
The world-size factor in the loss compensates for FSDP gradient averaging, so
each real board has equal weight even with unequal board sizes and partial steps.
Distributed validation gathers disjoint real boards and checks exact coverage.

Each epoch checkpoint has `distributed/` model+optimizer shards, per-rank RNG /
scheduler files, and a consolidated `model/` export for ordinary evaluate/predict
commands. Resume requires the original world size, configuration, data hash and
software versions, and a new output directory:

```bash
# Edit output_dir to a new path; keep the original epoch schedule.
bash scripts/train_jev_8xh800.sh configs/jev_visual_sft_8xh800.yaml \
  --resume outputs/previous_8gpu_run/epoch_001
```

CPU initialization currently loads one FP32 copy per rank before sharding; for
4B parameters, eight copies are roughly 128 GB of host weight storage, plus
loading overhead. Full export also needs host memory on rank zero. Check server
RAM and disk capacity: checkpoints store both sharded training state and an
inference export. Interrupted `.partial` checkpoints are not overwritten; retain
or inspect them and use a new output path. The regular VERL verification lock is
independent of this FSDP implementation.
