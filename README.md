# Sokoban State Modeling for VERL

Standalone Sokoban state generation, deterministic RGB rendering, rule-based
rewards, single-turn datasets, and multi-turn environment sessions. Runtime
code does not import VAGEN.

## Status

The domain/data baseline, multimodal Dataset contract, token-aware reward
manager, two reasoning profiles, and commit-specific VERL V1 sync Trainer
adapter are implemented independently. The official checkout at commit
`61134f92099913b6f8fa5920ac5839a4a314212d` has been source-audited and hashed.
Training remains fail-closed until the remaining CPU/tokenizer and GPU probes
set `constraints/verl.lock.json` to `verified=true`.

The sync adapter keeps upstream PPO/GAE/checkpoint logic, replaces only the
DataLoaders to enforce full coverage and `drop_last=false`, and adds centralized
sum/count aggregation for the fixed Sokoban reward schema. Padding rollouts are
audited but excluded from domain-quality metrics.

## Development

```bash
python -m pip install -e ".[test]"
python -m pytest
```

Generate a small smoke manifest:

```bash
sokoban-generate --output-dir ./data/smoke --split train --num-tasks 20
```

The generator uses cheap category/wall predicates before shortest-path search
and a deterministic 64-bit bitboard BFS for the supported 6x6--8x8 boards. The
original NumPy BFS remains as a differential-test oracle. A 30-row train build
on the development machine improved from 11.041 s to 2.508 s while preserving
the manifest and index byte-for-byte.

The complete migration and release gates are documented in
`STANDALONE_VERL_MIGRATION_PLAN.md`; the immutable single-turn dataset contract
is documented in `DATASET_PLAN.md`.

## Server staging

Do not copy `.deps/verl` implicitly through Git: `.deps/` is intentionally
ignored. Clone the audited checkout separately and keep it clean:

```bash
export SOKOBAN_PROJECT_ROOT=/absolute/path/sokoban-state-modeling-verl
export VERL_CHECKOUT=/absolute/path/verl
git clone https://github.com/verl-project/verl.git "$VERL_CHECKOUT"
git -C "$VERL_CHECKOUT" checkout 61134f92099913b6f8fa5920ac5839a4a314212d
```

Install the CUDA/torch/rollout stack from that checkout's lock, then install
this package into the same environment. The audited VERL checkout currently
defines `vllm` and `fsdp` extras; its exact CUDA requirements must match the
server rather than being guessed from the development machine.

Generate and publish the three formal splits (the command is resumable at
committed chunks):

```bash
export PYTHONPATH="$SOKOBAN_PROJECT_ROOT/src:$VERL_CHECKOUT${PYTHONPATH:+:$PYTHONPATH}"
python -m sokoban_state_modeling.scripts.generate_dataset --output-dir ./data/release --split train --num-tasks 50000 --resume
python -m sokoban_state_modeling.scripts.generate_dataset --output-dir ./data/release --split validation --num-tasks 2000 --resume
python -m sokoban_state_modeling.scripts.generate_dataset --output-dir ./data/release --split test --num-tasks 10000 --resume
python -m sokoban_state_modeling.scripts.audit_splits \
  --train ./data/release/manifests/train.jsonl \
  --validation ./data/release/manifests/validation.jsonl \
  --test ./data/release/manifests/test.jsonl \
  --output ./data/release/audits/split_leakage.json
```

Export the training inputs and select one explicit reasoning mode:

```bash
export MODEL_PATH=/absolute/path/Qwen3.5-4B
export SOKOBAN_STATE_TRAIN_MANIFEST=$PWD/data/release/manifests/train.jsonl
export SOKOBAN_STATE_TRAIN_INDEX=$PWD/data/release/manifests/train.idx
export SOKOBAN_STATE_VALIDATION_MANIFEST=$PWD/data/release/manifests/validation.jsonl
export SOKOBAN_STATE_VALIDATION_INDEX=$PWD/data/release/manifests/validation.idx
export TRAIN_ALGORITHM=ppo  # or grpo
export REASONING_MODE=thinking  # or non_thinking
```

Before enabling training, verify the source lock and run the CPU tokenizer,
Hydra, custom Dataset, multimodal collate, and reward-boundary probe for both
profiles:

```bash
python -m sokoban_state_modeling.scripts.preflight_training \
  --check-no-vagen-runtime --verify-source-lock --verl-checkout "$VERL_CHECKOUT"
python -m sokoban_state_modeling.scripts.probe_runtime \
  --verl-checkout "$VERL_CHECKOUT" \
  --project-config "$SOKOBAN_PROJECT_ROOT/configs/ppo_qwen35_4b_${REASONING_MODE}.yaml" \
  --model-path "$MODEL_PATH" \
  --report "$SOKOBAN_PROJECT_ROOT/outputs/runtime_probe_${REASONING_MODE}.json"
```

`scripts/train_qwen35_4b.sh` validates the 50,000/2,000 train/validation rows,
selects the matching PPO/GRPO and reasoning profile, and prevents imports from
a caller's working directory. GRPO uses four responses per prompt, no critic,
and otherwise shares the exact Dataset/reward/coverage contract. The launcher
intentionally refuses to train while the runtime lock is unverified. Run a
one-step FSDP/rollout probe for every algorithm/reasoning profile that will be
used, and record the resulting versions in `constraints/runtime.lock.txt`
before changing the official lock to `verified=true`.
