# Configuration status

The directory contains complete thinking/non-thinking project profiles for
both PPO and GRPO. The launcher selects one from `TRAIN_ALGORITHM=ppo|grpo` and
`REASONING_MODE=thinking|non_thinking`, merges it after Hydra composition,
resolves all required environment variables, and writes the optional resolved
config.

All profiles use VERL V1 synchronous training, FSDP, the custom manifest
Dataset, the token-boundary-aware reward manager, and the same state
answer/reward contract. PPO uses GAE and an FSDP critic. GRPO disables the
critic, generates four responses per prompt, and applies actor-side KL loss.

These are release candidates, not evidence that a particular CUDA machine is
compatible. `constraints/verl.lock.json` must remain `verified=false` until the
documented CPU tokenizer/Hydra contract probe and a one-step GPU FSDP/rollout
probe both pass on the server.
