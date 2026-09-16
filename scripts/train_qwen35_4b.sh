#!/usr/bin/env bash
set -euo pipefail

: "${VERL_CHECKOUT:?set VERL_CHECKOUT to the Stage-0-verified checkout}"
: "${SOKOBAN_PROJECT_ROOT:?set SOKOBAN_PROJECT_ROOT to this repository}"
: "${REASONING_MODE:?set REASONING_MODE to thinking or non_thinking}"
TRAIN_ALGORITHM="${TRAIN_ALGORITHM:-ppo}"
: "${MODEL_PATH:?set MODEL_PATH to Qwen3.5-4B or its local checkpoint}"
: "${SOKOBAN_STATE_TRAIN_MANIFEST:?set the 50,000-row train manifest}"
: "${SOKOBAN_STATE_TRAIN_INDEX:?set the train manifest index}"
: "${SOKOBAN_STATE_VALIDATION_MANIFEST:?set the 2,000-row validation manifest}"
: "${SOKOBAN_STATE_VALIDATION_INDEX:?set the validation manifest index}"

case "$REASONING_MODE" in
  thinking|non_thinking) ;;
  *) echo "REASONING_MODE must be thinking or non_thinking" >&2; exit 2 ;;
esac
case "$TRAIN_ALGORITHM" in
  ppo|grpo) ;;
  *) echo "TRAIN_ALGORITHM must be ppo or grpo" >&2; exit 2 ;;
esac

export SOKOBAN_VERL_LOCK="$SOKOBAN_PROJECT_ROOT/constraints/verl.lock.json"
export PYTHONPATH="$SOKOBAN_PROJECT_ROOT/src:$VERL_CHECKOUT${PYTHONPATH:+:$PYTHONPATH}"
PROJECT_CONFIG="$SOKOBAN_PROJECT_ROOT/configs/${TRAIN_ALGORITHM}_qwen35_4b_${REASONING_MODE}.yaml"

cd "$SOKOBAN_PROJECT_ROOT"

python -m sokoban_state_modeling.scripts.preflight_training \
  --check-no-vagen-runtime \
  --verify-lock \
  --verl-checkout "$VERL_CHECKOUT" \
  --train-manifest "$SOKOBAN_STATE_TRAIN_MANIFEST" \
  --train-index "$SOKOBAN_STATE_TRAIN_INDEX" \
  --validation-manifest "$SOKOBAN_STATE_VALIDATION_MANIFEST" \
  --validation-index "$SOKOBAN_STATE_VALIDATION_INDEX"

exec python -m sokoban_state_modeling.training.main_ppo \
  --project-config "$PROJECT_CONFIG" \
  "$@"
