#!/usr/bin/env bash
# One-stage prompt-open thinking SFT for Qwen3.5-4B on the formal Sokoban data.
set -euo pipefail

: "${SOKOBAN_PROJECT_ROOT:?set SOKOBAN_PROJECT_ROOT to this repository}"
: "${VERL_CHECKOUT:?set VERL_CHECKOUT to the pinned VERL checkout}"
: "${MODEL_PATH:?set MODEL_PATH to Qwen3.5-4B}"

SFT_DATA_ROOT=${SFT_DATA_ROOT:-"${SOKOBAN_PROJECT_ROOT}/data/release/sft"}
SFT_TRAIN_FILE=${SFT_TRAIN_FILE:-"${SFT_DATA_ROOT}/train.thinking.jsonl"}
SFT_VALIDATION_FILE=${SFT_VALIDATION_FILE:-"${SFT_DATA_ROOT}/validation.thinking.jsonl"}
SFT_OUTPUT_DIR=${SFT_OUTPUT_DIR:-"${SOKOBAN_PROJECT_ROOT}/outputs/sft/qwen35_4b_thinking_full"}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-48}
MAX_LENGTH=${MAX_LENGTH:-2048}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-32768}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
SAVE_FREQ=${SAVE_FREQ:-after_each_epoch}
TEST_FREQ=${TEST_FREQ:-after_each_epoch}
PROJECT_NAME=${PROJECT_NAME:-sokoban-state-modeling-sft}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen35_4b_thinking_full_sft}
LOGGER=${LOGGER:-console}
RESUME_MODE=${RESUME_MODE:-auto}
SP_SIZE=${SP_SIZE:-1}
PARAM_OFFLOAD=${PARAM_OFFLOAD:-false}
OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-false}

if [[ ! -d "${VERL_CHECKOUT}/verl" ]]; then
  echo "VERL checkout is missing: ${VERL_CHECKOUT}" >&2
  exit 2
fi
if (( NNODES != 1 )); then
  echo "this launcher is single-node; use NNODES=1" >&2
  exit 2
fi
if [[ ! -f "${SFT_TRAIN_FILE}" || ! -f "${SFT_VALIDATION_FILE}" ]]; then
  echo "formal SFT exports are missing under ${SFT_DATA_ROOT}" >&2
  exit 2
fi
if [[ ! -e "${MODEL_PATH}" && "${MODEL_PATH}" != */* ]]; then
  echo "MODEL_PATH must be a local path or a Hugging Face repository id" >&2
  exit 2
fi
if (( TRAIN_BATCH_SIZE % (NNODES * NGPUS_PER_NODE) != 0 )); then
  echo "TRAIN_BATCH_SIZE must be divisible by NNODES * NGPUS_PER_NODE" >&2
  exit 2
fi

export PYTHONPATH="${SOKOBAN_PROJECT_ROOT}/src:${VERL_CHECKOUT}${PYTHONPATH:+:${PYTHONPATH}}"
python -m sokoban_state_modeling.scripts.preflight_sft \
  --train "${SFT_TRAIN_FILE}" \
  --validation "${SFT_VALIDATION_FILE}" \
  --report "${SFT_OUTPUT_DIR}/data_preflight.json"

mkdir -p "${SFT_OUTPUT_DIR}"
cd "${VERL_CHECKOUT}"

torchrun \
  --nnodes="${NNODES}" \
  --nproc-per-node="${NGPUS_PER_NODE}" \
  --standalone \
  -m verl.trainer.sft_trainer \
  data.train_files="${SFT_TRAIN_FILE}" \
  data.val_files="${SFT_VALIDATION_FILE}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_length="${MAX_LENGTH}" \
  data.truncation=error \
  data.pad_mode=no_padding \
  data.use_dynamic_bsz=true \
  data.max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
  data.messages_key=messages \
  +data.image_key=images \
  data.enable_thinking_key=enable_thinking \
  data.enable_thinking_default=true \
  +data.apply_chat_template_kwargs.enable_thinking=true \
  data.ignore_input_ids_mismatch=true \
  data.num_workers=4 \
  data.custom_cls.path="${SOKOBAN_PROJECT_ROOT}/src/sokoban_state_modeling/sft/verl_dataset.py" \
  data.custom_cls.name=SokobanThinkingSFTDataset \
  model.path="${MODEL_PATH}" \
  model.trust_remote_code=true \
  model.enable_gradient_checkpointing=true \
  model.use_remove_padding=true \
  model.lora_rank=0 \
  model.lora_adapter_path=null \
  engine=fsdp \
  engine.strategy=fsdp2 \
  engine.ulysses_sequence_parallel_size="${SP_SIZE}" \
  engine.param_offload="${PARAM_OFFLOAD}" \
  engine.optimizer_offload="${OPTIMIZER_OFFLOAD}" \
  optim.lr="${LEARNING_RATE}" \
  optim.lr_warmup_steps_ratio=0.03 \
  optim.weight_decay=0.01 \
  optim.lr_scheduler_type=cosine \
  trainer.nnodes="${NNODES}" \
  trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.logger="['${LOGGER}']" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.default_local_dir="${SFT_OUTPUT_DIR}" \
  trainer.resume_mode="${RESUME_MODE}" \
  trainer.max_ckpt_to_keep=2 \
  "$@"
