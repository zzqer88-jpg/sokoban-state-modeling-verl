#!/usr/bin/env bash
# Run in the server environment after editing the selected YAML configuration.
set -euo pipefail
CONFIG=${1:-configs/jev_visual_sft_8xh800.yaml}
shift "$(( $# > 0 ? 1 : 0 ))"
exec torchrun --standalone --nnodes=1 --nproc-per-node=8 \
  -m sokoban_state_modeling.jev.cli train --config "$CONFIG" "$@"
