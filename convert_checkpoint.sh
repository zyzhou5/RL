#!/usr/bin/env bash
set -euo pipefail

MEGATRON_PATCH_DIR_PREFIX=""
if [[ "${1:-}" == "--ministral3" ]]; then
  MEGATRON_PATCH_DIR_PREFIX="/lustre/fsw/portfolios/coreai/users/snorouzi/nemorl_runtime_patches:"
  shift
fi
if [[ $# -gt 0 ]]; then
  echo "Unknown argument: $1" >&2
  exit 2
fi

MEGATRON_PATCH_DIR="${MEGATRON_PATCH_DIR_PREFIX}${MEGATRON_PATCH_DIR:-/home/snorouzi/code/Megatron-Bridge/src:/lustre/fsw/portfolios/coreai/users/snorouzi/hf_home/modules}" \
STEP_DIR="${STEP_DIR:?STEP_DIR is required}" \
OUT="${OUT:?OUT is required}" \
BASE_MODEL=/lustre/fsw/portfolios/coreai/users/snorouzi/hf_home/hub/models--nvidia--Nemotron-Diffusion-Exp-Ministral-3B-Instruct/snapshots/d7c52fbc82c29932c18a02478da6e93921daad34 \
tools/nemotron_diffusion/convert_nemotron_diffusion_checkpoint_to_hf.sh
