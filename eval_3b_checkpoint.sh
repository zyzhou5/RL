#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"

NUM_GPUS=1
MODEL="${MODEL:-/lustre/fsw/portfolios/coreai/users/snorouzi/checkpoints/ministral_3_step_900_hf}"
OUTDIR="${OUTDIR:-/lustre/fsw/portfolios/coreai/users/snorouzi/eval_results/ministral_3_step_900_hf_gsm8k_pass1}"

# Select benchmark: gsm8k (default), aime24, or aime25.
BENCHMARK="${BENCHMARK:-gsm8k}"
case "${BENCHMARK}" in
  gsm8k|aime24|aime25) ;;
  *)
    echo "Unsupported BENCHMARK=${BENCHMARK}; expected gsm8k, aime24, or aime25" >&2
    exit 2
    ;;
esac

NUM_SAMPLES="${NUM_SAMPLES:--1}"
# On a652eb48, AR mode is a model config override, not a DLLM algorithm.
DLLM_ALGORITHM="${DLLM_ALGORITHM:-NONE}"
if [[ -z "${JSON_MODEL_OVERRIDE_ARGS+x}" ]]; then
  JSON_MODEL_OVERRIDE_ARGS='{"ar_mode": true}'
fi
DLLM_TEMPERATURE="${DLLM_TEMPERATURE:-0.0}"
SELECTION_POLICY="${SELECTION_POLICY:-confidence}"
THRESHOLD="${THRESHOLD:-0.9}"

SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-coreai_dlalgo_llm}"
SBATCH_PARTITION="${SBATCH_PARTITION:-batch_short}"
SBATCH_TIME="${SBATCH_TIME:-02:00:00}"
SBATCH_JOB_NAME="${SBATCH_JOB_NAME:-eval_3b_instruct_${BENCHMARK}}"
CONTAINER_IMAGE=/lustre/fsw/portfolios/nvr/users/lwhalen/docker/lxaw_ministral_sft_eval/nemo-rl-lxaw-sft-eval.sqsh
CONTAINER_MOUNTS=/home/snorouzi:/home/snorouzi,/lustre:/lustre

# In-container venv + CUDA paths. flashinfer JIT subprocess-invokes ninja/nvcc,
# so these must be on PATH and CUDA_HOME must be set inside the container.
VENV="${VENV:-/lustre/fsw/portfolios/coreai/users/snorouzi/sglang_nemotron_torch291_cu129_uvpy312_venv}"
SGLANG_REPO="${SGLANG_REPO:-/home/snorouzi/code/sglang-nemotron-dllm-a652eb48}"
SGLANG_COMMIT="${SGLANG_COMMIT:-a652eb48fa69bf0762c7de9f389474abbebe7b9d}"
CUDA_HOME_IN_CONTAINER="${CUDA_HOME_IN_CONTAINER:-/usr/local/cuda}"

if [[ "${1:-}" == "--sbatch" ]]; then
  mkdir -p "${OUTDIR}"
  job_id="$(
    sbatch --parsable \
      -A "${SBATCH_ACCOUNT}" \
      -p "${SBATCH_PARTITION}" \
      --time="${SBATCH_TIME}" \
      --gpus="${NUM_GPUS}" \
      --job-name="${SBATCH_JOB_NAME}" \
      --output="${OUTDIR}/slurm-%j.out" \
      --export=ALL \
      --wrap="srun --container-image=${CONTAINER_IMAGE} --container-mounts=${CONTAINER_MOUNTS} --container-workdir=${SCRIPT_DIR} bash ${SCRIPT_PATH}"
  )"
  echo "job_id=${job_id}"
  echo "outdir=${OUTDIR}"
  echo "slurm_out=${OUTDIR}/slurm-${job_id}.out"
  squeue -j "${job_id}" -o "%.18i %.9P %.32j %.8T %.10M %.10l %.6D %R"
  exit 0
fi

if [[ $# -gt 0 ]]; then
  echo "Unknown argument: $1" >&2
  echo "Usage: $0 [--sbatch]" >&2
  exit 2
fi

export PATH="${VENV}/bin:${CUDA_HOME_IN_CONTAINER}/bin:${PATH}"
export CUDA_HOME="${CUDA_HOME_IN_CONTAINER}"
export CUDA_PATH="${CUDA_HOME_IN_CONTAINER}"
export LD_LIBRARY_PATH="${CUDA_HOME_IN_CONTAINER}/lib64:${LD_LIBRARY_PATH:-}"

MODEL="${MODEL}" \
OUTDIR="${OUTDIR}" \
BENCHMARK="${BENCHMARK}" \
SGLANG_REPO="${SGLANG_REPO}" \
SGLANG_COMMIT="${SGLANG_COMMIT}" \
JSON_MODEL_OVERRIDE_ARGS="${JSON_MODEL_OVERRIDE_ARGS}" \
NUM_SAMPLES="${NUM_SAMPLES}" \
DP_SIZE="${NUM_GPUS}" \
CUDA_VISIBLE_DEVICES="$(seq -s, 0 "$((NUM_GPUS - 1))")" \
CONCURRENT="${NUM_GPUS}" \
DLLM_ALGORITHM="${DLLM_ALGORITHM}" \
DLLM_TEMPERATURE="${DLLM_TEMPERATURE}" \
SELECTION_POLICY="${SELECTION_POLICY}" \
THRESHOLD="${THRESHOLD}" \
tools/nemotron_diffusion/run_sglang_gsm8k_benchmark_eval.sh
