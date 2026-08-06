#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Run GSM8K or AIME24/AIME25 with the SGLang benchmark evaluator against one or more
# local SGLang OpenAI-compatible servers. Select via BENCHMARK=gsm8k|aime24|aime25.

set -euo pipefail

SGLANG_REPO="${SGLANG_REPO:-/home/snorouzi/code/sglang-nemotron-dllm-a652eb48}"
SGLANG_COMMIT="${SGLANG_COMMIT:-a652eb48fa69bf0762c7de9f389474abbebe7b9d}"
VENV="${VENV:-/lustre/fsw/portfolios/coreai/users/snorouzi/sglang_nemotron_torch291_cu129_uvpy312_venv}"
MODEL="${MODEL:?MODEL must point to a Hugging Face checkpoint directory}"
TOKENIZER="${TOKENIZER:-${MODEL}}"
OUTDIR="${OUTDIR:-/lustre/fsw/portfolios/coreai/users/snorouzi/eval_results/sglang_gsm8k_benchmark}"

BENCHMARK="${BENCHMARK:-gsm8k}"
case "${BENCHMARK}" in
  gsm8k|aime24|aime25) ;;
  *)
    echo "Unsupported BENCHMARK=${BENCHMARK}; expected gsm8k, aime24, or aime25" >&2
    exit 2
    ;;
esac

NUM_SAMPLES="${NUM_SAMPLES:--1}"
if [[ "${BENCHMARK}" == "aime24" || "${BENCHMARK}" == "aime25" ]]; then
  MAX_TOKENS="${MAX_TOKENS:-4096}"
else
  MAX_TOKENS="${MAX_TOKENS:-1024}"
fi
TEMPERATURE="${TEMPERATURE:-0.0}"
CONCURRENT="${CONCURRENT:-16}"
HOST="${HOST:-127.0.0.1}"
PORT_BASE="${PORT_BASE:-32000}"
DP_SIZE="${DP_SIZE:-1}"

DTYPE="${DTYPE:-bfloat16}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.7}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"
CUDA_GRAPH_BS="${CUDA_GRAPH_BS-1 2 4 8 16 32 64 128}"
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-0}"
DISABLE_PIECEWISE_CUDA_GRAPH="${DISABLE_PIECEWISE_CUDA_GRAPH:-1}"

# On a652eb48, NemotronLabsDiffusion AR mode is enabled via
# --json-model-override-args '{"ar_mode": true}', not a DLLM algorithm.
DLLM_ALGORITHM="${DLLM_ALGORITHM:-NONE}"
if [[ -z "${JSON_MODEL_OVERRIDE_ARGS+x}" ]]; then
  JSON_MODEL_OVERRIDE_ARGS='{"ar_mode": true}'
fi
CAUSAL_CONTEXT="${CAUSAL_CONTEXT:-true}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
MAX_STEPS="${MAX_STEPS:-32}"
THRESHOLD="${THRESHOLD:-}"
LORA_PATH="${LORA_PATH:-}"
INCLUDE_STATS_FILE="${INCLUDE_STATS_FILE:-1}"

DLLM_TEMPERATURE="${DLLM_TEMPERATURE:-}"
SELECTION_POLICY="${SELECTION_POLICY:-}"

DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${OUTDIR}"
CLIENT_LOG="${OUTDIR}/client.log"
RESULTS_JSON="${OUTDIR}/results.json"

export HF_HOME="${HF_HOME:-/lustre/fsw/portfolios/coreai/users/snorouzi/hf_home}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK="${SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK:-1}"
export PYTHONPATH="${SGLANG_REPO}/python:${PYTHONPATH:-}"

if [[ ! -d "${SGLANG_REPO}/.git" ]]; then
  echo "SGLANG_REPO is not a git checkout: ${SGLANG_REPO}" >&2
  exit 1
fi

actual_sglang_commit="$(git -C "${SGLANG_REPO}" rev-parse HEAD)"
if [[ "${actual_sglang_commit}" != "${SGLANG_COMMIT}" ]]; then
  echo "SGLANG_REPO commit mismatch." >&2
  echo "  expected: ${SGLANG_COMMIT}" >&2
  echo "  actual:   ${actual_sglang_commit}" >&2
  echo "  repo:     ${SGLANG_REPO}" >&2
  exit 1
fi

if [[ -n "${DLLM_TEMPERATURE}" ]]; then
  export DLLM_TEMPERATURE
fi
if [[ -n "${SELECTION_POLICY}" ]]; then
  export SELECTION_POLICY
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS="," read -r -a visible_gpus <<<"${CUDA_VISIBLE_DEVICES}"
else
  visible_gpus=()
  for gpu_idx in $(seq 0 "$((DP_SIZE - 1))"); do
    visible_gpus+=("${gpu_idx}")
  done
fi

if (( ${#visible_gpus[@]} < DP_SIZE )); then
  echo "CUDA_VISIBLE_DEVICES has ${#visible_gpus[@]} GPUs but DP_SIZE=${DP_SIZE}" >&2
  exit 1
fi

write_dllm_config() {
  local rank="$1"
  local config_path="$2"

  {
    echo "algorithm: ${DLLM_ALGORITHM}"
    echo "causal_context: ${CAUSAL_CONTEXT}"
    if [[ "${DLLM_ALGORITHM}" != "AR" ]]; then
      echo "block_size: ${BLOCK_SIZE}"
      if [[ -n "${MAX_STEPS}" ]]; then
        echo "max_steps: ${MAX_STEPS}"
      fi
      if [[ -n "${THRESHOLD}" ]]; then
        echo "threshold: ${THRESHOLD}"
      fi
      if [[ -n "${SELECTION_POLICY}" ]]; then
        echo "selection_policy: ${SELECTION_POLICY}"
      fi
      if [[ -n "${LORA_PATH}" ]]; then
        echo "lora_path: ${LORA_PATH}"
      fi
    fi
    if [[ "${INCLUDE_STATS_FILE}" == "1" ]]; then
      echo "stats_file: ${OUTDIR}/stats_rank${rank}.jsonl"
    fi
  } >"${config_path}"
}

server_pids=()
urls=()

cleanup() {
  for pid in "${server_pids[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  for pid in "${server_pids[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT

{
  echo "SGLANG_REPO=${SGLANG_REPO}"
  echo "SGLANG_COMMIT=${SGLANG_COMMIT}"
  echo "VENV=${VENV}"
  echo "MODEL=${MODEL}"
  echo "TOKENIZER=${TOKENIZER}"
  echo "OUTDIR=${OUTDIR}"
  echo "BENCHMARK=${BENCHMARK}"
  echo "NUM_SAMPLES=${NUM_SAMPLES}"
  echo "MAX_TOKENS=${MAX_TOKENS}"
  echo "TEMPERATURE=${TEMPERATURE}"
  echo "CONCURRENT=${CONCURRENT}"
  echo "HOST=${HOST}"
  echo "PORT_BASE=${PORT_BASE}"
  echo "DP_SIZE=${DP_SIZE}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "VISIBLE_GPUS=${visible_gpus[*]}"
  echo "DLLM_ALGORITHM=${DLLM_ALGORITHM}"
  echo "JSON_MODEL_OVERRIDE_ARGS=${JSON_MODEL_OVERRIDE_ARGS}"
  echo "DLLM_TEMPERATURE=${DLLM_TEMPERATURE:-<unset>}"
  echo "SELECTION_POLICY=${SELECTION_POLICY:-<unset>}"
  echo "CAUSAL_CONTEXT=${CAUSAL_CONTEXT}"
  echo "BLOCK_SIZE=${BLOCK_SIZE}"
  echo "MAX_STEPS=${MAX_STEPS:-<unset>}"
  echo "THRESHOLD=${THRESHOLD:-<unset>}"
  echo "LORA_PATH=${LORA_PATH:-<none>}"
  echo "MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC}"
  echo "MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS}"
  echo "MAX_MODEL_LEN=${MAX_MODEL_LEN:-<default>}"
  echo "ATTENTION_BACKEND=${ATTENTION_BACKEND}"
  echo "CUDA_GRAPH_BS=${CUDA_GRAPH_BS}"
  echo "DISABLE_CUDA_GRAPH=${DISABLE_CUDA_GRAPH}"
  echo "DISABLE_PIECEWISE_CUDA_GRAPH=${DISABLE_PIECEWISE_CUDA_GRAPH}"
  echo "DRY_RUN=${DRY_RUN}"
} | tee "${CLIENT_LOG}"

for rank in $(seq 0 "$((DP_SIZE - 1))"); do
  port="$((PORT_BASE + rank))"
  server_log="${OUTDIR}/server_rank${rank}.log"
  urls+=("http://${HOST}:${port}/v1")

  server_args=(
    --model-path "${MODEL}"
    --tokenizer-path "${TOKENIZER}"
    --served-model-name default
    --trust-remote-code
    --host "${HOST}"
    --port "${port}"
    --tp-size 1
    --dtype "${DTYPE}"
    --mem-fraction-static "${MEM_FRACTION_STATIC}"
    --max-running-requests "${MAX_RUNNING_REQUESTS}"
    --attention-backend "${ATTENTION_BACKEND}"
  )

  if [[ -n "${JSON_MODEL_OVERRIDE_ARGS}" ]]; then
    server_args+=(--json-model-override-args "${JSON_MODEL_OVERRIDE_ARGS}")
  fi

  if [[ -n "${MAX_MODEL_LEN}" ]]; then
    server_args+=(--context-length "${MAX_MODEL_LEN}")
  fi
  if [[ -n "${CUDA_GRAPH_BS}" ]]; then
    read -r -a cuda_graph_bs_args <<<"${CUDA_GRAPH_BS}"
    server_args+=(--cuda-graph-bs "${cuda_graph_bs_args[@]}")
  fi
  if [[ "${DISABLE_CUDA_GRAPH}" == "1" ]]; then
    server_args+=(--disable-cuda-graph)
  fi
  if [[ "${DISABLE_PIECEWISE_CUDA_GRAPH}" == "1" ]]; then
    server_args+=(--disable-piecewise-cuda-graph)
  fi

  if [[ "${DLLM_ALGORITHM}" != "NONE" && "${DLLM_ALGORITHM}" != "none" ]]; then
    dllm_config="${OUTDIR}/dllm_config_rank${rank}.yaml"
    write_dllm_config "${rank}" "${dllm_config}"
    server_args+=(
      --dllm-algorithm "${DLLM_ALGORITHM}"
      --dllm-algorithm-config "${dllm_config}"
    )
    {
      echo "DLLM_CONFIG_RANK_${rank}:"
      cat "${dllm_config}"
    } | tee -a "${CLIENT_LOG}"
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf "SERVER_COMMAND_RANK_%s:" "${rank}" | tee -a "${CLIENT_LOG}"
    printf " %q" env "CUDA_VISIBLE_DEVICES=${visible_gpus[rank]}" \
      "${VENV}/bin/python" -m sglang.launch_server "${server_args[@]}" \
      | tee -a "${CLIENT_LOG}"
    printf "\n" | tee -a "${CLIENT_LOG}"
    continue
  fi

  env "CUDA_VISIBLE_DEVICES=${visible_gpus[rank]}" \
    "${VENV}/bin/python" -m sglang.launch_server "${server_args[@]}" \
    >"${server_log}" 2>&1 &
  server_pids+=("$!")
  echo "STARTED rank=${rank} pid=${server_pids[-1]} gpu=${visible_gpus[rank]} port=${port}" \
    | tee -a "${CLIENT_LOG}"
done

if [[ "${DRY_RUN}" == "1" ]]; then
  IFS=","
  echo "BASE_URLS=${urls[*]}" | tee -a "${CLIENT_LOG}"
  printf "EVAL_COMMAND:" | tee -a "${CLIENT_LOG}"
  printf " %q" "${VENV}/bin/python" "${SGLANG_REPO}/benchmark/gsm8k/eval_sglang.py" \
    --benchmark "${BENCHMARK}" \
    --base_url "${urls[*]}" \
    --model default \
    --no_thinking \
    --prompt_style v2 \
    --max_tokens "${MAX_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --concurrent "${CONCURRENT}" \
    --num_samples "${NUM_SAMPLES}" \
    --output "${RESULTS_JSON}" \
    | tee -a "${CLIENT_LOG}"
  printf "\n" | tee -a "${CLIENT_LOG}"
  exit 0
fi

for rank in $(seq 0 "$((DP_SIZE - 1))"); do
  port="$((PORT_BASE + rank))"
  server_log="${OUTDIR}/server_rank${rank}.log"
  for i in $(seq 1 600); do
    if ! kill -0 "${server_pids[rank]}" 2>/dev/null; then
      echo "SERVER_EXITED rank=${rank}" | tee -a "${CLIENT_LOG}"
      tail -n 200 "${server_log}" | tee -a "${CLIENT_LOG}"
      exit 1
    fi
    if curl -sf "http://${HOST}:${port}/health" >/dev/null 2>&1; then
      echo "SERVER_READY rank=${rank} after ${i}s" | tee -a "${CLIENT_LOG}"
      break
    fi
    if [[ "${i}" == "600" ]]; then
      echo "SERVER_TIMEOUT rank=${rank}" | tee -a "${CLIENT_LOG}"
      tail -n 200 "${server_log}" | tee -a "${CLIENT_LOG}"
      exit 1
    fi
    sleep 1
  done
done

IFS=","
base_urls="${urls[*]}"
unset IFS
echo "BASE_URLS=${base_urls}" | tee -a "${CLIENT_LOG}"

"${VENV}/bin/python" "${SGLANG_REPO}/benchmark/gsm8k/eval_sglang.py" \
  --benchmark "${BENCHMARK}" \
  --base_url "${base_urls}" \
  --model default \
  --no_thinking \
  --prompt_style v2 \
  --max_tokens "${MAX_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --concurrent "${CONCURRENT}" \
  --num_samples "${NUM_SAMPLES}" \
  --output "${RESULTS_JSON}" \
  2>&1 | tee -a "${CLIENT_LOG}"

echo "Client log: ${CLIENT_LOG}" | tee -a "${CLIENT_LOG}"
echo "Results: ${RESULTS_JSON}" | tee -a "${CLIENT_LOG}"
