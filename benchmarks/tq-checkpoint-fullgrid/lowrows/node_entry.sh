#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail
source '/lustre/fsw/portfolios/coreai/users/zezhou/ws/dev/dataplane/tq-checkpoint-fullgrid-lowrows-4n8o-428520fcc-20260914T214500Z/campaign.env'
cd "${RL_SOURCE}"
export CAMPAIGN_ROOT RL_SOURCE RL_SHA RL_TREE TQ_SOURCE TQ_SHA BENCHMARK_SHA CONTAINER
export MC_BENCH_OVERLAY MC_BENCH_WHEEL MC_BENCH_WHEEL_SHA256
export MC_BENCH_SEGMENT_BYTES MC_BENCH_BUFFER_BYTES OPTIMIZED_MODULE_SHA256
export PYTHONPATH="${CAMPAIGN_ROOT}:${MC_BENCH_OVERLAY}:${TQ_SOURCE}:${RL_SOURCE}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export MC_MOONCAKE_PROTOCOL=rdma MC_MOONCAKE_DEVICE=mlx5_0,mlx5_1,mlx5_3,mlx5_4
export MC_GID_INDEX=0 MC_RDMA_PORT=1 MC_STORE_MEMCPY=0
export TQ_NUM_THREADS=8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export BENCH_PROFILE=0
export CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void
export RAY_TMPDIR="/tmp/nrl-ray/${SLURM_JOB_ID}" TMPDIR="/tmp/nrl-ray/${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${CAMPAIGN_ROOT}/.nrl/cache/${SLURMD_NODENAME}"
export UV_CACHE_DIR="${CAMPAIGN_ROOT}/.nrl/uv/${SLURMD_NODENAME}"
export RAY_USAGE_STATS_CONFIG_PATH="${CAMPAIGN_ROOT}/.nrl/ray-usage-stats-${SLURMD_NODENAME}.json"
export RAY_USAGE_STATS_ENABLED=0 RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export RAY_OVERRIDE_RESOURCES='{"CPU":128,"GPU":0}'
unset RAY_ADDRESS RAY_USE_MULTIPROCESSING_CPU_COUNT RAY_OBJECT_SPILLING_DIRECTORY RAY_OBJECT_SPILLING_CONFIG RAY_object_spilling_directory RAY_object_spilling_config UV_CACHE_DIR_OVERRIDE
mkdir -p "${RAY_TMPDIR}" "${XDG_CACHE_HOME}" "${UV_CACHE_DIR}"
exec /root/.local/bin/uv run --no-project --no-sync --python /opt/nemo_rl_venv/bin/python \
  "${CAMPAIGN_ROOT}/node_runner.py"
