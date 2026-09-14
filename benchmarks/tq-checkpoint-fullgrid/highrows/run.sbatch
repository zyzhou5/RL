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

#SBATCH --job-name=tq-grid-highpar-428-221200
#SBATCH --account=nemotron_sw_post
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --mem=920G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/zezhou/ws/dev/dataplane/tq-checkpoint-fullgrid-highrows-parallel-4n8o-428520fcc-20260914T221200Z/slurm/%j.out
#SBATCH --error=/lustre/fsw/portfolios/coreai/users/zezhou/ws/dev/dataplane/tq-checkpoint-fullgrid-highrows-parallel-4n8o-428520fcc-20260914T221200Z/slurm/%j.err
set -euo pipefail
source '/lustre/fsw/portfolios/coreai/users/zezhou/ws/dev/dataplane/tq-checkpoint-fullgrid-highrows-parallel-4n8o-428520fcc-20260914T221200Z/campaign.env'
[[ "${SLURM_CLUSTER_NAME}" == oci-hsg-cs-001 && "${SLURM_JOB_ID}" =~ ^[0-9]+$ ]]
[[ "${SLURM_JOB_NUM_NODES}" == 4 && "${SLURM_CPUS_PER_TASK}" == 128 ]]
cd "${CAMPAIGN_ROOT}"
[[ "$(git -C "${RL_SOURCE}" rev-parse HEAD)" == "${RL_SHA}" ]]
[[ "$(git -C "${RL_SOURCE}" rev-parse HEAD^{tree})" == "${RL_TREE}" ]]
[[ "$(git -C "${RL_SOURCE}" status --porcelain=v1 --untracked-files=normal)" == "$(cat "${CAMPAIGN_ROOT}/rl-status.expected")" ]]
[[ "$(git -C "${TQ_SOURCE}" rev-parse HEAD)" == "${TQ_SHA}" ]]
[[ "$(git -C "${TQ_SOURCE}" status --porcelain=v1 --untracked-files=normal)" == "$(cat "${CAMPAIGN_ROOT}/tq-status.expected")" ]]
printf '%s  %s\n' "${UV_LOCK_SHA}" "${RL_SOURCE}/uv.lock" | sha256sum --check -
printf '%s  %s\n' "${OPTIMIZED_MODULE_SHA256}" "${RL_SOURCE}/nemo_rl/data_plane/adapters/tq_mooncake_checkpoint.py" | sha256sum --check -
sha256sum --check launcher.sha256
sha256sum --check source.sha256
mkdir -p evidence .nrl/control
scontrol show hostnames "${SLURM_JOB_NODELIST}" >evidence/allocated-hosts.txt
[[ "$(wc -l <evidence/allocated-hosts.txt)" == 4 ]]
export BENCH_EXPECTED_HOSTS
BENCH_EXPECTED_HOSTS="$(paste -sd, evidence/allocated-hosts.txt)"
export CAMPAIGN_ROOT NVIDIA_VISIBLE_DEVICES=void CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
scontrol show job "${SLURM_JOB_ID}" >"evidence/slurm-job-${SLURM_JOB_ID}.txt"
stat -c '%U|%s|%y|%n' "${CONTAINER}" >evidence/container-stat.txt
git -C "${RL_SOURCE}" submodule status >evidence/submodules.txt
git -C "${RL_SOURCE}" status --porcelain=v1 --untracked-files=normal >evidence/source-status.txt
df -h "${CAMPAIGN_ROOT}" >evidence/filesystem.txt
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --cpus-per-task=128 --cpu-bind=cores \
  --kill-on-bad-exit=1 --container-image="${CONTAINER}" \
  --container-mounts=/lustre:/lustre,/dev/infiniband:/dev/infiniband \
  --container-workdir="${RL_SOURCE}" --no-container-mount-home --no-container-entrypoint \
  --container-env=BENCH_EXPECTED_HOSTS,NVIDIA_VISIBLE_DEVICES,CUDA_VISIBLE_DEVICES,PYTHONDONTWRITEBYTECODE,PYTHONNOUSERSITE \
  --output="${CAMPAIGN_ROOT}/slurm/${SLURM_JOB_ID}-node-%t.out" \
  --error="${CAMPAIGN_ROOT}/slurm/${SLURM_JOB_ID}-node-%t.err" \
  bash "${CAMPAIGN_ROOT}/node_entry.sh"
printf 'FOUR_NODE_COMPARISON_PASS job_id=%s cases=20 repetitions=1\n' "${SLURM_JOB_ID}" >completed.ok
