---
name: nemotron-experiment-submit
description: Submit and monitor NeMo-RL NemotronLabsDiffusion GRPO-family experiments on dfw, including AR GRPO, JustGRPO, and DiffuGRPO configs, run naming, env tags, rebuild flags, logs, and Slurm status checks.
---

# RL on NemotronLabsDiffusion Experiment Submission

Use this skill when submitting or debugging NeMo-RL training experiments from the current RL repo on dfw.

## Working Directory

Run from the active RL checkout:

```bash
cd ~/diffusion_RL/RL
```

Submit through the repo's Nemotron diffusion sbatch wrapper:

```bash
tools/nemotron_diffusion/submit_grpo_nemotron_ar_megatron_sbatch.sh
```

The wrapper name contains `ar`, but it is also used for JustGRPO and DiffuGRPO configs unless the repo introduces a newer algorithm-specific wrapper. The selected algorithm comes from `CONFIG=...` and any `EXTRA_CONFIG_OVERRIDES`, not from the script name.

Always pass the intended config file explicitly with `CONFIG=...`; do not rely on the script default when launching a named experiment.

## Standard Submit Command

```bash
RUN_NAME=<run_name> \
WANDB_RUN_NAME=<run_name> \
JOB_NAME=<short_slurm_name> \
PARTITION=batch \
TIME=04:00:00 \
NODES=1 \
ENV_TAG=mb_3rdparty_sglagn_local_fork \
CONFIG=examples/configs/<config_file>.yaml \
NRL_FORCE_REBUILD_VENVS=false \
FORCE_REINSTALL_PACKAGES=false \
FORCE_REINSTALL_SGLANG=false \
bash tools/nemotron_diffusion/submit_grpo_nemotron_ar_megatron_sbatch.sh --sbatch
```

After submission, explicitly share the exact submission command with the user together with the Slurm job id.

Use PARTITION=batch for all sbatch submissions. The dfw batch partition has a 4-hour limit, so keep TIME at or below 04:00:00. Adjust NODES as needed, but do not submit these experiments to backfill, batch_long, batch_large, or batch_large_long.

## vLLM Diffusion Backend (async-GRPO + NeMo-Gym)

The default backend above is SGLang. There is a second, separate path that runs
the dLLM over a **custom vLLM fork** with non-colocated async-GRPO and NeMo-Gym
data routing -- this is what the v30 deterministic-agent runs use.

Configs live under `examples/configs/*_vllm_*` (and
`examples/configs/grpo_sudoku6x6_*_vllm*` for the sudoku smokes). The production
v30 config is:

- main: `CONFIG=examples/configs/nemotron_labs_diffusion_8b_vllm_block_just_grpo_async_grpo_nemogym_v30.yaml`
- toy:  `CONFIG=examples/configs/nemotron_labs_diffusion_8b_vllm_block_just_grpo_async_grpo_nemogym_v30_toy.yaml`

The v30 config bakes in its own overrides (8B checkpoint, node split, grpo
sizes, `enforce_eager`, `sequence_parallel`, etc.); read its header comment for
the current values rather than re-passing them on the command line.

### What differs from an SGLang submit

- **Entry script**: `RUN_SCRIPT=examples/nemo_gym/run_grpo_nemo_gym.py` (not the
  default `examples/run_grpo.py`).
- **uv extras**: `UV_EXTRAS="mcore nemo_gym"` (adds the `nemo_gym`
  optional-dependency group on top of `mcore`).
- **NeMo-Gym server venv**: `NEMO_GYM_VENV_DIR=/lustre/fsw/portfolios/coreai/users/snorouzi/gym_venvs`
  (pre-provisioned; the gym agent/server runs from it).
- **vLLM runtime**: `NRL_VLLM_PY_EXECUTABLE=<vllm-runtime-venv>/bin/python-compat`
  -- see below.
- **ENV_TAG**: a vLLM/nemo_gym driver-env tag, distinct from the SGLang
  `mb_3rdparty_sglagn_local_fork`, e.g. `mb_3rdparty_vllm_local_fork`. The
  driver env must carry the `nemo_gym` extra; because `--no-sync` will not
  add it, provision the tag once with `--build-env` and the intended
  `UV_EXTRAS` (see "Provisioning a New ENV_TAG" below) before normal reuse.

### `NRL_VLLM_PY_EXECUTABLE`

The vLLM generation workers run a **custom, pre-built vLLM fork** (the
diffusion-capable engine), not the vLLM uv would install into the driver env.
`NRL_VLLM_PY_EXECUTABLE` is the absolute path to that runtime's python, e.g.:

```
NRL_VLLM_PY_EXECUTABLE=/lustre/fsw/portfolios/coreai/users/snorouzi/vllm_runtimes/nemotron_dllm_792ab07/bin/python-compat
```

When it is set, two things change (see `ray_actor_environment_registry.py` and
`vllm_worker.py`):

1. `VLLM_EXECUTABLE` becomes this path, so Ray launches the vLLM generation
   actors directly on this python -- no per-actor uv venv is built for them. The
   `AsyncTrajectoryCollector` / `ReplayBuffer` actors deliberately stay on the
   stock `PY_EXECUTABLES.VLLM` uv env (they need the full nemo_rl import closure
   and never run an engine).
2. The vLLM source monkey-patches (`_patch_vllm_init_workers_ray`, eagle3,
   hermes) are **skipped**, so the launcher does not rewrite the fork's source
   tree. Apply any equivalent change in the fork deliberately if TP/PP>1 needs
   it.

Configs reference it via `py_executable: ${oc.env:NRL_VLLM_PY_EXECUTABLE,unknown}`,
so if you forget to export it the run fails fast with an `unknown` executable
rather than silently using the wrong python.

### Diffusion temperature must match

vLLM denoises the whole canvas per step under one engine-wide temperature, so a
request's per-request temperature is only a gate (0 or 1). The worker asserts
`policy.generation.temperature == policy.generation.vllm_kwargs.diffusion_config.temperature`
and raises if they differ: rollout logprobs are recorded at the
`diffusion_config` value while the Megatron recompute tempers by
`generation.temperature`, so a mismatch silently desyncs train-vs-behavior and
blows up gen-KL. Keep the two equal.

### Example v30 toy smoke submit

```bash
RUN_NAME=grpo_nd8b_v30_smoke \
WANDB_RUN_NAME=grpo_nd8b_v30_smoke \
JOB_NAME=v30_smoke \
ACCOUNT=coreai_dlalgo_genai \
PARTITION=batch \
TIME=00:40:00 \
NODES=3 \
ENV_TAG=mb_3rdparty_vllm_local_fork \
RUN_SCRIPT=examples/nemo_gym/run_grpo_nemo_gym.py \
UV_EXTRAS="mcore nemo_gym" \
NEMO_GYM_VENV_DIR=/lustre/fsw/portfolios/coreai/users/snorouzi/gym_venvs \
NRL_VLLM_PY_EXECUTABLE=/lustre/fsw/portfolios/coreai/users/snorouzi/vllm_runtimes/nemotron_dllm_792ab07/bin/python-compat \
CONFIG=examples/configs/nemotron_labs_diffusion_8b_vllm_block_just_grpo_async_grpo_nemogym_v30_toy.yaml \
NRL_FORCE_REBUILD_VENVS=false \
FORCE_REINSTALL_PACKAGES=false \
FORCE_REINSTALL_SGLANG=false \
bash tools/nemotron_diffusion/submit_grpo_nemotron_ar_megatron_sbatch.sh --sbatch
```

The toy config is 3 nodes (2 train + 1 gen). For the full production run, swap
to the non-toy `..._v30.yaml` config and set `NODES=15` -- it must match the
config's baked-in `cluster.num_nodes: 15` node split (12 train + 3 gen).

## Account Selection (Fair Share)

Pick `ACCOUNT` based on Slurm fair-share: submit under the eligible account with the **highest current FairShare**, since that account gets the best scheduling priority (shortest queue wait).

Check fair-share before submitting:

```bash
sshare -U -u $USER -o Account,User,RawShares,NormShares,RawUsage,EffectvUsage,FairShare -p
```

The `FairShare` column is what matters (higher = better priority, range 0-1). Choose the account with the largest value, with two caveats:

- Restrict the choice to accounts compatible with this work: the run dirs and HF cache live under the `coreai` portfolio (`/lustre/fsw/portfolios/coreai/...`), so use a `coreai_dlalgo_*` account. Do not use a non-coreai account such as `nvr_lpr_llm` even if it shows a higher FairShare, since it does not have access to these paths/partition.
- An account's FairShare drops as its recent usage rises, so the best choice changes over time. Re-check `sshare` for each new submission rather than hardcoding one account.

In practice `coreai_dlalgo_genai` has had a higher FairShare than the wrapper default `coreai_dlalgo_llm` (whose FairShare is depressed by heavy recent usage), so prefer `ACCOUNT=coreai_dlalgo_genai` unless `sshare` says otherwise at submit time.

## Common Configs

AR GRPO:

- main: `CONFIG=examples/configs/gsm8k_nemotron_labs_diffusion_3b_sglang_ar_megatron.yaml`
- toy: `CONFIG=examples/configs/gsm8k_nemotron_labs_diffusion_3b_sglang_ar_megatron_toy_p8_g8.yaml`

JustGRPO leftmost reveal:

- main: `CONFIG=examples/configs/gsm8k_nemotron_labs_diffusion_3b_sglang_justgrpo_leftmost_megatron.yaml`
- toy: `CONFIG=examples/configs/gsm8k_nemotron_labs_diffusion_3b_sglang_justgrpo_leftmost_megatron_toy_p8_g8.yaml`

DiffuGRPO with FastDiffuser:

- main: `CONFIG=examples/configs/gsm8k_nemotron_labs_diffusion_3b_sglang_diffugrpo_megatron.yaml`
- toy: `CONFIG=examples/configs/gsm8k_nemotron_labs_diffusion_3b_sglang_diffugrpo_megatron_toy_p8_g8.yaml`

For a smoke test, use the toy config for the relevant mode and cap steps through `EXTRA_CONFIG_OVERRIDES`, for example `grpo.max_num_steps=3 grpo.val_at_end=false checkpointing.enabled=false logger.wandb_enabled=false`.


## DeepScaleR dataset

DeepScaleR variants train on `agentica-org/DeepScaleR-Preview-Dataset` (`dataset_name: DeepScaler`, ~40K math problems) and validate on AIME 2024 (`HuggingFaceH4/aime_2024`, `dataset_name: AIME2024`, `repeat: 16`). Both route through the `math` env / `hf_math_verify` reward. The datasets are prefetched into the offline HF cache at `/lustre/fsw/portfolios/coreai/users/$USER/hf_home/datasets` (the training container is HF-offline; prefetch new datasets from a login node into that path before submitting).

Configs (DeepScaleR train + AIME2024 validation; 4096 sequence budget = 350 prompt + 3744 generation; KL off; `val_period`/`save_period` = 10):

- AR GRPO:   `CONFIG=examples/configs/deepscaler_nemotron_labs_diffusion_3b_sglang_ar_megatron.yaml`
- DiffuGRPO: `CONFIG=examples/configs/deepscaler_nemotron_labs_diffusion_3b_sglang_diffugrpo_megatron.yaml`

The diffugrpo config inherits the data block + 4096 sequence budget from the AR DeepScaleR config; KL-off (`loss_fn.reference_policy_kl_penalty: 0.0` + `grpo.skip_reference_policy_logprobs_calculation: true`) and the val/save intervals live in the AR parent.

### max_new_tokens must be divisible by FastDiffuser block_size (32)

For DiffuGRPO, `policy.generation.max_new_tokens` MUST be a multiple of the FastDiffuser `block_size` (32). A non-multiple (e.g. 3746) crashes generation with a RoPE view error (`view size is not compatible with input tensor's size and stride ...` in `rotary_embedding`). Use 3744 (= 117*32). AR mode has no such constraint but is kept at 3744 for consistency.

### Multi-node: set cluster.num_nodes to match NODES

`NODES` only sizes the Slurm allocation; the wrapper does NOT inject `cluster.num_nodes`, and the config defaults to `cluster.num_nodes: 1`. For multi-node runs you MUST override it via `EXTRA_CONFIG_OVERRIDES`, or NeMo-RL runs on a single node:

```bash
NODES=8 ... EXTRA_CONFIG_OVERRIDES='cluster.num_nodes=8' ...
```

### Example 8-node AR DeepScaleR submit

```bash
RUN_NAME=deepscaler_ar_3b_8n WANDB_RUN_NAME=deepscaler_ar_3b_8n JOB_NAME=ds_ar_8n \
ACCOUNT=coreai_dlalgo_genai PARTITION=batch TIME=04:00:00 NODES=8 \
ENV_TAG=mb_3rdparty_sglagn_local_fork \
CONFIG=examples/configs/deepscaler_nemotron_labs_diffusion_3b_sglang_ar_megatron.yaml \
NRL_FORCE_REBUILD_VENVS=false FORCE_REINSTALL_PACKAGES=false FORCE_REINSTALL_SGLANG=false \
EXTRA_CONFIG_OVERRIDES='cluster.num_nodes=8 policy.logprob_batch_size=4' \
bash tools/nemotron_diffusion/submit_grpo_nemotron_ar_megatron_sbatch.sh --sbatch
```

## Rebuild and Reinstall Flags

Use these deliberately:

- `NRL_FORCE_REBUILD_VENVS=false`: reuse worker venvs. Use this for normal reruns after envs exist.
- `NRL_FORCE_REBUILD_VENVS=true`: rebuild worker venvs. Use after dependency changes or when stale worker packages are suspected.
- `FORCE_REINSTALL_NEMO_RL=false`: keep the editable NeMo-RL install. Use this for normal Python-only changes under `nemo_rl/...`; new driver/worker processes import the active worktree directly.
- `FORCE_REINSTALL_SGLANG=true`: force reinstall SGLang into the driver env. Use after changing the SGLang dependency source in `pyproject.toml` or when validating a new SGLang checkout.
- `FORCE_REINSTALL_PACKAGES=true`: broad reinstall switch. Avoid unless needed; it can trigger long package builds.

NeMo-RL is installed editable in the usual driver and Ray worker environments, so source-only edits to the active RL worktree do not require `NRL_FORCE_REBUILD_VENVS=true` or `FORCE_REINSTALL_NEMO_RL=true`. Rebuilding the venv can unnecessarily trigger slow dependency builds such as TransformerEngine.

If `sglang` is an editable path dependency in `pyproject.toml`, normal Python edits in that SGLang checkout are picked up by new worker processes after restart. Dependency, metadata, kernel, or compiled-extension changes still require reinstall/rebuild.

## Environment Sync: the launcher runs `uv run --no-sync`

The wrapper runs the driver process, the Ray head, and every Ray worker with
`uv run --no-sync` (the driver `uv run`, plus the exported `RAY_START_CMD` /
`RAY_STATUS_CMD`). `--no-sync` tells uv to **use the project env on disk exactly
as-is and never reconcile it to `uv.lock`**. This is baked into the launcher;
you do not pass it yourself.

Why it is required (not cosmetic): ray bringup runs `RAY_START_CMD` on the head
and on every worker **concurrently**, all pointed at the same shared driver env
under `UV_PROJECT_ENVIRONMENT` on lustre. Without `--no-sync` each of those is a
`uv run --locked`, i.e. a destructive reconcile (uninstall + reinstall) of that
one shared directory at the same moment. Concurrent reconciles corrupt the env
("failed to create directory" / "Failed to read metadata"), which then surfaces
as a Ray start race (`AttributeError: 'NoneType' object has no attribute
'decode'`) and `check_srun_processes` tears the whole job down. `--no-sync`
drops the reconcile entirely, so during bringup the shared env is only read,
never mutated. (The main SGLang runs did not hit this only because their env
happened to already be in sync; it bit the vLLM/nemo_gym env right after a
`pyproject.toml` change forced a reconcile.)

Operational consequence: because normal runs no longer sync the env, **a
dependency change is not picked up automatically**. After editing
`pyproject.toml` / `uv.lock` (new dependency, changed git pin, etc.) you must
provision the env deliberately once -- run with `NRL_FORCE_REBUILD_VENVS=true`
(and/or the relevant `FORCE_REINSTALL_*` flag), or bump `ENV_TAG` to build a
fresh env -- before going back to normal `--no-sync` reuse. Source-only edits to
editable trees (`nemo_rl/...`, an editable `sglang` or vLLM checkout) are
unaffected and need no sync.

## Provisioning a New ENV_TAG (`--build-env`)

A training submit runs `uv run --no-sync` (see above), so it will **never
create the driver env**. A brand-new `ENV_TAG` must be provisioned once, up
front, with the launcher's `--build-env` mode -- that path runs `uv run` *with*
sync and is the only thing that materializes
`nemorl_uv_driver_envs/diffusion_RL_RL_${ENV_TAG}`. Skip it and bringup fails
with confusing ray import / `Failed to read metadata` errors rather than a
clear "env missing" message.

Run it inside the container on a compute node -- it needs nvcc for the mcore /
TransformerEngine builds, which login nodes do not have:

```bash
srun -A coreai_dlalgo_llm --partition interactive --time 1:30:00 --nodes=1 --gpus-per-node=8 \
  --container-image /lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_llm/users/sfawzy/nemo-rl-nightly.sqsh \
  --container-mounts=/home/snorouzi:/home/snorouzi,/lustre:/lustre \
  bash -lc 'cd /home/snorouzi/diffusion_RL/RL && \
    ENV_TAG=<new_tag> \
    UV_EXTRAS="mcore nemo_gym" \
    bash tools/nemotron_diffusion/submit_grpo_nemotron_ar_megatron_sbatch.sh --build-env'
```

Budget 30-45 min for a cold tag (TransformerEngine compiles from source); later
runs reusing the tag are fast. Pass the same `UV_EXTRAS` you intend to run with
-- `--no-sync` cannot add an extra later, so a tag built without `nemo_gym`
stays without it.

### Do NOT override `UV_CACHE_DIR` / `UV_CACHE_DIR_OVERRIDE`

Leave both unset, at build time and at submit time. `ENV_TAG` already derives
them consistently, and the run depends on that consistency:

```text
wrapper:94    UV_CACHE_DIR="${UV_CACHE_DIR:-.../uv_cache_${ENV_TAG}}"                    # driver side
wrapper:261   UV_CACHE_DIR_OVERRIDE="${UV_CACHE_DIR_OVERRIDE:-.../uv_cache_${ENV_TAG}}"
ray.sub:105   MOUNTS+=",$UV_CACHE_DIR_OVERRIDE:/root/.cache/uv"                          # in-container uv
```

### Pre-build the per-actor worker venvs (or the idle-GPU watchdog kills the run)

`--build-env` provisions only the DRIVER env. Each Ray actor class additionally
gets its own venv under `nemo_rl_worker_venvs_${ENV_TAG}/<actor-FQN>`, built
LAZILY by `_env_builder` the first time the driver instantiates that class
(`nemo_rl/utils/venvs.py`). On a fresh tag they are built in two separate
stages:

- the Megatron policy worker -- during `setup()`
  (`grpo.py:806 initialize_generation_with_policy`)
- `ReplayBuffer` + `AsyncTrajectoryCollector` -- much later, inside
  `async_grpo_train` (`grpo.py:2971` / `grpo.py:3006`), i.e. AFTER the model
  is initialized

The vLLM generation workers build nothing -- they run on
`NRL_VLLM_PY_EXECUTABLE` directly.

Both stages leave every GPU in the allocation idle. On a fresh tag that easily
exceeds 30 minutes, and `svc-hwinf-cs-sched` (uid 146504) reaps idle-GPU jobs at
~31 min:

```text
sacct -j <id>  ->  CANCELLED by 146504 | 0:0 | 00:30:42
```

The job is not broken; it never got to use a GPU in time. So pre-build the
venvs on ONE node first, then submit the real multi-node run. `_env_builder`
early-returns when `<venv>/bin/python` already exists, so the run just picks
them up:

```bash
srun -A coreai_dlalgo_llm --partition interactive --time 0:50:00 --nodes=1 --gpus-per-node=8 \
  --container-image /lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_llm/users/sfawzy/nemo-rl-nightly.sqsh \
  --container-mounts=/home/snorouzi:/home/snorouzi,/lustre:/lustre \
  bash -lc '
L=/lustre/fsw/portfolios/coreai/users/snorouzi
export HOME=$L/container_home
export NEMO_RL_VENV_DIR=$L/nemo_rl_worker_venvs_<ENV_TAG>
export UV_CACHE_DIR=$L/uv_cache_<ENV_TAG>
cd /home/snorouzi/diffusion_RL/RL
$L/nemorl_uv_driver_envs/diffusion_RL_RL_<ENV_TAG>/bin/python -c "
from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
from nemo_rl.utils.venvs import create_local_venv
FQNS = [
    # the policy worker for THIS run -- copy from policy.worker_cls_fqn in the config
    \"nemo_rl.models.policy.workers.block_just_grpo_megatron_policy_worker.BlockJustGRPOMegatronPolicyWorker\",
    # async-GRPO only; omit for a synchronous run
    \"nemo_rl.algorithms.async_utils.ReplayBuffer\",
    \"nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector\",
]
for fqn in FQNS:
    print(fqn, \"->\", create_local_venv(get_actor_python_env(fqn), fqn))
"'
```

Keep `NEMO_RL_VENV_DIR` / `UV_CACHE_DIR` keyed to the same `ENV_TAG` the run
will use, for the reason in the previous section.

Build EVERY actor class the run will instantiate, not just the async pair --
the policy worker is the largest venv (~666 packages) and the slowest stage.
Use `get_actor_python_env(fqn)` rather than hardcoding an executable: it returns
the same per-class env the runtime would pick (`--extra mcore` for the Megatron
policy workers, `--extra vllm` for ReplayBuffer / AsyncTrajectoryCollector), so
the pre-built venv is byte-for-byte the one the job looks for. Swap the policy
FQN for whichever worker the config names, e.g.
`...trace_grpo_megatron_policy_worker.TraceGRPOMegatronPolicyWorker` for Trace.

Verify before submitting -- a complete set looks like:

```bash
V=/lustre/fsw/portfolios/coreai/users/snorouzi/nemo_rl_worker_venvs_${ENV_TAG}
for w in $V/*/; do echo "$(basename $w): $(ls $w/lib/python3.13/site-packages | wc -l)"; done
#   ...BlockJustGRPOMegatronPolicyWorker: 666
#   ...async_utils.ReplayBuffer:          620
#   ...async_utils.AsyncTrajectoryCollector: 620
```


## Env Tag

Use this shared env tag for the current dependency layout:

```bash
ENV_TAG=mb_3rdparty_sglagn_local_fork
```

Use this same env tag for DiffuGRPO megatron runs as well:

```bash
ENV_TAG=mb_3rdparty_sglagn_local_fork
```

Do not change `ENV_TAG` for normal Python edits in the existing editable dependency trees. Reuse the tag so follow-up runs reuse the same driver and worker envs.

Change `ENV_TAG` only when one of the dependency paths changes, for example a new `pyproject.toml` path for SGLang or a different Megatron-Bridge/Megatron-LM workspace path. After changing dependency paths, run once with the new tag and the needed reinstall/rebuild flags. The first run with a fresh tag may spend significant time building packages such as TransformerEngine; later runs with the same tag should be faster.

## Logs and Checkpoints

The default run directory is:

```text
/lustre/fsw/portfolios/coreai/users/$USER/runs/diffusion_rl/<RUN_NAME>
```

Key files:

```text
run log:     /lustre/fsw/portfolios/coreai/users/$USER/runs/diffusion_rl/<RUN_NAME>/run.log
slurm log:   /lustre/fsw/portfolios/coreai/users/$USER/runs/diffusion_rl/<RUN_NAME>/slurm-<jobid>.out
checkpoints: /lustre/fsw/portfolios/coreai/users/$USER/runs/diffusion_rl/<RUN_NAME>/checkpoints
```

## Monitoring

Check Slurm status:

```bash
squeue -j <jobid> -o "%.18i %.12P %.35j %.8u %.2t %.10M %.6D %R"
sacct -j <jobid> --format=JobID,JobName%30,State,ExitCode,Elapsed,Start,End -P
```

Tail logs:

```bash
RUN_DIR=/lustre/fsw/portfolios/coreai/users/$USER/runs/diffusion_rl/<RUN_NAME>
tail -200 "$RUN_DIR/run.log"
tail -200 "$RUN_DIR/slurm-<jobid>.out"
```

Useful grep checks:

```bash
grep -n "SGLANG_SOURCE\|runtime_versions\|FastDiffuser: block_size\|selection_policy\|Started a local Ray\|Step [0-9]/\|Loss:\|RuntimeError\|Traceback" "$RUN_DIR"/run.log "$RUN_DIR"/slurm-<jobid>.out
```

For JustGRPO leftmost runs, confirm SGLang logs include `selection_policy=leftmost`. If the log only prints `FastDiffuser: block_size=... max_steps=... temperature=... threshold=...`, the worker may be importing an older SGLang build.

For DiffuGRPO confidence runs, confirm SGLang logs include `FastDiffuser: block_size=32`, `threshold=0.9`, and `selection_policy=confidence`, and confirm the policy worker is `DiffuGRPOMegatronPolicyWorker`.

## Local or Interactive Run

Only run locally inside an allocated interactive GPU node/container:

```bash
RUN_NAME=<run_name> \
WANDB_RUN_NAME=<run_name> \
ENV_TAG=mb_3rdparty_sglagn_local_fork \
CONFIG=examples/configs/<config_file>.yaml \
bash tools/nemotron_diffusion/submit_grpo_nemotron_ar_megatron_sbatch.sh --local
```
