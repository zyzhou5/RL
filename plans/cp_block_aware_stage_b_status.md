# Block-aware CP (stage b) -- implementation status, 2026-07-27

Companion to `cp_kv_sharding_blockaware_ring.md` (which lives untracked in the
main repo working tree, not in git).

## Where the code is

| | branch | worktree |
|---|---|---|
| Megatron-Bridge | `cp-block-aware` (on `efb1adbb`) | `/lustre/fsw/portfolios/coreai/users/snorouzi/bridge-cp-block-aware` |
| NeMo-RL | `cp-block-aware-rl` (on `559259b36`) | `/lustre/fsw/portfolios/coreai/users/snorouzi/RL-cp-block-aware` |

The RL worktree's `3rdparty/*` are SYMLINKS: Megatron-Bridge points at the
bridge worktree above, the others at the main repo. They show as `T`
(typechange) in `git status` and must never be staged.

## The switch

`DIFFU_CP_BLOCK_AWARE`, read on BOTH sides with identical tolerant parsing --
the bridge attention layer and `nemo_rl/models/megatron/cp_block_aware.py`. Off
by default. Use a word (`enable`), not `1`/`true`: OmegaConf coerces those to
int/bool and Ray then rejects `runtime_env["env_vars"]`, which wants
`Dict[str, str]`.

It is deliberately one env var rather than a config key on each side. The data
path and attention MUST agree: under one global zigzag rank r owns a different
SET of positions than under two segmented zigzags. That is a cross-rank
difference, not a local permutation, so a disagreement is not recoverable
inside attention -- it silently reads logprobs from the wrong tokens.

## Done

**Bridge** -- `asymmetric_semi_ar_mask_mod` extracted as the one
global-coordinate predicate; `compute_asymmetric_semi_ar_block_aware_mask`
(Q_LEN=(N+C)/cp, KV_LEN=N/cp+C); segmented zigzag primitives in `cp_utils`;
`_asymmetric_semi_ar_forward` gains the clean-only gather, segmented RoPE ids
for Q *and* K, the block-aware mask, and skips the output scatter.

The mask_mod's index remap is a GATHER through a precomputed table, not
arithmetic: flex lowers mask_mod into the attention Triton kernel and
re-evaluates it on partial blocks, so it must be an indexing pattern that
lowers there (`prompt_lengths[b]` proves gathers do).

**NeMo-RL** -- segmented shard/re-gather primitives in `model_utils`;
`gather_cp_sharded_logits` (the SINGLE-zigzag inverse, still used by
`just_grpo_train.py` for plain leftmost-reveal JustGRPO) now raises if the
block-aware flag is set, because the flag is global and that path would
otherwise reconstruct the sequence in the wrong order silently;
`data.process_microbatch` shards per segment; `diffu_grpo_train` slices target
ids and re-gathers logprobs per segment; `diffu_grpo_logprobs` pads PER SEGMENT
(noisy to `2*cp*block_size`, clean to `2*cp`) rather than once globally.

## Validated

63/63 bridge tests and the RL parity suite, in the training environment
(nemo-rl-nightly container, torch 2.10.0+cu129, production
`fused_flex_attention`):

- fp64 parity vs cp=1: forward, dQ, and the asymmetric dK/dV split -- clean
  summed across ranks, **noisy exact per rank with no reduction**
- the same parity on the compiled CUDA path (tolerance 1e-4; measured drift
  ~1e-6)
- noisy-K/V locality on the dense mask, with a negative control showing the
  block-alignment requirement is load-bearing
- compiled vs eager BlockMask builds bit-identical
- a LAYER-level test running the real forward over a gloo group at cp in {2,4},
  whose per-rank outputs reassemble into the single-rank result -- aimed
  squarely at the RoPE change
- NeMo-RL's segmented sharding is asserted identical to Megatron-Bridge's

## End-to-end validation (2026-07-28)

Block JustGRPO toy, 1 node / 8 GPUs, 3 steps, ENV_TAG
`cpba_nd3b_sglang_a652eb48_mb3d58667e`. Submit scripts:
`~/submit_bjg_cp{1,2,4}_*.sh`.

| run | Generation KL Error | attention KV_LEN | job |
|---|---|---|---|
| cp=1 baseline | 0.0003 / 0.0007 / 0.0007 | 1216 (full) | 14492524 |
| cp=2 block-aware | 0.0003 / 0.0007 / 0.0003 | 960 | 14496247 |
| cp=4 block-aware | 0.0002 / 0.0003 / 0.0007 | 896 | 14497526 |

Gen KL matching the baseline is the substantive result: training logprobs agree
with what SGLang actually generated, through a different data layout. Shapes
confirm the layout -- at cp=4, Q=320 and KV=896 solve to N=512, C=768, with N
divisible by 128 (2*cp*block_size) and C by 8.

### Two real bugs this found, that the unit tests could not

Both are "non-contiguous tensor into an NCCL collective", which only the
per-segment layout produces (`narrow()` in forward; strided grads from the `cat`
backward):

1. `AllGatherCPTensor.forward` -> all_gather
2. `AllGatherCPTensor.backward` -> all_reduce (also breaks the `.view()` after)

**GLOO ACCEPTS non-contiguous tensors; only NCCL rejects them.** A CPU
round-trip test therefore passes against code that fails in production -- proven
by negative control, twice. Tests must assert contiguity at the collective
boundary rather than observe behaviour. `narrow()` on a 1-row tensor also still
reports contiguous (PyTorch ignores size-1 strides), so batch>=2 is required.
Bug 2 additionally needs real NCCL: that backward builds its index on a
hardcoded CUDA device and cannot run under gloo at all.

## NOT done

- **No end-to-end run.** Nothing has executed data path + attention + re-gather
  together on real data. This is the main remaining risk.
- **Gen-KL smoke** (section 6, train-vs-inference agreement ~1e-3) not run.
  That is what would catch a subtly wrong RoPE that unit tests pass through.
- **Section 4's 128K memory profile** -- the decision gate -- still never run,
  so the size of the prize is unmeasured. Under full activation checkpointing
  (b) is a throughput/comm win, not the multi-GB memory win in the payoff
  tables.
- Stage (c) untouched.

## Ready-to-submit smoke run (NOT yet submitted)

    cd /lustre/fsw/portfolios/coreai/users/snorouzi/RL-cp-block-aware

    RUN_NAME=bjg_cp2_blockaware_toy \
    WANDB_RUN_NAME=bjg_cp2_blockaware_toy \
    JOB_NAME=bjg_cp2_ba \
    ACCOUNT=coreai_dlalgo_genai \
    PARTITION=batch \
    TIME=01:00:00 \
    NODES=1 \
    ENV_TAG=cpba_nd3b_sglang_a652eb48_mb2cae14b4 \
    CONFIG=examples/configs/deepscaler_nemotron_labs_diffusion_3b_sglang_block_just_grpo_megatron_toy_8p8g_cp2_blockaware.yaml \
    NRL_FORCE_REBUILD_VENVS=false \
    FORCE_REINSTALL_PACKAGES=false \
    FORCE_REINSTALL_SGLANG=false \
    EXTRA_CONFIG_OVERRIDES='+policy.megatron_cfg.env_vars.DIFFU_CP_BLOCK_AWARE=enable grpo.max_num_steps=3 grpo.val_at_end=false checkpointing.enabled=false logger.wandb_enabled=false' \
    bash tools/nemotron_diffusion/submit_grpo_nemotron_ar_megatron_sbatch.sh --sbatch

Caveats before running it:

1. **ENV_TAG is new**, because the repo path changed. The first run with a new
   tag rebuilds the driver/worker envs and can spend a long time on packages
   such as TransformerEngine. That is expected, not a failure.
2. The wrapper has never been run from a worktree at this path; `REPO_DIR` and
   `MEGATRON_PATCH_DIR` resolution through the `3rdparty` symlinks is
   unverified.
3. A cp=2 / block_size=16 run needs the noisy segment divisible by 64 and the
   clean segment by 4. The batch builder now pads to satisfy both; if it
   somehow does not, the assert names the knob rather than failing obscurely.
4. What to check first in `run.log`: the two startup lines
   `DIFFU_CP_BLOCK_AWARE=1: block-aware CP active` (bridge) and that step-0
   logprobs are finite and comparable to a cp=1 baseline.
