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

"""Policy-training parallelism benchmark.

Replays a dumped (or synthesized -- see
tools/nemotron_diffusion/make_synthetic_train_batch.py) GRPO training batch
through policy.train() under a sweep of CP variants x (tp, cp) layouts, with no
generation / SGLang involvement at all. Each cell gets a fresh Policy (fresh
megatron worker group); OOMs are caught and recorded as results rather than
aborting the sweep.

The CP variant has to be swept by re-creating the Policy, not by flipping a
flag: both DIFFU_CP_* variables are read at MODULE IMPORT time (in
nemo_rl/models/megatron/cp_block_aware.py and in the Megatron-Bridge attention
layer), so they only take effect in a worker process that has not imported
those modules yet. Fresh Ray actors per cell give exactly that.

Env inputs:
  NRL_TRAIN_BENCH_BATCH        path to the batch .pt                 (required)
  NRL_TRAIN_BENCH_LAYOUTS      comma list of TPxCP, e.g. "8x4,4x4"   (default: config's own)
  NRL_TRAIN_BENCH_CP_VARIANTS  comma list from gather_all|local_q|block_aware
                                                                     (default: local_q)
  NRL_TRAIN_BENCH_ITERS        timed iterations per cell             (default 4)
"""

import argparse
import copy
import os
import time
import traceback

import torch
from omegaconf import OmegaConf

from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.algorithms.loss import ClippedPGLossFn
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)

# The three CP treatments of the asymmetric [noisy | clean] attention, as the
# attention layer branches on them:
#   gather_all  -- all-gather Q, K and V; every rank computes the full sequence.
#   local_q     -- Q stays sharded, K/V all-gathered in full  (variant "(a)").
#   block_aware -- Q sharded, only the CLEAN K/V gathered; the noisy K/V never
#                  leaves its owning rank  (variant "(b)").
# block_aware implies local-Q inside the layer, so its DIFFU_CP_LOCAL_Q value is
# irrelevant; it is set explicitly anyway so a cell's env fully describes it.
CP_VARIANTS = {
    "gather_all": {"DIFFU_CP_LOCAL_Q": "0", "DIFFU_CP_BLOCK_AWARE": "0"},
    "local_q": {"DIFFU_CP_LOCAL_Q": "1", "DIFFU_CP_BLOCK_AWARE": "0"},
    "block_aware": {"DIFFU_CP_LOCAL_Q": "1", "DIFFU_CP_BLOCK_AWARE": "1"},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Policy-training parallelism bench")
    parser.add_argument("--config", type=str, required=True)
    args, overrides = parser.parse_known_args()
    return args, overrides


def _worker_memory(policy, method: str):
    """Call a no-arg memory method on every megatron worker; max over ranks.

    Returns a list of per-worker dicts (or Nones for workers that did not
    report), so callers can take the max -- CP ranks are not symmetric under
    block-aware sharding, and the largest rank is what sets the OOM point.
    """
    import ray

    try:
        futures = policy.worker_group.run_all_workers_single_data(method)
        return [r if isinstance(r, dict) else None for r in ray.get(futures)]
    except Exception as exc:  # noqa: BLE001
        print(f"[train-bench] WARN: {method} failed: {exc!r}", flush=True)
        return [None]


def main() -> None:
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    config = load_config(args.config)
    if overrides:
        config = parse_hydra_overrides(config, overrides)
    config: MasterConfig = OmegaConf.to_container(config, resolve=True)

    batch_path = os.environ["NRL_TRAIN_BENCH_BATCH"]
    # Default: benchmark the config's own layout (set tp/cp via the normal
    # config/overrides). NRL_TRAIN_BENCH_LAYOUTS ("TPxCP,TPxCP,...") sweeps
    # several layouts in one job instead.
    layouts_env = os.environ.get("NRL_TRAIN_BENCH_LAYOUTS")
    if layouts_env:
        layouts = [tuple(int(x) for x in s.split("x")) for s in layouts_env.split(",")]
    else:
        layouts = [
            (
                config["policy"]["megatron_cfg"]["tensor_model_parallel_size"],
                config["policy"]["megatron_cfg"]["context_parallel_size"],
            )
        ]
    variants = [
        v.strip()
        for v in os.environ.get("NRL_TRAIN_BENCH_CP_VARIANTS", "local_q").split(",")
        if v.strip()
    ]
    unknown = [v for v in variants if v not in CP_VARIANTS]
    if unknown:
        raise ValueError(
            f"Unknown CP variant(s) {unknown}; expected from {sorted(CP_VARIANTS)}"
        )
    n_iters = int(os.environ.get("NRL_TRAIN_BENCH_ITERS", "4"))

    print(f"[train-bench] batch: {batch_path}")
    print(f"[train-bench] layouts (tp x cp): {layouts}, variants: {variants}, "
          f"timed iters: {n_iters}")

    init_ray()
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])

    cluster_cfg = config["cluster"]
    cluster = RayVirtualCluster(
        name="train_bench_cluster",
        bundle_ct_per_node_list=[cluster_cfg["gpus_per_node"]]
        * cluster_cfg["num_nodes"],
        use_gpus=True,
        num_gpus_per_node=cluster_cfg["gpus_per_node"],
        max_colocated_worker_groups=1,
    )
    world_size = cluster_cfg["gpus_per_node"] * cluster_cfg["num_nodes"]

    loss_fn = ClippedPGLossFn(config["loss_fn"])

    raw = torch.load(batch_path, map_location="cpu", weights_only=False)
    train_data = BatchedDataDict(raw)
    n_samples = train_data["sample_mask"].shape[0]
    seq_len = train_data["input_ids"].shape[1]
    print(
        f"[train-bench] batch loaded: {n_samples} samples x {seq_len} tokens, "
        f"keys={sorted(raw.keys())}"
    )

    results = []
    for tp, cp in layouts:
        for variant in variants:
            dp = world_size // (tp * cp)
            tag = f"{variant}_tp{tp}_cp{cp}_dp{dp}"
            if world_size % (tp * cp) != 0 or n_samples % dp != 0:
                print(f"[train-bench] SKIP {tag}: indivisible layout")
                results.append((tag, variant, (tp, cp), "skip", None))
                continue

            pol_cfg = copy.deepcopy(config["policy"])
            pol_cfg["megatron_cfg"]["tensor_model_parallel_size"] = tp
            pol_cfg["megatron_cfg"]["context_parallel_size"] = cp
            # Policy() asserts train_iters is set (grpo.setup computes it from
            # the dataloader to size the LR schedule); the bench just needs a
            # horizon.
            pol_cfg["megatron_cfg"]["train_iters"] = max(100, n_iters + 1)
            env_vars = dict(pol_cfg["megatron_cfg"].get("env_vars") or {})
            # Must reach the megatron workers, where BOTH the data path and the
            # attention layer read them -- and as plain strings, since Ray's
            # runtime_env rejects a Dict[str, str] with non-str values.
            env_vars.update(CP_VARIANTS[variant])
            pol_cfg["megatron_cfg"]["env_vars"] = env_vars

            print(f"[train-bench] ===== {tag}: init policy =====", flush=True)
            policy = None
            try:
                t0 = time.perf_counter()
                policy = Policy(
                    cluster=cluster,
                    config=pol_cfg,
                    tokenizer=tokenizer,
                    weights_path=None,
                    optimizer_path=None,
                    init_optimizer=True,
                )
                init_s = time.perf_counter() - t0

                print(f"[train-bench] {tag}: warmup iter", flush=True)
                policy.train(train_data, loss_fn)

                # Static floor: measured AFTER warmup so allocator caches and
                # the optimizer state are already resident. Peak is reset here
                # so it is attributable to the timed iterations alone.
                static = _worker_memory(policy, "get_memory_stats")
                _worker_memory(policy, "reset_peak_memory_stats")

                times = []
                for i in range(n_iters):
                    t0 = time.perf_counter()
                    policy.train(train_data, loss_fn)
                    times.append(time.perf_counter() - t0)
                    print(
                        f"[train-bench] {tag}: iter {i + 1}/{n_iters} {times[-1]:.1f}s",
                        flush=True,
                    )

                peak = _worker_memory(policy, "get_memory_stats")
                static_gb = max((s or {}).get("allocated_gb", 0.0) for s in static)
                peak_gb = max((s or {}).get("max_allocated_gb", 0.0) for s in peak)
                reserved_gb = max((s or {}).get("max_reserved_gb", 0.0) for s in peak)
                # Activation = peak - static is the discriminator in section 4
                # of plans/cp_kv_sharding_blockaware_ring.md: if it barely moves
                # as cp grows the run is bound by the full-length gathered K/V.
                act_gb = peak_gb - static_gb

                # Median, not mean: a single straggler iteration (a stray
                # autotune or an allocator flush) would otherwise swamp a gain
                # that is a few percent of step time.
                srt = sorted(times)
                med_t = srt[len(srt) // 2]
                mean_t = sum(times) / len(times)
                results.append(
                    (
                        tag,
                        variant,
                        (tp, cp),
                        "ok",
                        {
                            "init_s": init_s,
                            "iters": times,
                            "mean_s": mean_t,
                            "median_s": med_t,
                            "static_gb": static_gb,
                            "peak_gb": peak_gb,
                            "act_gb": act_gb,
                            "reserved_gb": reserved_gb,
                        },
                    )
                )
                print(
                    f"[train-bench] RESULT {tag}: median {med_t:.1f}s (mean {mean_t:.1f}s) "
                    f"over {n_iters} iters (init {init_s:.0f}s) | static {static_gb:.2f} GiB, "
                    f"peak {peak_gb:.2f} GiB, activation {act_gb:.2f} GiB, "
                    f"max_reserved {reserved_gb:.2f} GiB",
                    flush=True,
                )
            except Exception as e:
                traceback.print_exc()
                results.append((tag, variant, (tp, cp), f"FAILED: {type(e).__name__}", None))
                print(f"[train-bench] RESULT {tag}: FAILED {type(e).__name__}: {e}", flush=True)
            finally:
                if policy is not None:
                    try:
                        policy.shutdown()
                    except Exception:
                        traceback.print_exc()

    print("[train-bench] ================ SUMMARY ================")
    for tag, _variant, _layout, status, detail in results:
        if detail:
            print(
                f"[train-bench]   {tag}: median {detail['median_s']:.1f}s  "
                f"static {detail['static_gb']:.2f} GiB  peak {detail['peak_gb']:.2f} GiB  "
                f"activation {detail['act_gb']:.2f} GiB"
            )
        else:
            print(f"[train-bench]   {tag}: {status}")

    # Per-layout variant deltas: the actual deliverable. Baseline is the first
    # requested variant, so "local_q,block_aware" reads as (b) vs (a).
    if len(variants) > 1:
        base_variant = variants[0]
        print(f"[train-bench] ========== DELTA vs {base_variant} ==========")
        ok = {
            (v, lay): d
            for _tag, v, lay, status, d in results
            if status == "ok" and d
        }
        for tp, cp in layouts:
            base = ok.get((base_variant, (tp, cp)))
            if base is None:
                continue
            for variant in variants[1:]:
                cur = ok.get((variant, (tp, cp)))
                if cur is None:
                    continue
                d_t = cur["median_s"] - base["median_s"]
                d_act = cur["act_gb"] - base["act_gb"]
                d_peak = cur["peak_gb"] - base["peak_gb"]
                print(
                    f"[train-bench]   tp{tp}_cp{cp} {variant}: "
                    f"step {d_t:+.2f}s ({100 * d_t / base['median_s']:+.1f}%)  "
                    f"activation {d_act:+.3f} GiB ({100 * d_act / max(base['act_gb'], 1e-9):+.1f}%)  "
                    f"peak {d_peak:+.3f} GiB"
                )


if __name__ == "__main__":
    main()
