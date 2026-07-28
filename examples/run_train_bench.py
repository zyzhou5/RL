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

Replays a dumped GRPO training batch (see the NRL_DUMP_TRAIN_BATCH hook in
nemo_rl/algorithms/grpo.py) through policy.train() under a sweep of
(tp, cp) layouts, with no generation / SGLang involvement at all. Each layout
gets a fresh Policy (fresh megatron worker group); OOMs are caught and
recorded as results rather than aborting the sweep.

Env inputs:
  NRL_TRAIN_BENCH_BATCH    path to the dumped batch .pt        (required)
  NRL_TRAIN_BENCH_LAYOUTS  comma list of TPxCP, e.g. "8x4,4x4" (default 8x4)
  NRL_TRAIN_BENCH_ITERS    timed iterations per layout          (default 4)
  DIFFU_CP_LOCAL_Q         forwarded to megatron workers        (default "1")
"""

import argparse
import copy
import os
import pprint
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
    n_iters = int(os.environ.get("NRL_TRAIN_BENCH_ITERS", "4"))

    print(f"[train-bench] batch: {batch_path}")
    print(f"[train-bench] layouts (tp x cp): {layouts}, timed iters: {n_iters}")

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
    print(f"[train-bench] batch loaded: {n_samples} samples, keys={sorted(raw.keys())}")

    results = []
    for tp, cp in layouts:
        dp = world_size // (tp * cp)
        tag = f"tp{tp}_cp{cp}_dp{dp}"
        if world_size % (tp * cp) != 0 or n_samples % dp != 0:
            print(f"[train-bench] SKIP {tag}: indivisible layout")
            results.append((tag, "skip", None))
            continue

        pol_cfg = copy.deepcopy(config["policy"])
        pol_cfg["megatron_cfg"]["tensor_model_parallel_size"] = tp
        pol_cfg["megatron_cfg"]["context_parallel_size"] = cp
        # Policy() asserts train_iters is set (grpo.setup computes it from the
        # dataloader to size the LR schedule); the bench just needs a horizon.
        pol_cfg["megatron_cfg"]["train_iters"] = max(100, n_iters + 1)
        env_vars = dict(pol_cfg["megatron_cfg"].get("env_vars") or {})
        env_vars["DIFFU_CP_LOCAL_Q"] = os.environ.get("DIFFU_CP_LOCAL_Q", "1")
        # Block-aware CP (stage b): noisy K/V stays rank-local, only the clean
        # section is gathered. Must reach the megatron workers, where BOTH the
        # data path and the attention layer read it.
        if os.environ.get("DIFFU_CP_BLOCK_AWARE"):
            env_vars["DIFFU_CP_BLOCK_AWARE"] = os.environ["DIFFU_CP_BLOCK_AWARE"]
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

            # Static floor: measured AFTER warmup so allocator caches and the
            # optimizer state are already resident. Peak is reset here so it is
            # attributable to the timed iterations alone.
            static = _worker_memory(policy, "get_memory_stats")
            _worker_memory(policy, "reset_peak_memory_stats")

            times = []
            for i in range(n_iters):
                t0 = time.perf_counter()
                policy.train(train_data, loss_fn)
                times.append(time.perf_counter() - t0)
                print(f"[train-bench] {tag}: iter {i + 1}/{n_iters} {times[-1]:.1f}s", flush=True)

            peak = _worker_memory(policy, "get_memory_stats")
            static_gb = max((s or {}).get("allocated_gb", 0.0) for s in static)
            peak_gb = max((s or {}).get("max_allocated_gb", 0.0) for s in peak)
            reserved_gb = max((s or {}).get("max_reserved_gb", 0.0) for s in peak)
            # Activation = peak - static is the discriminator in section 4 of
            # plans/cp_kv_sharding_blockaware_ring.md: if it barely moves as cp
            # grows the run is bound by the full-length gathered K/V.
            act_gb = peak_gb - static_gb

            mean_t = sum(times) / len(times)
            results.append(
                (
                    tag,
                    "ok",
                    {
                        "init_s": init_s,
                        "iters": times,
                        "mean_s": mean_t,
                        "static_gb": static_gb,
                        "peak_gb": peak_gb,
                        "act_gb": act_gb,
                        "reserved_gb": reserved_gb,
                    },
                )
            )
            print(
                f"[train-bench] RESULT {tag}: mean {mean_t:.1f}s over {n_iters} iters "
                f"(init {init_s:.0f}s) | static {static_gb:.2f} GiB, peak {peak_gb:.2f} GiB, "
                f"activation {act_gb:.2f} GiB, max_reserved {reserved_gb:.2f} GiB",
                flush=True,
            )
        except Exception as e:
            traceback.print_exc()
            results.append((tag, f"FAILED: {type(e).__name__}", None))
            print(f"[train-bench] RESULT {tag}: FAILED {type(e).__name__}: {e}", flush=True)
        finally:
            if policy is not None:
                try:
                    policy.shutdown()
                except Exception:
                    traceback.print_exc()

    print("[train-bench] ================ SUMMARY ================")
    for tag, status, detail in results:
        if detail:
            print(
                f"[train-bench]   {tag}: mean {detail['mean_s']:.1f}s  "
                f"static {detail['static_gb']:.2f} GiB  peak {detail['peak_gb']:.2f} GiB  "
                f"activation {detail['act_gb']:.2f} GiB"
            )
        else:
            print(f"[train-bench]   {tag}: {status}")


if __name__ == "__main__":
    main()
