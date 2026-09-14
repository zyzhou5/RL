#!/usr/bin/env python3
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

"""Run paired Simple/Mooncake production-grid shapes with the unchanged benchmark driver."""

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROW_SIZES = (8192, 16384, 32768)
CONTEXT_SIZES = (8192, 16384, 32768, 65536, 131072)
SHAPES = tuple((rows, context) for rows in ROW_SIZES for context in CONTEXT_SIZES)
CASE_COUNT = 2 * len(SHAPES)
FIELDS = ("backend", "rows", "context", "logical_gib", "save_s", "load_s", "correctness", "run_dir")


def commands(script, root, ray_address):
    for index, (rows, context) in enumerate(SHAPES):
        order = ("simple", "mooncake") if index % 2 == 0 else ("mooncake", "simple")
        for backend in order:
            name = f"prod-{rows}x{context}-train-ready-{backend}"
            command = [
                sys.executable, str(script), "--checkpoint-root", str(root),
                "--run-name", name, "--num-rows", str(rows),
                "--min-seq-len", str(context), "--max-seq-len", str(context),
                "--payload-profile", "train-ready", "--num-storage-units", "8",
                "--producer-mode", "quiescent", "--batch-rows", "256",
                "--verify-mode", "sample", "--verify-samples", "64",
                "--verify-batch-rows", "32", "--seed", "42",
                "--group-size", "8", "--weight-version", "0",
                "--torch-num-threads", "1", "--phase-timeout-s", "1800",
                "--ray-address", ray_address,
            ]
            yield backend, rows, context, name, command


def result_row(result, backend, rows, context, run_dir):
    load = result["load"]
    verified, restored = load["verification"], load["restored"]
    missing = ("missing_base_rows", "missing_guaranteed_producer_rows", "malformed_keys", "unexpected_keys")
    if (verified["status"] != "pass" or any(verified[key] != 0 for key in missing)
            or restored["base_rows"] != rows or restored["total_rows"] != rows
            or verified["mode"] != "sample" or verified["verified_rows"] != 64):
        raise RuntimeError("Original harness did not confirm all keys and 64 sampled rows")
    logical_bytes = rows * (context * 20 + 12)
    if restored["logical_tensor_bytes"] != logical_bytes:
        raise RuntimeError("Restored logical payload size differs from the planned shape")
    save_s = result["save"]["checkpoint"]["duration_s"]
    load_s = load["checkpoint"]["load_duration_s"]
    if not all(math.isfinite(value) and value > 0 for value in (save_s, load_s)):
        raise RuntimeError("Checkpoint times must be finite and positive")
    return dict(zip(FIELDS, (
        backend, rows, context, logical_bytes / 1024**3, save_s, load_s, "pass", str(run_dir),
    ), strict=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-script", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--ray-address", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    script, root = args.benchmark_script.resolve(), args.checkpoint_root.resolve()
    if not script.is_file():
        parser.error(f"Benchmark script not found: {script}")
    if not args.ray_address.strip():
        parser.error("Ray address must not be empty")
    plan = list(commands(script, root, args.ray_address))
    total = sum(rows * (context * 20 + 12) for _, rows, context, _, _ in plan)
    print(f"Planned cases={CASE_COUNT} repetitions=1 logical_bytes={total} logical_gib={total / 1024**3:.6f}", flush=True)
    if args.dry_run:
        for backend, _, _, _, command in plan:
            print(f"BENCH_BACKEND={backend} " + shlex.join(command))
        return
    root.mkdir(parents=True, exist_ok=False)
    logs = root / "logs"
    logs.mkdir()
    with (root / "summary.csv").open("x", newline="") as summary:
        writer = csv.DictWriter(summary, fieldnames=FIELDS)
        writer.writeheader()
        summary.flush()
        for index, (backend, rows, context, name, command) in enumerate(plan, 1):
            env = dict(os.environ, BENCH_BACKEND=backend)
            print(f"[{index}/{CASE_COUNT}] START {name}", flush=True)
            with (logs / f"{name}.log").open("x") as log:
                log.write(f"$ BENCH_BACKEND={backend} " + shlex.join(command) + "\n\n")
                log.flush()
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            run_dir = root / name
            result = json.loads((run_dir / "result.json").read_text())
            row = result_row(result, backend, rows, context, run_dir)
            writer.writerow(row)
            summary.flush()
            print(f"[{index}/{CASE_COUNT}] PASS {name} save_s={row['save_s']} load_s={row['load_s']}", flush=True)
    print(f"FOUR_NODE_SUBSET_PASS cases={CASE_COUNT} summary={root / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
