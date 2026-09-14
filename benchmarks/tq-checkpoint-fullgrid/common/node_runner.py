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

"""One process per allocated node: start Ray once and retain job-owned logs."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import socket
import subprocess
import sys
import time


def write_json(path: Path, value: object) -> None:
    with path.open("x") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")


def preflight(root: Path, host: str) -> None:
    # Optional cluster packages are loaded only inside the recorded container.
    import mooncake.store
    import psutil
    import transfer_queue

    affinity = sorted(os.sched_getaffinity(0))
    if platform.machine() != "aarch64" or len(affinity) != 128:
        raise RuntimeError(f"Unexpected architecture/CPU allocation: {platform.machine()}, {affinity}")
    if Path(sys.executable).parent != Path("/opt/nemo_rl_venv/bin"):
        raise RuntimeError(f"Unexpected interpreter: {sys.executable}")
    if not Path(transfer_queue.__file__).resolve().is_relative_to(Path(os.environ["TQ_SOURCE"]).resolve()):
        raise RuntimeError("TQ import differs from frozen source")
    if transfer_queue.__version__ != "0.1.9":
        raise RuntimeError("Unexpected TQ version")
    store = Path(mooncake.store.__file__).resolve()
    if not store.is_relative_to(Path(os.environ["MC_BENCH_OVERLAY"]).resolve()):
        raise RuntimeError("Mooncake import differs from recorded overlay")
    if importlib.metadata.version("mooncake-transfer-engine-cuda13") != "0.3.11.post1":
        raise RuntimeError("Unexpected Mooncake version")
    if not Path("/dev/infiniband").is_dir():
        raise RuntimeError("RDMA devices are unavailable")
    memlock = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    required = int(os.environ["MC_BENCH_SEGMENT_BYTES"]) + int(os.environ["MC_BENCH_BUFFER_BYTES"])
    if memlock[0] != resource.RLIM_INFINITY and memlock[0] < required:
        raise RuntimeError(f"Insufficient memlock: {memlock}")
    module = Path(os.environ["RL_SOURCE"]) / "nemo_rl/data_plane/adapters/tq_mooncake_checkpoint.py"
    if hashlib.sha256(module.read_bytes()).hexdigest() != os.environ["OPTIMIZED_MODULE_SHA256"]:
        raise RuntimeError("Checkpoint module differs from agreed optimized version")
    free = shutil.disk_usage(root).free
    if free < 3000 * 1024**3:
        raise RuntimeError(f"Insufficient checkpoint filesystem headroom: {free}")
    report = {
        "hostname": host, "python": sys.version, "executable": sys.executable,
        "architecture": platform.machine(), "cpu_affinity": affinity,
        "host_memory_total_bytes": psutil.virtual_memory().total,
        "filesystem_free_bytes": free, "memlock_limits": memlock,
        "packages": {name: importlib.metadata.version(name) for name in (
            "numpy", "omegaconf", "psutil", "ray", "tensordict", "torch", "mooncake-transfer-engine-cuda13")},
        "transfer_queue": {"version": transfer_queue.__version__, "path": transfer_queue.__file__},
        "mooncake_store": {"path": str(store), "sha256": hashlib.sha256(store.read_bytes()).hexdigest()},
        "environment": {key: os.environ.get(key) for key in (
            "SLURM_JOB_ID", "SLURM_CLUSTER_NAME", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE",
            "BENCH_EXPECTED_HOSTS", "RL_SOURCE", "RL_SHA", "RL_TREE", "TQ_SOURCE", "TQ_SHA",
            "BENCHMARK_SHA", "CONTAINER", "PYTHONPATH", "TQ_NUM_THREADS", "OMP_NUM_THREADS", "BENCH_PROFILE",
            "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAY_TMPDIR", "TMPDIR",
            "RAY_OVERRIDE_RESOURCES", "RAY_ENABLE_UV_RUN_RUNTIME_ENV", "CUDA_VISIBLE_DEVICES",
            "MC_MOONCAKE_PROTOCOL", "MC_MOONCAKE_DEVICE", "MC_GID_INDEX", "MC_RDMA_PORT",
            "MC_STORE_MEMCPY", "MC_TCP_BIND_ADDRESS", "MC_BENCH_SEGMENT_BYTES", "MC_BENCH_BUFFER_BYTES",
            "MC_BENCH_WHEEL", "MC_BENCH_WHEEL_SHA256", "OPTIMIZED_MODULE_SHA256")},
    }
    write_json(root / "evidence" / f"runtime-{host}.json", report)


def wait_file(path: Path, timeout: int = 600) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            value = path.read_text().strip()
            if value:
                return value
        time.sleep(1)
    raise TimeoutError(f"Timed out waiting for {path}")


def start_ray(root: Path, host: str, ip: str, address: str, head: bool) -> None:
    command = [
        sys.executable, "-m", "ray.scripts.scripts", "start",
        "--disable-usage-stats", f"--node-ip-address={ip}", "--num-cpus=128", "--num-gpus=0",
        "--min-worker-port=2000", "--max-worker-port=2999",
        "--node-manager-port=1301", "--object-manager-port=1303",
        "--runtime-env-agent-port=1305", "--dashboard-agent-grpc-port=1307",
        "--metrics-export-port=1309", "--dashboard-agent-listen-port=1311",
    ]
    if head:
        command += ["--head", "--port=1200", "--ray-client-server-port=1201",
                    "--include-dashboard=false", f"--temp-dir={os.environ['RAY_TMPDIR']}"]
    else:
        command += [f"--address={address}"]
    write_json(root / "evidence" / f"ray-command-{host}.json", command)
    with (root / "slurm" / f"ray-start-{host}.log").open("x") as log:
        subprocess.run(command, check=True, timeout=300, stdout=log, stderr=subprocess.STDOUT)


def wait_nodes(address: str, expected: list[str], root: Path) -> None:
    # This checks the allocated Ray cluster; it does not create another cluster.
    import ray

    ray.init(address=address, namespace="four-node-benchmark-launch")
    try:
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            nodes = [node for node in ray.nodes() if node["Alive"]]
            actual = sorted(node["NodeManagerHostname"].split(".")[0] for node in nodes)
            if actual == sorted(expected):
                write_json(root / "evidence" / "ray-cluster.json", nodes)
                return
            if len(nodes) > 4:
                raise RuntimeError(f"Unexpected Ray cluster membership: {actual}")
            time.sleep(2)
        raise TimeoutError(f"Ray workers did not join: {actual}, expected {expected}")
    finally:
        ray.shutdown()


def main() -> int:
    root = Path(os.environ["CAMPAIGN_ROOT"])
    host = socket.gethostname().split(".")[0]
    expected = os.environ["BENCH_EXPECTED_HOSTS"].split(",")
    if len(expected) != 4 or len(set(expected)) != 4 or host not in expected:
        raise RuntimeError(f"Unexpected allocation: {host}, {expected}")
    head = int(os.environ["SLURM_PROCID"]) == 0
    ip = socket.gethostbyname(socket.gethostname())
    os.environ["MC_TCP_BIND_ADDRESS"] = ip
    control = root / ".nrl/control"
    exit_path = control / "head-exit.json"
    result = 1
    try:
        preflight(root, host)
        if head:
            address = f"{ip}:1200"
            start_ray(root, host, ip, address, True)
            with (control / "head-address.txt").open("x") as output:
                output.write(address + "\n")
            wait_nodes(address, expected, root)
            command = [sys.executable, str(root / "run_subset.py"),
                       "--benchmark-script", str(root / "tq_checkpoint_benchmark.py"),
                       "--checkpoint-root", str(root / "checkpoints/subset"),
                       "--ray-address", address]
            result = subprocess.run(command, check=False).returncode
        else:
            address = wait_file(control / "head-address.txt")
            start_ray(root, host, ip, address, False)
            status = json.loads(wait_file(exit_path, timeout=7200))
            result = int(status["returncode"])
        return result
    finally:
        if head:
            write_json(exit_path, {"returncode": result, "hostname": host})
        logs = Path(os.environ["RAY_TMPDIR"]) / "session_latest/logs"
        if logs.is_dir():
            # Copy logs only, never plasma/spill files or checkpoints. No deletion.
            target = root / "evidence/ray" / host
            target.mkdir(parents=True, exist_ok=False)
            subprocess.run(["rsync", "-a", "--include=*/", "--include=*.log", "--include=*.out",
                            "--include=*.err", "--include=*.json", "--exclude=*",
                            str(logs) + "/", str(target) + "/"], check=False, timeout=60)
        # Slurm owns this node's Ray daemons and reaps them when the step exits.


if __name__ == "__main__":
    sys.exit(main())
