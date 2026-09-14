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

"""Four-node, phase-owned TQ bootstrap for the checkpoint microbenchmark.

The benchmark's save/load body is unchanged.  This module only connects each
fresh phase process to the job-owned Ray cluster and places eight data owners
as exactly two owners on each of four allocated nodes.
"""

from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any, Callable


OWNER_COUNT = 8
NODE_COUNT = 4
OWNERS_PER_NODE = 2
SEGMENT_BYTES = 48 * 1024**3
BUFFER_BYTES = 1024**3
READY_TIMEOUT_S = 300
NODE_RESOURCE_FRACTION = 0.001


@dataclass
class _Phase:
    backend: str
    run_dir: Path
    phase: str
    nodes: list[dict[str, Any]]
    placement_group: Any = None
    placement_group_name: str | None = None
    owners: list[Any] = field(default_factory=list)
    ready: list[dict[str, Any]] = field(default_factory=list)
    controller: dict[str, Any] | None = None
    master: Any = None
    tq: Any = None
    simple_bootstrap: Any = None
    original_simple_pg_factory: Any = None


_STATE: _Phase | None = None


def _write_json(path: Path, value: Any) -> None:
    with path.open("x") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")


def _short_hostname(value: str) -> str:
    return value.split(".", 1)[0]


def _id_string(value: Any) -> str:
    method = getattr(value, "hex", None)
    return method() if callable(method) else str(value)


def _expected_hosts() -> list[str]:
    hosts = [_short_hostname(item.strip()) for item in os.environ["BENCH_EXPECTED_HOSTS"].split(",") if item.strip()]
    if len(hosts) != NODE_COUNT or len(set(hosts)) != NODE_COUNT:
        raise ValueError("BENCH_EXPECTED_HOSTS must name exactly four distinct hosts")
    return hosts


def _resolve_nodes(ray: Any) -> list[dict[str, Any]]:
    expected = _expected_hosts()
    alive = [node for node in ray.nodes() if node.get("Alive")]
    if len(alive) != NODE_COUNT:
        raise RuntimeError(f"Expected exactly four alive Ray nodes, found {len(alive)}")
    by_host: dict[str, dict[str, Any]] = {}
    for node in alive:
        hostname = _short_hostname(str(node.get("NodeManagerHostname", "")))
        if not hostname or hostname in by_host:
            raise RuntimeError(f"Ray node hostnames are missing or duplicated: {alive!r}")
        by_host[hostname] = node
    if set(by_host) != set(expected):
        raise RuntimeError(f"Ray hosts {sorted(by_host)} differ from allocation hosts {sorted(expected)}")

    resolved = []
    for hostname in expected:
        node = by_host[hostname]
        ip = str(node["NodeManagerAddress"])
        resource_key = f"node:{ip}"
        if float(node.get("Resources", {}).get(resource_key, 0)) < NODE_RESOURCE_FRACTION * OWNERS_PER_NODE:
            raise RuntimeError(f"Ray node {hostname} does not advertise usable {resource_key!r}")
        resolved.append(
            {
                "hostname": hostname,
                "ip": ip,
                "node_id": _id_string(node["NodeID"]),
                "resource_key": resource_key,
            }
        )
    return resolved


def _create_placement_group(ray: Any, state: _Phase, actors: int, cpus: int) -> Any:
    if actors != OWNER_COUNT or cpus != 1:
        raise ValueError(f"Expected eight one-CPU data owners, got actors={actors} cpus={cpus}")
    if state.placement_group is not None:
        raise RuntimeError("This phase already owns a placement group")
    bundles = []
    for node in state.nodes:
        bundles.extend(
            {"CPU": 1, node["resource_key"]: NODE_RESOURCE_FRACTION}
            for _ in range(OWNERS_PER_NODE)
        )
    state.placement_group_name = f"tq-checkpoint-{state.backend}-{state.phase}-{os.getpid()}"
    state.placement_group = ray.util.placement_group(
        bundles,
        strategy="PACK",
        name=state.placement_group_name,
    )
    ray.get(state.placement_group.ready(), timeout=READY_TIMEOUT_S)
    return state.placement_group


def _assert_no_tq_controller(ray: Any) -> None:
    try:
        ray.get_actor("TransferQueueController", namespace="transfer_queue")
    except ValueError:
        return
    raise RuntimeError(
        "A TransferQueueController already exists in this job-owned Ray cluster; "
        "refusing to attach to storage from another phase"
    )


def _init_phase_tq(ray: Any, state: _Phase, tq: Any, conf: Any) -> Any:
    """Create this phase's controller on the benchmark driver node."""
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    from transfer_queue import interface as tq_interface

    driver_ip = ray.util.get_node_ip_address()
    matches = [node for node in state.nodes if node["ip"] == driver_ip]
    if len(matches) != 1:
        raise RuntimeError(f"Benchmark driver IP {driver_ip} is not one allocated Ray node")
    driver_node = matches[0]
    original = tq_interface.TransferQueueController
    strategy = NodeAffinitySchedulingStrategy(node_id=driver_node["node_id"], soft=False)

    class _PinnedController:
        @staticmethod
        def options(**kwargs: Any) -> Any:
            if "scheduling_strategy" in kwargs:
                raise RuntimeError("TQ unexpectedly supplied its own controller scheduling strategy")
            return original.options(scheduling_strategy=strategy, **kwargs)

    tq_interface.TransferQueueController = _PinnedController
    try:
        resolved = tq.init(conf=conf)
    finally:
        tq_interface.TransferQueueController = original
    handle = tq_interface._TQ_CONTROLLER
    info = ray.get(handle.get_zmq_server_info.remote(), timeout=READY_TIMEOUT_S)
    if str(info.ip) != driver_ip:
        raise RuntimeError(f"TQ controller advertised {info.ip}, expected driver IP {driver_ip}")
    state.controller = {
        "actor_id": _id_string(handle._actor_id),
        "node_id": driver_node["node_id"],
        "hostname": driver_node["hostname"],
        "node_ip": driver_ip,
        "zmq_ip": str(info.ip),
        "zmq_ports": dict(info.ports),
    }
    return resolved


def _validate_distribution(records: list[dict[str, Any]], state: _Phase) -> None:
    if len(records) != OWNER_COUNT:
        raise RuntimeError(f"Expected eight data owners, found {len(records)}")
    expected_hosts = Counter({node["hostname"]: OWNERS_PER_NODE for node in state.nodes})
    expected_nodes = Counter({node["node_id"]: OWNERS_PER_NODE for node in state.nodes})
    actual_hosts = Counter(record["hostname"] for record in records)
    actual_nodes = Counter(record["node_id"] for record in records)
    if actual_hosts != expected_hosts or actual_nodes != expected_nodes:
        raise RuntimeError(
            f"Data-owner placement differs from two per node: hosts={actual_hosts}, nodes={actual_nodes}"
        )
    for record in records:
        expected = state.nodes[record["rank"] // OWNERS_PER_NODE]
        if record["hostname"] != expected["hostname"] or record["node_id"] != expected["node_id"]:
            raise RuntimeError(f"Owner rank {record['rank']} escaped its prescribed Ray bundle")


def _prepare_mooncake_client() -> tuple[Any, Any, Any]:
    # Engine variables and the node-local bind address must precede Mooncake's eager import.
    from nemo_rl.data_plane.adapters.transfer_queue_env import configure_engine_env

    configure_engine_env({"backend": "mooncake_cpu"})
    import mooncake.store
    import transfer_queue as tq
    from nemo_rl.data_plane.adapters import transfer_queue as adapter
    from nemo_rl.data_plane.adapters import tq_mooncake_checkpoint as checkpoint
    from nemo_rl.data_plane.interfaces import MooncakeCpuConfig

    overlay = Path(os.environ["MC_BENCH_OVERLAY"]).resolve()
    if not Path(mooncake.store.__file__).resolve().is_relative_to(overlay):
        raise RuntimeError("Mooncake import is outside the recorded overlay")
    adapter._patch_mooncake_register_check()
    defaults = MooncakeCpuConfig()
    if defaults.reuse_registered_buffers:
        adapter._patch_mooncake_staging_buffers(defaults.staging_buffer_size)
    checkpoint.install_tq_mooncake_checkpoint_plugin()
    return tq, adapter, checkpoint


class _MooncakeOwner:
    def __init__(self, rank: int, torch_threads: int):
        import ray
        import torch

        self.rank = rank
        self.local_ip = ray.util.get_node_ip_address()
        os.environ["MC_TCP_BIND_ADDRESS"] = self.local_ip
        torch.set_num_threads(torch_threads)
        tq, _, _ = _prepare_mooncake_client()
        tq.init()

    def ready(self) -> dict[str, Any]:
        import ray
        import transfer_queue as tq

        manager = tq.get_client().storage_manager
        participant = manager._checkpoint_participant
        if participant is None:
            raise RuntimeError("Mooncake owner has no checkpoint participant")
        client = manager.storage_client
        if client.global_segment_size != SEGMENT_BYTES:
            raise RuntimeError("Mooncake owner segment capacity differs from the experiment")
        if client.replica_config.replica_num != 1:
            raise RuntimeError("Benchmark requires one memory replica per object")
        if client.local_hostname != self.local_ip or os.environ.get("MC_TCP_BIND_ADDRESS") != self.local_ip:
            raise RuntimeError("Mooncake owner does not use its Ray node-local address")
        if os.environ.get("MC_STORE_MEMCPY") != "0":
            raise RuntimeError("Mooncake owner did not inherit MC_STORE_MEMCPY=0")
        context = ray.get_runtime_context()
        return {
            "rank": self.rank,
            "pid": os.getpid(),
            "hostname": _short_hostname(socket.gethostname()),
            "node_ip": self.local_ip,
            "node_id": _id_string(context.get_node_id()),
            "segment_bytes": client.global_segment_size,
            "mc_store_memcpy": os.environ.get("MC_STORE_MEMCPY"),
            "participant": asdict(participant.info),
        }

    def mooncake_checkpoint(self, *, body: dict[str, Any]) -> dict[str, Any] | None:
        from nemo_rl.data_plane.adapters.tq_mooncake_checkpoint import run_checkpoint_command

        return run_checkpoint_command(body)

    def checkpoint_profile_snapshot(self) -> dict[str, Any]:
        import checkpoint_profile

        return checkpoint_profile.snapshot()


def _start_master(state: _Phase, host: str) -> tuple[int, int]:
    binary = Path(os.environ["MC_BENCH_OVERLAY"]) / "mooncake/mooncake_master"
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"Recorded Mooncake master is not executable: {binary}")
    with ExitStack() as stack:
        listeners = [stack.enter_context(socket.socket()) for _ in range(2)]
        for listener in listeners:
            listener.bind((host, 0))
        rpc_port, http_port = [listener.getsockname()[1] for listener in listeners]
    command = [
        str(binary),
        "--client_ttl=30",
        "--default_kv_lease_ttl=999999",
        "--default_kv_soft_pin_ttl=999999",
        "--allow_evict_soft_pinned_objects=false",
        f"--rpc_address={host}",
        f"--rpc_port={rpc_port}",
        "--enable_http_metadata_server=true",
        f"--http_metadata_server_host={host}",
        f"--http_metadata_server_port={http_port}",
        "--eviction_high_watermark_ratio=1.0",
        "--eviction_ratio=0.0",
    ]
    log_path = state.run_dir / f"mooncake-{state.phase}-master.log"
    with log_path.open("x") as log:
        state.master = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    _write_json(
        state.run_dir / f"mooncake-{state.phase}-master.json",
        {"pid": state.master.pid, "command": command, "log": str(log_path)},
    )
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if state.master.poll() is not None:
            raise RuntimeError(f"Mooncake master exited {state.master.returncode}; see {log_path}")
        try:
            for port in (rpc_port, http_port):
                with socket.create_connection((host, port), timeout=1):
                    pass
        except OSError:
            time.sleep(0.2)
        else:
            return rpc_port, http_port
    raise TimeoutError(f"Mooncake master did not become ready; see {log_path}")


def _init_simple(ray: Any, state: _Phase, config: Any, config_factory: Callable[[int], Any]) -> Any:
    import transfer_queue as tq
    from transfer_queue.storage.bootstrap import simple_storage_bootstrap
    from transfer_queue import interface as tq_interface

    state.tq = tq
    state.simple_bootstrap = simple_storage_bootstrap
    state.original_simple_pg_factory = simple_storage_bootstrap.get_placement_group

    def exact_placement_group(num_ray_actors: int, num_cpus_per_actor: int = 1) -> Any:
        return _create_placement_group(ray, state, num_ray_actors, num_cpus_per_actor)

    simple_storage_bootstrap.get_placement_group = exact_placement_group
    _init_phase_tq(ray, state, tq, config_factory(config.num_storage_units))
    handles_by_name = tq_interface._TQ_STORAGE.get("SimpleStorage", {})
    if len(handles_by_name) != OWNER_COUNT:
        raise RuntimeError(f"Expected eight SimpleStorageUnit actors, found {len(handles_by_name)}")
    ordered: list[tuple[int, str, Any]] = []
    for name, handle in handles_by_name.items():
        rank = int(name.rsplit("#", 1)[1])
        ordered.append((rank, name, handle))
    ordered.sort()
    if [rank for rank, _, _ in ordered] != list(range(OWNER_COUNT)):
        raise RuntimeError("SimpleStorageUnit ranks are not exactly 0..7")
    handles = [handle for _, _, handle in ordered]
    infos = ray.get([handle.get_zmq_server_info.remote() for handle in handles], timeout=READY_TIMEOUT_S)
    node_by_ip = {node["ip"]: node for node in state.nodes}
    ready = []
    for (rank, name, handle), info in zip(ordered, infos, strict=True):
        actor_id = _id_string(handle._actor_id)
        node = node_by_ip.get(str(info.ip))
        if node is None:
            raise RuntimeError(f"SimpleStorage actor {name} advertises an unexpected node IP {info.ip}")
        ready.append(
            {
                "rank": rank,
                "name": name,
                "actor_id": actor_id,
                "node_id": node["node_id"],
                "hostname": node["hostname"],
                "node_ip": node["ip"],
                "zmq_ip": str(info.ip),
                "zmq_ports": dict(info.ports),
            }
        )
    _validate_distribution(ready, state)
    state.owners = handles
    state.ready = ready
    _write_json(
        state.run_dir / f"simple-{state.phase}-ready.json",
        {
            "backend": "simple",
            "phase": state.phase,
            "nodes": state.nodes,
            "controller": state.controller,
            "owners": ready,
        },
    )
    print(f"SIMPLE_READY phase={state.phase} owners=8 nodes=4 owners_per_node=2", flush=True)
    return tq


def _init_mooncake(ray: Any, state: _Phase, config: Any) -> Any:
    if (
        int(os.environ["MC_BENCH_SEGMENT_BYTES"]) != SEGMENT_BYTES
        or int(os.environ["MC_BENCH_BUFFER_BYTES"]) != BUFFER_BYTES
        or os.environ["MC_MOONCAKE_PROTOCOL"] != "rdma"
        or os.environ.get("MC_STORE_MEMCPY") != "0"
    ):
        raise ValueError("Mooncake capacity/transport/memcpy settings differ from the experiment")
    driver_ip = ray.util.get_node_ip_address()
    os.environ["MC_TCP_BIND_ADDRESS"] = driver_ip
    tq, adapter, checkpoint = _prepare_mooncake_client()
    state.tq = tq
    from omegaconf import OmegaConf

    rpc_port, http_port = _start_master(state, driver_ip)
    resolved = {
        "controller": {"polling_mode": True},
        "backend": {
            "storage_backend": "MooncakeStore",
            "MooncakeStore": {
                "auto_init": False,
                "global_segment_size": SEGMENT_BYTES,
                "local_buffer_size": BUFFER_BYTES,
                # Each attached process must resolve its own Ray node address.
                "local_hostname": "",
                "metadata_server": f"{driver_ip}:{http_port}",
                "master_server_address": f"{driver_ip}:{rpc_port}",
                "hard_pin": True,
                "offload": {"enabled": False},
                "use_gdr": False,
                "gdr_staging_buffer_mb": 1024,
                "checkpoint": {"enabled": True},
                **adapter._mooncake_transport_config(),
            },
        },
    }
    _init_phase_tq(ray, state, tq, OmegaConf.create(resolved))
    if tq.get_client().storage_manager.storage_client.global_segment_size != 0:
        raise RuntimeError("Ordinary Mooncake benchmark driver unexpectedly owns a storage segment")
    placement_group = _create_placement_group(ray, state, OWNER_COUNT, 1)
    owner_type = ray.remote(num_cpus=1)(_MooncakeOwner)
    for rank in range(OWNER_COUNT):
        owner = owner_type.options(
            placement_group=placement_group,
            placement_group_bundle_index=rank,
            name=f"TQCheckpointMooncakeOwner#{state.phase}#{rank}#{os.getpid()}",
        ).remote(rank, config.torch_num_threads)
        state.owners.append(owner)
    state.ready = ray.get([owner.ready.remote() for owner in state.owners], timeout=READY_TIMEOUT_S)
    _validate_distribution(state.ready, state)
    endpoints = {owner["participant"]["transport_endpoint"] for owner in state.ready}
    if len(endpoints) != OWNER_COUNT:
        raise RuntimeError("Mooncake owners have duplicate transfer endpoints")
    checkpoint.configure_checkpoint_workers(state.owners)
    _write_json(
        state.run_dir / f"mooncake-{state.phase}-ready.json",
        {
            "backend": "mooncake",
            "phase": state.phase,
            "nodes": state.nodes,
            "controller": state.controller,
            "owners": state.ready,
            "driver_node_ip": driver_ip,
            "driver_segment_bytes": 0,
            "mounted_owner_bytes": OWNER_COUNT * SEGMENT_BYTES,
            "placement_note": "Two checkpoint owners pinned to each of four Ray nodes.",
        },
    )
    print(
        f"MOONCAKE_READY phase={state.phase} owners=8 nodes=4 owners_per_node=2 "
        "capacity_gib=384 driver_segment=0",
        flush=True,
    )
    return tq


def init_tq(config: Any, simple_config_factory: Callable[[int], Any]) -> tuple[Any, Any]:
    global _STATE
    if _STATE is not None:
        raise RuntimeError("A benchmark phase is already active")
    backend = os.environ["BENCH_BACKEND"].strip().lower()
    if backend not in {"simple", "mooncake"}:
        raise ValueError("BENCH_BACKEND must be 'simple' or 'mooncake'")
    if config.num_storage_units != OWNER_COUNT:
        raise ValueError("Four-node benchmark requires exactly eight data owners")
    if not config.ray_address or config.ray_address == "local":
        raise ValueError("Four-node benchmark requires an external job-owned Ray address")
    phase = sys.argv[sys.argv.index("--_phase") + 1]
    if phase not in {"save", "load"}:
        raise ValueError(f"Unexpected benchmark phase: {phase}")

    import ray
    import torch

    torch.set_num_threads(config.torch_num_threads)
    ray.init(
        address=config.ray_address,
        namespace=f"tq-checkpoint-4n-{Path(config.run_dir).name}-{phase}",
        include_dashboard=False,
    )
    _STATE = _Phase(backend=backend, run_dir=Path(config.run_dir), phase=phase, nodes=[])
    try:
        _STATE.nodes = _resolve_nodes(ray)
        _assert_no_tq_controller(ray)
        tq = (
            _init_simple(ray, _STATE, config, simple_config_factory)
            if backend == "simple"
            else _init_mooncake(ray, _STATE, config)
        )
        return ray, tq
    except BaseException:
        close_tq(ray, _STATE.tq)
        raise


def _record_mooncake_ownership(state: _Phase) -> None:
    manifest_path = state.run_dir / "checkpoint/mooncake_storage/manifest.json"
    if state.backend != "mooncake" or state.phase != "save" or not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text())
    byte_counts: Counter[str] = Counter()
    object_counts: Counter[str] = Counter()
    if "format_version" not in manifest:
        for entry in manifest["objects"]:
            byte_counts[entry["saved_owner"]] += entry["size"]
            object_counts[entry["saved_owner"]] += 1
        shard_count = len({entry["shard"] for entry in manifest["objects"]})
    elif type(manifest["format_version"]) is int and manifest["format_version"] in (2, 3):
        for shard in manifest["shards"]:
            byte_counts[shard["saved_owner"]] += shard["payload_bytes"]
            object_counts[shard["saved_owner"]] += shard["object_count"]
        shard_count = len({shard["shard"] for shard in manifest["shards"]})
    else:
        raise RuntimeError("Unsupported Mooncake ownership evidence manifest version")
    known = {owner["participant"]["transport_endpoint"] for owner in state.ready}
    if not set(byte_counts).issubset(known):
        raise RuntimeError("Checkpoint contains payload owned outside this phase's Mooncake actors")
    _write_json(
        state.run_dir / "mooncake-save-ownership.json",
        {
            "owners_with_payload": len(byte_counts),
            "bytes_by_owner": dict(byte_counts),
            "objects_by_owner": dict(object_counts),
            "total_payload_bytes": sum(byte_counts.values()),
            "shard_count": shard_count,
        },
    )


def _wait_controller_gone(ray: Any, timeout_s: float = 30) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            ray.get_actor("TransferQueueController", namespace="transfer_queue")
        except ValueError:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("Phase-owned TransferQueueController remained after tq.close()")
        time.sleep(0.1)


def _wait_placement_group_gone(ray: Any, name: str, timeout_s: float = 30) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            ray.util.get_placement_group(name)
        except ValueError:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Phase-owned placement group {name!r} remained after removal")
        time.sleep(0.1)


def close_tq(ray: Any, tq: Any) -> None:
    global _STATE
    state, _STATE = _STATE, None
    if state is None:
        if ray is not None and ray.is_initialized():
            ray.shutdown()  # Disconnect this child; the external Ray cluster survives.
        return
    original_error = sys.exc_info()[1]
    errors: list[BaseException] = []
    if os.environ.get("BENCH_PROFILE") == "1" and tq is not None and ray.is_initialized():
        try:
            # The caller has stopped both the public API and validation timers.
            # Collect only aggregate metadata before phase-owned actors exit.
            import checkpoint_profile
            from transfer_queue import interface as tq_interface

            handles = [tq_interface._TQ_CONTROLLER, *state.owners]
            profiles = ray.get(
                [handle.checkpoint_profile_snapshot.remote() for handle in handles],
                timeout=READY_TIMEOUT_S,
            )
            _write_json(
                state.run_dir / f"{state.backend}-{state.phase}-profile.json",
                {
                    "backend": state.backend,
                    "phase": state.phase,
                    "driver": checkpoint_profile.snapshot(),
                    "controller": profiles[0],
                    "owners": profiles[1:],
                },
            )
        except BaseException as error:
            errors.append(error)
    try:
        _record_mooncake_ownership(state)
    except BaseException as error:
        errors.append(error)
    if state.backend == "mooncake" and ray is not None and ray.is_initialized():
        for owner in state.owners:
            try:
                ray.kill(owner, no_restart=True)
            except BaseException as error:
                errors.append(error)
    if tq is not None:
        try:
            tq.close()
        except BaseException as error:
            errors.append(error)
        if ray is not None and ray.is_initialized():
            try:
                _wait_controller_gone(ray)
            except BaseException as error:
                errors.append(error)
    if state.simple_bootstrap is not None and state.original_simple_pg_factory is not None:
        state.simple_bootstrap.get_placement_group = state.original_simple_pg_factory
    if state.placement_group is not None and ray is not None and ray.is_initialized():
        try:
            ray.util.remove_placement_group(state.placement_group)
            _wait_placement_group_gone(ray, state.placement_group_name)
        except BaseException as error:
            errors.append(error)
    if ray is not None and ray.is_initialized():
        try:
            ray.shutdown()  # Driver detach only; never stop the job-owned external cluster.
        except BaseException as error:
            errors.append(error)
    process = state.master
    if process is not None and process.poll() is None:
        try:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        except BaseException as error:
            errors.append(error)
    if errors:
        message = "; ".join(f"{type(error).__name__}: {error}" for error in errors)
        if original_error is not None:
            original_error.add_note(f"Phase-owned teardown errors: {message}")
        else:
            raise RuntimeError(f"Phase-owned teardown errors: {message}") from errors[0]
