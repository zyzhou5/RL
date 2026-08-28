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
"""NeMo-owned checkpoint hooks for TQ's Mooncake backend.

The pinned Mooncake release can create one shared-filesystem ``DISK`` replica
for every object when ``mooncake_master`` is launched with ``--root_fs_dir``.
The client that performs a PUT/upsert writes that replica itself; there is no
centralized payload relay.  Disk writes finish asynchronously, and the master
marks a replica ``COMPLETE`` only after the file write succeeds.

TransferQueue v0.1.9 does not connect this facility to its storage checkpoint
hooks.  :func:`install_tq_mooncake_checkpoint_plugin` fills that gap at runtime:

* replace TQ's Mooncake bootstrap provider so the master enables direct disk
  replicas under a per-run live directory;
* replace only TQ's thin Mooncake manager with a subclass that implements
  ``save_checkpoint`` and ``load_checkpoint`` and fences key retirement;
* preserve TQ's normal Mooncake encoding and tensor transfer paths while
  fencing a repeated physical-key upsert behind its previous DISK write.

The caller must hold a checkpoint boundary that prevents referenced keys from
being updated or removed from controller save through payload copying. The
protocol targets Ray/Slurm process recovery; Mooncake's direct POSIX writer does
not fsync the live replica, so this is not a power-loss-atomic filesystem
snapshot.

The module deliberately imports only the Python standard library at import
time.  TransferQueue and Mooncake are resolved lazily by the installer, after
NeMo-RL has configured the engine environment.
"""

from __future__ import annotations

import asyncio
import atexit
import ctypes
import hashlib
import importlib
import json
import mmap
import os
import pickle
import re
import subprocess
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse

MOONCAKE_CHECKPOINT_SESSION_ENV = "NEMO_RL_TQ_MOONCAKE_CHECKPOINT_SESSION_ID"

_CONTROLLER_FILENAME = "controller_state.pkl"
_STORAGE_SUBDIR = "mooncake_storage"
_MANIFEST_FILENAME = "manifest.json"
_MANIFEST_SCHEMA_VERSION = 1
_MANIFEST_BACKEND = "mooncake_direct_disk"
_QUERY_BATCH_SIZE = 400
_COPY_BUFFER_SIZE = 8 * 1024 * 1024
_MOONCAKE_REPLICA_IS_NOT_READY = -703
_MOONCAKE_OBJECT_NOT_FOUND = -704
_UPSERT_MAX_RETRIES = 3
_UPSERT_RETRY_DELAY_S = 1.0
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GDR_CHUNK_KEY_RE = re.compile(r":c[0-9]+$")
_SESSION_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_PLUGIN_MARKER = "_nemo_rl_mooncake_checkpoint_plugin_v1"
_INSTALL_LOCK = threading.Lock()
_OWNED_MASTER_LOCK = threading.Lock()
_OWNED_MASTER_PROCESS: subprocess.Popen[Any] | None = None
_OWNED_MASTER_PID: int | None = None


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    """Return a mapping from a dict, pydantic model, or OmegaConf node."""
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    if hasattr(value, "get"):
        return value
    raise TypeError(f"{name} must be mapping-like, got {type(value).__name__}")


def _checkpoint_block(config: Any) -> Mapping[str, Any]:
    config_map = _mapping(config, name="MooncakeStore config")
    raw = config_map.get("checkpoint", {})
    if raw is None:
        return {}
    return _mapping(raw, name="MooncakeStore.checkpoint")


def mooncake_checkpoint_enabled(data_plane_config: Any) -> bool:
    """Read the opt-in without importing NeMo's torch-bearing config module."""
    config = _mapping(data_plane_config, name="data_plane config")
    if config.get("backend") != "mooncake_cpu":
        return False
    mooncake = config.get("mooncake_cpu") or {}
    mooncake_map = _mapping(mooncake, name="data_plane.mooncake_cpu")
    checkpoint = mooncake_map.get("checkpoint") or {}
    checkpoint_map = _mapping(checkpoint, name="data_plane.mooncake_cpu.checkpoint")
    return checkpoint_map.get("enabled", False) is True


@dataclass(frozen=True)
class _RuntimeConfig:
    storage_root: Path
    session_id: str
    durability_timeout_s: float
    poll_interval_s: float
    restore_batch_size: int

    @property
    def live_parent(self) -> Path:
        return self.storage_root / "live"

    @property
    def live_session_root(self) -> Path:
        return self.live_parent / self.session_id

    @classmethod
    def from_manager_config(cls, config: Any) -> _RuntimeConfig:
        checkpoint = _checkpoint_block(config)
        if checkpoint.get("enabled", False) is not True:
            raise NotImplementedError("Mooncake checkpoint plugin is disabled")

        storage_root_raw = checkpoint.get("storage_root")
        if not isinstance(storage_root_raw, str) or not storage_root_raw:
            raise RuntimeError(
                "Mooncake checkpointing requires a non-empty checkpoint.storage_root"
            )
        storage_root = Path(storage_root_raw)
        if not storage_root.is_absolute():
            raise RuntimeError(
                "Mooncake checkpointing requires an absolute checkpoint.storage_root, "
                f"got {storage_root_raw!r}"
            )
        storage_root = storage_root.resolve()

        session_id = checkpoint.get("session_id")
        if (
            not isinstance(session_id, str)
            or len(session_id) > 128
            or session_id in {".", ".."}
            or not _SESSION_RE.fullmatch(session_id)
        ):
            raise RuntimeError(
                "Mooncake checkpoint session_id must be at most 128 characters and "
                "contain only letters, digits, underscores, dots, and dashes; "
                "'.' and '..' are not valid session IDs"
            )

        durability_timeout_s = float(checkpoint.get("durability_timeout_s", 300.0))
        poll_interval_s = float(checkpoint.get("poll_interval_s", 0.1))
        restore_batch_size = int(checkpoint.get("restore_batch_size", 1))
        if durability_timeout_s <= 0:
            raise RuntimeError("durability_timeout_s must be positive")
        if poll_interval_s <= 0:
            raise RuntimeError("poll_interval_s must be positive")
        if restore_batch_size <= 0:
            raise RuntimeError("restore_batch_size must be positive")

        return cls(
            storage_root=storage_root,
            session_id=session_id,
            durability_timeout_s=durability_timeout_s,
            poll_interval_s=poll_interval_s,
            restore_batch_size=restore_batch_size,
        )


def _scalar_status(value: Any, *, partition_id: str, index: int, field: str) -> int:
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise RuntimeError(
            "Malformed TQ production status for "
            f"partition={partition_id!r}, index={index}, field={field!r}: {value!r}"
        )
    return value


def enumerate_physical_keys(controller_state: Mapping[str, Any]) -> list[str]:
    """Enumerate exactly the Mooncake keys referenced by a TQ controller cut.

    A row may have only a subset of its partition's fields produced.  Therefore
    this function consults ``production_status`` instead of taking the Cartesian
    product of rows and fields.  GDR values split by TQ are expanded to their
    physical ``:cN`` keys using the controller's per-field backend metadata.
    """
    if not isinstance(controller_state, Mapping):
        raise TypeError("controller checkpoint must contain a mapping")
    partitions = controller_state.get("partitions")
    if not isinstance(partitions, Mapping):
        raise RuntimeError("controller checkpoint is missing a partitions mapping")

    keys: list[str] = []
    seen: set[str] = set()
    for partition_id, partition in sorted(
        partitions.items(), key=lambda item: str(item[0])
    ):
        if not isinstance(partition_id, str):
            raise RuntimeError(
                f"TQ partition id must be a string, got {partition_id!r}"
            )
        global_indexes = getattr(partition, "global_indexes", None)
        field_mapping = getattr(partition, "field_name_mapping", None)
        production_status = getattr(partition, "production_status", None)
        custom_meta = getattr(partition, "field_custom_backend_meta", None)
        if not isinstance(global_indexes, (set, list, tuple)):
            raise RuntimeError(
                f"Partition {partition_id!r} has malformed global_indexes"
            )
        if not isinstance(field_mapping, Mapping):
            raise RuntimeError(
                f"Partition {partition_id!r} has malformed field_name_mapping"
            )
        if production_status is None:
            raise RuntimeError(f"Partition {partition_id!r} has no production_status")
        if custom_meta is None:
            custom_meta = {}
        if not isinstance(custom_meta, Mapping):
            raise RuntimeError(
                f"Partition {partition_id!r} has malformed field_custom_backend_meta"
            )

        fields: list[tuple[str, int]] = []
        for field_name, column in field_mapping.items():
            if not isinstance(field_name, str) or not field_name:
                raise RuntimeError(
                    f"Partition {partition_id!r} contains an invalid field name: {field_name!r}"
                )
            if isinstance(column, bool) or not isinstance(column, int) or column < 0:
                raise RuntimeError(
                    f"Partition {partition_id!r} field {field_name!r} has invalid column {column!r}"
                )
            fields.append((field_name, column))

        for global_index in sorted(global_indexes):
            if (
                isinstance(global_index, bool)
                or not isinstance(global_index, int)
                or global_index < 0
            ):
                raise RuntimeError(
                    f"Partition {partition_id!r} contains invalid global index {global_index!r}"
                )
            per_index_meta = custom_meta.get(global_index, {})
            if per_index_meta is None:
                per_index_meta = {}
            if not isinstance(per_index_meta, Mapping):
                raise RuntimeError(
                    f"Partition {partition_id!r} index {global_index} has malformed backend metadata"
                )
            for field_name, column in sorted(fields):
                try:
                    status = production_status[global_index, column]
                except Exception as error:
                    raise RuntimeError(
                        "Could not index TQ production status for "
                        f"partition={partition_id!r}, index={global_index}, field={field_name!r}"
                    ) from error
                if (
                    _scalar_status(
                        status,
                        partition_id=partition_id,
                        index=global_index,
                        field=field_name,
                    )
                    == 0
                ):
                    continue

                base_key = f"{global_index}@{field_name}"
                backend_meta = per_index_meta.get(field_name)
                if isinstance(backend_meta, Mapping) and "n_chunks" in backend_meta:
                    n_chunks = backend_meta["n_chunks"]
                    if (
                        isinstance(n_chunks, bool)
                        or not isinstance(n_chunks, int)
                        or n_chunks <= 0
                    ):
                        raise RuntimeError(
                            f"Invalid n_chunks for Mooncake key {base_key!r}: {n_chunks!r}"
                        )
                    physical_keys = [f"{base_key}:c{i}" for i in range(n_chunks)]
                else:
                    physical_keys = [base_key]

                for physical_key in physical_keys:
                    if physical_key in seen:
                        raise RuntimeError(
                            f"TQ controller metadata maps more than one value to {physical_key!r}"
                        )
                    seen.add(physical_key)
                    keys.append(physical_key)

    return sorted(keys)


@dataclass(frozen=True)
class _DiskObject:
    key: str
    path: Path
    size: int


def _status_name(status: Any) -> str:
    name = getattr(status, "name", None)
    if isinstance(name, str):
        return name
    return str(status).rsplit(".", 1)[-1]


def _query_complete_disk_objects(
    store: Any,
    keys: Sequence[str],
    *,
    timeout_s: float,
    poll_interval_s: float,
) -> list[_DiskObject]:
    """Wait until every key has exactly one COMPLETE shared-filesystem replica."""
    remaining = set(keys)
    objects: dict[str, _DiskObject] = {}
    deadline = time.monotonic() + timeout_s

    while remaining:
        pending = sorted(remaining)
        for offset in range(0, len(pending), _QUERY_BATCH_SIZE):
            batch = pending[offset : offset + _QUERY_BATCH_SIZE]
            descriptors_by_key = store.batch_get_replica_desc(batch)
            if not isinstance(descriptors_by_key, Mapping):
                raise RuntimeError(
                    "Mooncake batch_get_replica_desc returned a non-mapping result"
                )
            for key in batch:
                descriptors = descriptors_by_key.get(key, [])
                if descriptors is None:
                    descriptors = []
                complete_disk: list[Any] = []
                for descriptor in descriptors:
                    try:
                        is_disk = descriptor.is_disk_replica()
                    except Exception as error:
                        raise RuntimeError(
                            f"Malformed Mooncake replica descriptor for key {key!r}"
                        ) from error
                    if not is_disk:
                        continue
                    status_name = _status_name(descriptor.status)
                    if status_name == "FAILED":
                        raise RuntimeError(
                            f"Mooncake DISK replica failed while checkpointing key {key!r}"
                        )
                    if status_name == "COMPLETE":
                        complete_disk.append(descriptor)
                if not complete_disk:
                    continue
                if len(complete_disk) != 1:
                    raise RuntimeError(
                        f"Expected one COMPLETE Mooncake DISK replica for {key!r}, "
                        f"found {len(complete_disk)}"
                    )
                disk = complete_disk[0].get_disk_descriptor()
                file_path = getattr(disk, "file_path", None)
                object_size = getattr(disk, "object_size", None)
                if not isinstance(file_path, str) or not file_path:
                    raise RuntimeError(
                        f"Mooncake DISK replica for {key!r} has no file path"
                    )
                if (
                    isinstance(object_size, bool)
                    or not isinstance(object_size, int)
                    or object_size <= 0
                ):
                    raise RuntimeError(
                        f"Mooncake DISK replica for {key!r} has invalid size {object_size!r}"
                    )
                objects[key] = _DiskObject(
                    key=key, path=Path(file_path), size=object_size
                )
                remaining.remove(key)

        if not remaining:
            break
        delay = min(poll_interval_s, deadline - time.monotonic())
        if delay <= 0:
            sample = sorted(remaining)[:10]
            raise RuntimeError(
                "Timed out waiting for COMPLETE Mooncake DISK replicas: "
                f"pending={sample!r}, total_pending={len(remaining)}"
            )
        time.sleep(delay)

    return [objects[key] for key in keys]


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _copy_and_hash(source: Path, destination: Path, *, expected_size: int) -> str:
    before = source.stat()
    if before.st_size != expected_size:
        raise RuntimeError(
            f"Mooncake DISK file {source} has size {before.st_size}, expected {expected_size}"
        )
    digest = hashlib.sha256()
    copied = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("xb") as dst:
        while True:
            block = src.read(_COPY_BUFFER_SIZE)
            if not block:
                break
            dst.write(block)
            digest.update(block)
            copied += len(block)
        dst.flush()
        os.fsync(dst.fileno())
    after = source.stat()
    if copied != expected_size:
        raise RuntimeError(
            f"Copied {copied} bytes for Mooncake object {source}, expected {expected_size}"
        )
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(
            f"Mooncake object changed while its checkpoint copy was being made: {source}"
        )
    return digest.hexdigest()


def _write_manifest(storage_dir: Path, payload: Mapping[str, Any]) -> None:
    tmp_path = storage_dir / (_MANIFEST_FILENAME + ".tmp")
    final_path = storage_dir / _MANIFEST_FILENAME
    with tmp_path.open("x", encoding="utf-8") as manifest_file:
        json.dump(payload, manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")
        manifest_file.flush()
        os.fsync(manifest_file.fileno())
    os.replace(tmp_path, final_path)


def _reject_checkpoint_destination_over_live_storage(
    checkpoint_root: Path,
    runtime: _RuntimeConfig,
) -> None:
    """Reject a TQ destination whose atomic replacement deletes live replicas."""
    candidates = [checkpoint_root.resolve()]
    # Pinned TQ saves into ``<destination>.tmp`` and then removes/replaces the
    # destination. The manager receives only that temporary path, so recover
    # the final path for the overlap check as part of this pinned integration.
    if checkpoint_root.name.endswith(".tmp"):
        final_name = checkpoint_root.name[: -len(".tmp")]
        if final_name:
            candidates.append(checkpoint_root.with_name(final_name).resolve())

    live_session_root = runtime.live_session_root.resolve()
    for candidate in candidates:
        if _is_relative_to(live_session_root, candidate):
            raise RuntimeError(
                "TQ checkpoint destination would replace Mooncake's live "
                f"storage tree: checkpoint={candidate}, live={live_session_root}"
            )


def _save_storage_checkpoint(manager: Any, checkpoint_dir: str) -> None:
    runtime = _RuntimeConfig.from_manager_config(manager.config)
    checkpoint_root = Path(checkpoint_dir)
    _reject_checkpoint_destination_over_live_storage(checkpoint_root, runtime)
    controller_path = checkpoint_root / _CONTROLLER_FILENAME
    if not controller_path.is_file():
        raise RuntimeError(
            f"TQ controller checkpoint must exist before Mooncake storage save: {controller_path}"
        )
    with controller_path.open("rb") as controller_file:
        controller_state = pickle.load(controller_file)
    keys = enumerate_physical_keys(controller_state)
    manager_config = _mapping(manager.config, name="MooncakeStore config")
    use_gdr = manager_config.get("use_gdr", False)
    gdr_staging_buffer_mb = manager_config.get("gdr_staging_buffer_mb", 1024)
    if not isinstance(use_gdr, bool):
        raise RuntimeError(f"Mooncake use_gdr must be boolean, got {use_gdr!r}")
    if (
        isinstance(gdr_staging_buffer_mb, bool)
        or not isinstance(gdr_staging_buffer_mb, int)
        or gdr_staging_buffer_mb < 0
    ):
        raise RuntimeError(
            "Mooncake gdr_staging_buffer_mb must be a nonnegative integer, got "
            f"{gdr_staging_buffer_mb!r}"
        )
    has_chunked_gdr_objects = any(_GDR_CHUNK_KEY_RE.search(key) for key in keys)
    if has_chunked_gdr_objects and (not use_gdr or gdr_staging_buffer_mb == 0):
        raise RuntimeError(
            "TQ controller references GDR chunk objects but Mooncake GDR is not "
            "enabled with a positive staging buffer"
        )

    store = manager.storage_client._store
    disk_objects = _query_complete_disk_objects(
        store,
        keys,
        timeout_s=runtime.durability_timeout_s,
        poll_interval_s=runtime.poll_interval_s,
    )

    live_root = runtime.live_session_root.resolve()
    resolved_objects: list[tuple[_DiskObject, Path]] = []
    live_paths: dict[Path, str] = {}
    for disk_object in disk_objects:
        source = disk_object.path.resolve(strict=True)
        if not _is_relative_to(source, live_root):
            raise RuntimeError(
                f"Mooncake returned a DISK path outside this run's live root: {source}"
            )
        previous_key = live_paths.get(source)
        if previous_key is not None:
            # Mooncake sanitizes keys into filenames non-injectively. Never let
            # two catalog entries silently checkpoint the same live file.
            raise RuntimeError(
                "Mooncake DISK path collision between keys "
                f"{previous_key!r} and {disk_object.key!r}: {source}"
            )
        live_paths[source] = disk_object.key
        resolved_objects.append((disk_object, source))

    storage_dir = checkpoint_root / _STORAGE_SUBDIR
    storage_dir.mkdir(parents=True, exist_ok=False)
    entries: list[dict[str, Any]] = []
    total_size = 0
    for ordinal, (disk_object, source) in enumerate(resolved_objects):
        relative_path = Path("objects") / f"{ordinal:08d}.bin"
        digest = _copy_and_hash(
            source,
            storage_dir / relative_path,
            expected_size=disk_object.size,
        )
        entries.append(
            {
                "key": disk_object.key,
                "path": relative_path.as_posix(),
                "size": disk_object.size,
                "sha256": digest,
            }
        )
        total_size += disk_object.size

    _write_manifest(
        storage_dir,
        {
            "schema_version": _MANIFEST_SCHEMA_VERSION,
            "backend": _MANIFEST_BACKEND,
            "committed": True,
            "source_session_id": runtime.session_id,
            "object_count": len(entries),
            "total_size": total_size,
            "gdr_layout": {
                "has_chunked_objects": has_chunked_gdr_objects,
                "use_gdr": use_gdr,
                "gdr_staging_buffer_mb": gdr_staging_buffer_mb,
            },
            "objects": entries,
        },
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while True:
            block = input_file.read(_COPY_BUFFER_SIZE)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _validated_manifest(
    checkpoint_dir: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    storage_dir = Path(checkpoint_dir) / _STORAGE_SUBDIR
    manifest_path = storage_dir / _MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Mooncake checkpoint manifest not found: {manifest_path}"
        )
    with manifest_path.open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if not isinstance(manifest, Mapping):
        raise RuntimeError("Mooncake checkpoint manifest must be a mapping")
    schema_version = manifest.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != _MANIFEST_SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported Mooncake checkpoint manifest schema: {schema_version!r}"
        )
    if manifest.get("backend") != _MANIFEST_BACKEND:
        raise RuntimeError(
            f"Mooncake checkpoint backend mismatch: {manifest.get('backend')!r}"
        )
    if manifest.get("committed") is not True:
        raise RuntimeError("Mooncake checkpoint manifest is not committed")
    object_count = manifest.get("object_count")
    if (
        isinstance(object_count, bool)
        or not isinstance(object_count, int)
        or object_count < 0
    ):
        raise RuntimeError(
            "Mooncake checkpoint object_count must be a nonnegative integer"
        )
    total_size_expected = manifest.get("total_size")
    if (
        isinstance(total_size_expected, bool)
        or not isinstance(total_size_expected, int)
        or total_size_expected < 0
    ):
        raise RuntimeError(
            "Mooncake checkpoint total_size must be a nonnegative integer"
        )
    objects = manifest.get("objects")
    if not isinstance(objects, list):
        raise RuntimeError("Mooncake checkpoint manifest has no objects list")
    if object_count != len(objects):
        raise RuntimeError(
            "Mooncake checkpoint object_count does not match its objects list"
        )

    seen_keys: set[str] = set()
    seen_paths: set[Path] = set()
    total_size = 0
    validated: list[dict[str, Any]] = []
    storage_root = storage_dir.resolve()
    for entry in objects:
        if not isinstance(entry, Mapping):
            raise RuntimeError("Mooncake checkpoint object entry must be a mapping")
        key = entry.get("key")
        relative_raw = entry.get("path")
        size = entry.get("size")
        digest = entry.get("sha256")
        if not isinstance(key, str) or not key or key in seen_keys:
            raise RuntimeError(f"Invalid or duplicate Mooncake checkpoint key: {key!r}")
        if not isinstance(relative_raw, str) or not relative_raw:
            raise RuntimeError(f"Mooncake checkpoint key {key!r} has an invalid path")
        relative = Path(relative_raw)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(
                f"Mooncake checkpoint key {key!r} has an unsafe path: {relative_raw!r}"
            )
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise RuntimeError(
                f"Mooncake checkpoint key {key!r} has invalid size {size!r}"
            )
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise RuntimeError(f"Mooncake checkpoint key {key!r} has invalid sha256")
        object_path = (storage_dir / relative).resolve(strict=True)
        if not _is_relative_to(object_path, storage_root):
            raise RuntimeError(
                f"Mooncake checkpoint key {key!r} resolves outside the checkpoint"
            )
        if object_path in seen_paths:
            raise RuntimeError(
                f"Mooncake checkpoint key {key!r} resolves to a duplicate path: {relative_raw!r}"
            )
        actual_size = object_path.stat().st_size
        if actual_size != size:
            raise RuntimeError(
                f"Mooncake checkpoint key {key!r} has size {actual_size}, expected {size}"
            )
        actual_digest = _sha256_file(object_path)
        if actual_digest != digest:
            raise RuntimeError(f"Mooncake checkpoint checksum mismatch for key {key!r}")

        seen_keys.add(key)
        seen_paths.add(object_path)
        total_size += size
        validated.append(
            {"key": key, "path": object_path, "size": size, "sha256": digest}
        )

    if total_size_expected != total_size:
        raise RuntimeError("Mooncake checkpoint total_size does not match its objects")
    gdr_layout = manifest.get("gdr_layout")
    if not isinstance(gdr_layout, Mapping):
        raise RuntimeError("Mooncake checkpoint manifest has no GDR layout")
    has_chunked_objects = gdr_layout.get("has_chunked_objects")
    saved_use_gdr = gdr_layout.get("use_gdr")
    saved_staging_mb = gdr_layout.get("gdr_staging_buffer_mb")
    if not isinstance(has_chunked_objects, bool) or not isinstance(saved_use_gdr, bool):
        raise RuntimeError("Mooncake checkpoint GDR flags must be boolean")
    if (
        isinstance(saved_staging_mb, bool)
        or not isinstance(saved_staging_mb, int)
        or saved_staging_mb < 0
    ):
        raise RuntimeError(
            "Mooncake checkpoint gdr_staging_buffer_mb must be a nonnegative integer"
        )
    actual_chunked_objects = any(
        _GDR_CHUNK_KEY_RE.search(entry["key"]) for entry in validated
    )
    if has_chunked_objects != actual_chunked_objects:
        raise RuntimeError(
            "Mooncake checkpoint GDR layout does not match its object keys"
        )
    if has_chunked_objects and (not saved_use_gdr or saved_staging_mb == 0):
        raise RuntimeError("Mooncake checkpoint has an invalid chunked GDR layout")
    return validated, {
        "has_chunked_objects": has_chunked_objects,
        "use_gdr": saved_use_gdr,
        "gdr_staging_buffer_mb": saved_staging_mb,
    }


def _find_existing_keys(store: Any, keys: Sequence[str]) -> list[str]:
    """Return exact existing keys and fail closed on metadata-service errors."""
    existing: list[str] = []
    for offset in range(0, len(keys), _QUERY_BATCH_SIZE):
        batch = list(keys[offset : offset + _QUERY_BATCH_SIZE])
        statuses = store.batch_is_exist(batch)
        if (
            not isinstance(statuses, Sequence)
            or isinstance(statuses, (str, bytes, bytearray))
            or len(statuses) != len(batch)
        ):
            raise RuntimeError(
                "Mooncake batch_is_exist returned malformed statuses for "
                f"keys {batch[:10]!r}: {statuses!r}"
            )
        for key, status in zip(batch, statuses, strict=True):
            if isinstance(status, bool) or not isinstance(status, int):
                raise RuntimeError(
                    "Mooncake batch_is_exist returned a non-integer status for "
                    f"key {key!r}: {status!r}"
                )
            if status == 1:
                existing.append(key)
            elif status != 0:
                # Mooncake preserves metadata-service/RPC errors as negative
                # status codes. Treating them as absence would bypass the
                # same-key fence and restore collision preflight.
                raise RuntimeError(
                    f"Mooncake existence query failed for key {key!r}: {status}"
                )
    return existing


def _expand_physical_keys(
    keys: Sequence[str],
    custom_backend_meta: Sequence[Any] | None,
) -> list[str]:
    """Expand TQ logical keys into the Mooncake objects that must be retired."""
    if custom_backend_meta is None:
        metadata: Sequence[Any] = [None] * len(keys)
    else:
        if len(custom_backend_meta) != len(keys):
            raise RuntimeError(
                "Mooncake clear metadata length does not match its logical keys: "
                f"keys={len(keys)}, metadata={len(custom_backend_meta)}"
            )
        metadata = custom_backend_meta

    physical: list[str] = []
    seen: set[str] = set()
    for key, field_meta in zip(keys, metadata, strict=True):
        if not isinstance(key, str) or not key:
            raise RuntimeError(f"Invalid Mooncake logical key during clear: {key!r}")
        if isinstance(field_meta, Mapping) and "n_chunks" in field_meta:
            n_chunks = field_meta["n_chunks"]
            if (
                isinstance(n_chunks, bool)
                or not isinstance(n_chunks, int)
                or n_chunks <= 0
            ):
                raise RuntimeError(
                    f"Invalid n_chunks for Mooncake key {key!r}: {n_chunks!r}"
                )
            expanded = [f"{key}:c{i}" for i in range(n_chunks)]
        else:
            expanded = [key]
        for physical_key in expanded:
            if physical_key in seen:
                raise RuntimeError(
                    f"Mooncake clear maps more than one value to {physical_key!r}"
                )
            seen.add(physical_key)
            physical.append(physical_key)
    return physical


def _fence_disk_objects(
    store: Any,
    keys: Sequence[str],
    *,
    runtime: _RuntimeConfig,
) -> None:
    """Wait for known keys' direct-DISK replicas and validate their paths."""
    if not keys:
        return
    disk_objects = _query_complete_disk_objects(
        store,
        keys,
        timeout_s=runtime.durability_timeout_s,
        poll_interval_s=runtime.poll_interval_s,
    )
    live_root = runtime.live_session_root.resolve()
    for disk_object in disk_objects:
        path = disk_object.path.resolve(strict=True)
        if not _is_relative_to(path, live_root):
            raise RuntimeError(
                f"Mooncake key {disk_object.key!r} is durable outside this "
                f"run's live root: {path}"
            )
        if path.stat().st_size != disk_object.size:
            raise RuntimeError(
                f"Mooncake key {disk_object.key!r} has a truncated DISK replica"
            )


def _fence_existing_disk_objects(
    store: Any,
    keys: Sequence[str],
    *,
    runtime: _RuntimeConfig,
) -> None:
    """Wait for earlier generations before a sequential same-key upsert."""
    _fence_disk_objects(
        store,
        _find_existing_keys(store, keys),
        runtime=runtime,
    )


def _remove_physical_keys_after_write_fence(
    store: Any,
    keys: Sequence[str],
    *,
    timeout_s: float,
    poll_interval_s: float,
) -> None:
    """Converge partial removals after waiting out PROCESSING replicas.

    Mooncake's force-remove returns REPLICA_IS_NOT_READY while any replica for
    the key is still PROCESSING. ``batch_remove`` is not multi-key atomic and a
    transient RPC may arrive after some keys were removed, so every nonterminal
    status (and malformed/raised batch response) is retried idempotently. TQ
    cannot release/reuse the associated global index until this function
    returns successfully. A timeout leaves the live system recovery-required.
    """
    remaining = list(keys)
    deadline = time.monotonic() + timeout_s
    last_failure: str | None = None
    while remaining:
        try:
            results = store.batch_remove(remaining, force=True)
        except Exception as error:
            pending = list(remaining)
            last_failure = f"{type(error).__name__}: {error}"
        else:
            if (
                not isinstance(results, Sequence)
                or isinstance(results, (str, bytes, bytearray))
                or len(results) != len(remaining)
            ):
                pending = list(remaining)
                last_failure = f"malformed statuses: {results!r}"
            else:
                pending = []
                nonterminal: list[tuple[str, Any]] = []
                for key, result in zip(remaining, results, strict=True):
                    if (
                        not isinstance(result, bool)
                        and isinstance(result, int)
                        and result in (0, _MOONCAKE_OBJECT_NOT_FOUND)
                    ):
                        continue
                    pending.append(key)
                    nonterminal.append((key, result))
                last_failure = f"nonterminal statuses: {nonterminal[:10]!r}"
        if not pending:
            return

        delay = min(poll_interval_s, deadline - time.monotonic())
        if delay <= 0:
            raise RuntimeError(
                "Timed out converging Mooncake key retirement before TQ could "
                f"release its indexes: pending={pending[:10]!r}, "
                f"total_pending={len(pending)}, last_failure={last_failure}"
            )
        remaining = pending
        time.sleep(delay)


def _fence_failed_upsert_attempt(
    store: Any,
    keys: Sequence[str],
    *,
    runtime: _RuntimeConfig,
) -> None:
    """Make every failed raw attempt safe before reusing its file path."""
    existing = set(_find_existing_keys(store, keys))
    # Status 1 can describe a valid older COMPLETE generation (for example an
    # OBJECT_REPLICA_BUSY upsert). Preserve it for readers, but wait until its
    # direct-DISK replica is complete before retrying the same path.
    _fence_disk_objects(store, sorted(existing), runtime=runtime)

    # Status 0 means either genuinely absent or metadata with only PROCESSING
    # replicas. Force-remove is the master-side disambiguating fence: -704 is
    # already absent, -703 waits, and 0 retires the completed failed generation.
    uncertain = [key for key in keys if key not in existing]
    if uncertain:
        _remove_physical_keys_after_write_fence(
            store,
            uncertain,
            timeout_s=runtime.durability_timeout_s,
            poll_interval_s=runtime.poll_interval_s,
        )


def _install_sequential_upsert_fence(manager: Any, runtime: _RuntimeConfig) -> None:
    """Own TQ's raw retries so every same-path attempt is durability-fenced."""
    client = manager.storage_client
    original = getattr(client, "_batch_upsert_with_retry", None)
    if not callable(original):
        raise RuntimeError(
            "TransferQueue Mooncake client no longer exposes "
            "_batch_upsert_with_retry; cannot fence repeated key generations"
        )
    marker = "_nemo_rl_checkpoint_upsert_fence_v1"
    if getattr(client, marker, False):
        return

    def fenced_upsert(
        batch_keys: list[str], batch_ptrs: list[int], batch_sizes: list[int]
    ) -> None:
        if len(batch_ptrs) != len(batch_keys) or len(batch_sizes) != len(batch_keys):
            raise RuntimeError(
                "Mooncake upsert keys, pointers, and sizes must have equal lengths"
            )
        store = client._store
        _fence_existing_disk_objects(store, batch_keys, runtime=runtime)

        pending_indices = list(range(len(batch_keys)))
        pending_codes: list[int] = []
        for attempt in range(_UPSERT_MAX_RETRIES + 1):
            keys = [batch_keys[index] for index in pending_indices]
            ptrs = [batch_ptrs[index] for index in pending_indices]
            sizes = [batch_sizes[index] for index in pending_indices]
            try:
                results = store.batch_upsert_from(
                    keys,
                    ptrs,
                    sizes,
                    config=client.replica_config,
                )
            except BaseException:
                # The raw call may have queued writes before its binding raised.
                # Fence the unknown attempt before returning failure to TQ.
                _fence_failed_upsert_attempt(store, keys, runtime=runtime)
                raise
            if (
                not isinstance(results, Sequence)
                or isinstance(results, (str, bytes, bytearray))
                or len(results) != len(keys)
            ):
                _fence_failed_upsert_attempt(store, keys, runtime=runtime)
                raise RuntimeError(
                    "Mooncake batch_upsert_from returned malformed statuses for "
                    f"keys {keys[:10]!r}: {results!r}"
                )

            failed_positions: list[int] = []
            pending_codes = []
            for position, result in enumerate(results):
                if isinstance(result, bool) or not isinstance(result, int):
                    _fence_failed_upsert_attempt(store, keys, runtime=runtime)
                    raise RuntimeError(
                        "Mooncake batch_upsert_from returned a non-integer status "
                        f"for key {keys[position]!r}: {result!r}"
                    )
                if result != 0:
                    failed_positions.append(position)
                    pending_codes.append(result)
            if not failed_positions:
                return

            pending_indices = [
                pending_indices[position] for position in failed_positions
            ]
            failed_keys = [batch_keys[index] for index in pending_indices]
            _fence_failed_upsert_attempt(store, failed_keys, runtime=runtime)
            if attempt == _UPSERT_MAX_RETRIES:
                break
            time.sleep(_UPSERT_RETRY_DELAY_S)

        failed_keys = [batch_keys[index] for index in pending_indices]
        raise RuntimeError(
            "Mooncake batch_upsert_from failed for keys "
            f"{failed_keys[:10]!r} with error codes {pending_codes!r} after "
            f"retrying {_UPSERT_MAX_RETRIES} times"
        )

    setattr(client, "_batch_upsert_with_retry", fenced_upsert)
    setattr(client, marker, True)


def _restore_registered_batch(
    store: Any,
    entries: Sequence[Mapping[str, Any]],
    *,
    replica_config: Any,
) -> None:
    """Restore one batch through Mooncake's registered-buffer insert path.

    TQ's normal Mooncake writes use registered pointer APIs. Using the insert-
    only ``batch_put_from`` variant here avoids ``put_batch``'s dependency on
    the client's finite ``local_buffer_size``. File-backed private mappings
    avoid materializing checkpoint payloads as Python ``bytes``. Restore still
    requires global writer quiescence: the pinned client treats an
    ``OBJECT_ALREADY_EXISTS`` response as a successful idempotent PUT, so the
    descriptor preflights alone cannot close a concurrent same-key race.
    """
    keys = [entry["key"] for entry in entries]
    sizes = [entry["size"] for entry in entries]
    registered_ptrs: list[int] = []
    pointer_views: list[Any] = []
    operation_error: BaseException | None = None
    cleanup_failures: list[str] = []

    with ExitStack() as stack:
        ptrs: list[int] = []
        try:
            for entry in entries:
                checkpoint_file = stack.enter_context(entry["path"].open("rb"))
                mapping = stack.enter_context(
                    mmap.mmap(checkpoint_file.fileno(), 0, access=mmap.ACCESS_COPY)
                )
                pointer_views.append(ctypes.c_char.from_buffer(mapping))
                ptr = ctypes.addressof(pointer_views[-1])
                status = store.register_buffer(ptr, entry["size"])
                if isinstance(status, bool) or not isinstance(status, int):
                    raise RuntimeError(
                        "Mooncake register_buffer returned a non-integer status for "
                        f"key {entry['key']!r}: {status!r}"
                    )
                if status != 0:
                    raise RuntimeError(
                        "Mooncake register_buffer failed with status "
                        f"{status} for key {entry['key']!r}"
                    )
                registered_ptrs.append(ptr)
                ptrs.append(ptr)

            results = store.batch_put_from(
                keys,
                ptrs,
                sizes,
                config=replica_config,
            )
            if (
                not isinstance(results, Sequence)
                or isinstance(results, (str, bytes, bytearray))
                or len(results) != len(keys)
            ):
                raise RuntimeError(
                    "Mooncake batch_put_from returned malformed statuses for "
                    f"keys {keys[:10]!r}: {results!r}"
                )
            failures: list[tuple[str, Any]] = []
            for key, result in zip(keys, results, strict=True):
                if isinstance(result, bool) or not isinstance(result, int):
                    raise RuntimeError(
                        "Mooncake batch_put_from returned a non-integer status "
                        f"for key {key!r}: {result!r}"
                    )
                if result != 0:
                    failures.append((key, result))
            if failures:
                raise RuntimeError(
                    "Mooncake restore PUT failed: "
                    f"{failures[:10]!r}, total_failures={len(failures)}"
                )
        except BaseException as error:
            operation_error = error
        finally:
            # In the pinned Mooncake client, BatchPut finishes memory transfer
            # and copies each direct-DISK payload into an owned string before
            # returning. The later filesystem write is asynchronous, but these
            # source mappings no longer need to remain registered.
            for ptr in reversed(registered_ptrs):
                try:
                    status = store.unregister_buffer(ptr)
                    if (
                        isinstance(status, bool)
                        or not isinstance(status, int)
                        or status != 0
                    ):
                        cleanup_failures.append(f"ptr={ptr}: status={status!r}")
                except Exception as error:
                    cleanup_failures.append(f"ptr={ptr}: {error}")
            # Release ctypes' exported views before ExitStack closes the mmap
            # objects; otherwise mmap.close raises BufferError.
            pointer_views.clear()

        if operation_error is not None:
            if cleanup_failures:
                raise RuntimeError(
                    "Mooncake restore failed and registered-buffer cleanup also "
                    f"failed: {cleanup_failures[:10]!r}"
                ) from operation_error
            raise operation_error.with_traceback(operation_error.__traceback__)
        if cleanup_failures:
            raise RuntimeError(
                "Mooncake restore registered-buffer cleanup failed: "
                f"{cleanup_failures[:10]!r}"
            )


def _load_storage_checkpoint(manager: Any, checkpoint_dir: str) -> None:
    runtime = _RuntimeConfig.from_manager_config(manager.config)
    entries, saved_gdr_layout = _validated_manifest(checkpoint_dir)
    manager_config = _mapping(manager.config, name="MooncakeStore config")
    current_use_gdr = manager_config.get("use_gdr", False)
    current_staging_mb = manager_config.get("gdr_staging_buffer_mb", 1024)
    if not isinstance(current_use_gdr, bool) or (
        isinstance(current_staging_mb, bool)
        or not isinstance(current_staging_mb, int)
        or current_staging_mb < 0
    ):
        raise RuntimeError("Mooncake restore has an invalid configured GDR layout")
    # TQ chooses base keys versus :cN keys from the current effective GDR mode
    # and staging size, while controller metadata stores only n_chunks. Require
    # an exact configured layout even when this particular checkpoint contains
    # no oversized value; otherwise a tensor can cross the chunk threshold.
    if (
        current_use_gdr != saved_gdr_layout["use_gdr"]
        or current_staging_mb != saved_gdr_layout["gdr_staging_buffer_mb"]
    ):
        raise RuntimeError(
            "Mooncake checkpoint GDR layout mismatch: "
            f"saved_use_gdr={saved_gdr_layout['use_gdr']!r}, "
            f"current_use_gdr={current_use_gdr!r}, "
            f"saved_staging={saved_gdr_layout['gdr_staging_buffer_mb']!r} MiB, "
            f"current_staging={current_staging_mb!r} MiB"
        )

    # Bind the independently checksummed payload manifest to the exact TQ
    # controller cut before changing Mooncake. A self-consistent manifest that
    # omits one controller key is still not a restorable TQ checkpoint.
    controller_path = Path(checkpoint_dir) / _CONTROLLER_FILENAME
    if not controller_path.is_file():
        raise FileNotFoundError(
            f"TQ controller checkpoint not found: {controller_path}"
        )
    with controller_path.open("rb") as controller_file:
        controller_state = pickle.load(controller_file)
    controller_keys = set(enumerate_physical_keys(controller_state))
    manifest_keys = {entry["key"] for entry in entries}
    if controller_keys != manifest_keys:
        missing = sorted(controller_keys - manifest_keys)[:10]
        extra = sorted(manifest_keys - controller_keys)[:10]
        raise RuntimeError(
            "Mooncake payload manifest does not match the TQ controller checkpoint: "
            f"missing={missing!r}, extra={extra!r}"
        )

    store = manager.storage_client._store
    replica_config = manager.storage_client.replica_config

    # Preflight every target key before the first mutation. The recovery
    # coordinator must still provide a globally clean, writer-quiesced system:
    # the per-batch check below narrows but cannot close the TOCTOU race because
    # Mooncake reports an already-existing object as a successful idempotent put.
    collisions = _find_existing_keys(store, sorted(manifest_keys))
    if collisions:
        raise RuntimeError(
            "Mooncake restore requires an empty store; keys already exist: "
            f"{collisions[:10]!r}"
        )

    live_root = runtime.live_session_root.resolve()
    restored_live_paths: dict[Path, str] = {}
    for offset in range(0, len(entries), runtime.restore_batch_size):
        batch = entries[offset : offset + runtime.restore_batch_size]
        keys = [entry["key"] for entry in batch]
        collisions = _find_existing_keys(store, keys)
        if collisions:
            raise RuntimeError(
                "Mooncake restore requires an empty store; keys already exist: "
                f"{collisions[:10]!r}"
            )
        _restore_registered_batch(
            store,
            batch,
            replica_config=replica_config,
        )

        # Direct disk writes are asynchronous and retain a copy of each value
        # until StoreObject finishes.  Fence every batch before submitting the
        # next one to bound restore memory and to prevent later same-key writes
        # from racing an older queued restore write.
        disk_objects = _query_complete_disk_objects(
            store,
            keys,
            timeout_s=runtime.durability_timeout_s,
            poll_interval_s=runtime.poll_interval_s,
        )
        expected_sizes = {entry["key"]: entry["size"] for entry in batch}
        expected_digests = {entry["key"]: entry["sha256"] for entry in batch}
        for disk_object in disk_objects:
            path = disk_object.path.resolve(strict=True)
            if not _is_relative_to(path, live_root):
                raise RuntimeError(
                    f"Restored Mooncake key {disk_object.key!r} is durable outside "
                    f"this run's live root: {path}"
                )
            previous_key = restored_live_paths.get(path)
            if previous_key is not None:
                raise RuntimeError(
                    "Mooncake restored DISK path collision between keys "
                    f"{previous_key!r} and {disk_object.key!r}: {path}"
                )
            restored_live_paths[path] = disk_object.key
            if disk_object.size != expected_sizes[disk_object.key]:
                raise RuntimeError(
                    f"Restored Mooncake key {disk_object.key!r} has size "
                    f"{disk_object.size}, expected {expected_sizes[disk_object.key]}"
                )
            if path.stat().st_size != disk_object.size:
                raise RuntimeError(
                    f"Restored Mooncake key {disk_object.key!r} has a truncated DISK replica"
                )
            if _sha256_file(path) != expected_digests[disk_object.key]:
                raise RuntimeError(
                    f"Restored Mooncake checksum mismatch for key {disk_object.key!r}"
                )


def _parse_host_port(raw: Any, *, name: str) -> tuple[str, str]:
    value = str(raw).strip()
    parsed = urlparse(value if "://" in value else "//" + value)
    if not parsed.hostname or parsed.port is None:
        raise ValueError(f"Invalid {name} {value!r}; expected host:port")
    return parsed.hostname, str(parsed.port)


def _terminate_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        process.wait()
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _claim_owned_master(process: subprocess.Popen[Any]) -> None:
    global _OWNED_MASTER_PID, _OWNED_MASTER_PROCESS
    with _OWNED_MASTER_LOCK:
        current_pid = os.getpid()
        if (
            _OWNED_MASTER_PID == current_pid
            and _OWNED_MASTER_PROCESS is not None
            and _OWNED_MASTER_PROCESS.poll() is None
        ):
            raise RuntimeError(
                "This process already owns a checkpoint-enabled mooncake_master"
            )
        _OWNED_MASTER_PROCESS = process
        _OWNED_MASTER_PID = current_pid


def stop_tq_mooncake_checkpoint_master() -> None:
    """Stop only the exact ``mooncake_master`` process started by this plugin."""
    global _OWNED_MASTER_PID, _OWNED_MASTER_PROCESS
    with _OWNED_MASTER_LOCK:
        # Forked children inherit the Popen object and atexit handler. They must
        # never terminate the bootstrap parent's master.
        if _OWNED_MASTER_PID != os.getpid():
            return
        process = _OWNED_MASTER_PROCESS
        _OWNED_MASTER_PROCESS = None
        _OWNED_MASTER_PID = None
    if process is not None:
        _terminate_process(process)


atexit.register(stop_tq_mooncake_checkpoint_master)


def _initialize_checkpoint_mooncake_storage(conf: Any) -> subprocess.Popen[Any] | None:
    mooncake_config = conf.backend.MooncakeStore
    runtime = _RuntimeConfig.from_manager_config(mooncake_config)
    if not mooncake_config.auto_init:
        raise RuntimeError(
            "Experimental Mooncake checkpointing requires MooncakeStore.auto_init=true "
            "so NeMo-RL can enforce the direct-DISK master configuration"
        )
    offload = mooncake_config.get("offload", {})
    if offload and offload.get("enabled", False):
        raise RuntimeError(
            "Mooncake native TQ checkpointing uses direct DISK replicas and cannot be "
            "combined with TQ's centralized offload client"
        )

    metadata_raw = str(mooncake_config.metadata_server).strip()
    use_p2p = metadata_raw.upper() == "P2PHANDSHAKE"
    if use_p2p:
        metadata_host = metadata_port = None
    else:
        metadata_host, metadata_port = _parse_host_port(
            metadata_raw, name="Mooncake metadata_server"
        )
    master_host, master_port = _parse_host_port(
        mooncake_config.master_server_address,
        name="Mooncake master_server_address",
    )

    runtime.live_parent.mkdir(parents=True, exist_ok=True)
    try:
        runtime.live_session_root.mkdir()
    except FileExistsError as error:
        raise RuntimeError(
            "Mooncake checkpoint session already exists; each launch must use a "
            f"fresh session_id: {runtime.live_session_root}"
        ) from error
    log_dir = runtime.storage_root / "logs" / runtime.session_id
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "mooncake_master.log"

    command = [
        "mooncake_master",
        "-client_ttl=30",
        "-default_kv_lease_ttl=999999",
        "-default_kv_soft_pin_ttl=999999",
        "--allow_evict_soft_pinned_objects=false",
        f"--rpc_address={master_host}",
        f"--rpc_port={master_port}",
        "--enable_offload=false",
        "--offload_on_evict=false",
        "--eviction_high_watermark_ratio=1.0",
        "--eviction_ratio=0.0",
        "--enable_disk_eviction=false",
        f"--root_fs_dir={runtime.live_parent}",
        f"--cluster_id={runtime.session_id}",
    ]
    if not use_p2p:
        command.extend(
            [
                "--enable_http_metadata_server=true",
                f"--http_metadata_server_host={metadata_host}",
                f"--http_metadata_server_port={metadata_port}",
            ]
        )

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        try:
            _claim_owned_master(process)
        except Exception:
            _terminate_process(process)
            raise
        time.sleep(3)
    if process.poll() is not None:
        try:
            output = log_path.read_text(encoding="utf-8")
        except OSError as error:
            output = f"could not read log: {error}"
        try:
            raise RuntimeError(
                "mooncake_master exited during checkpoint-enabled startup. "
                f"Log: {log_path}\n{output}"
            )
        finally:
            stop_tq_mooncake_checkpoint_master()
    return process


def install_tq_mooncake_checkpoint_plugin(
    *, import_module: Callable[[str], Any] = importlib.import_module
) -> None:
    """Install the delegating TQ manager/bootstrap overrides once per process.

    TQ's registries are process-local, so every process that constructs a TQ
    client calls this function.  Disabled Mooncake configurations delegate to
    the original provider and retain the upstream manager's unsupported
    checkpoint behavior.
    """
    with _INSTALL_LOCK:
        manager_base = import_module("transfer_queue.storage.managers.base")
        manager_module = import_module(
            "transfer_queue.storage.managers.mooncake_manager"
        )
        provider_module = import_module("transfer_queue.storage.bootstrap.provider")
        bootstrap_module = import_module(
            "transfer_queue.storage.bootstrap.mooncake_bootstrap"
        )
        interface_module = import_module("transfer_queue.interface")

        manager_factory = manager_base.StorageManagerFactory
        provider_factory = provider_module.StorageBootstrapProvider
        manager_registry = getattr(manager_factory, "_registry", None)
        provider_registry = getattr(provider_factory, "_providers", None)
        if not isinstance(manager_registry, dict) or not isinstance(
            provider_registry, dict
        ):
            raise RuntimeError(
                "TransferQueue registry shape changed; cannot install Mooncake checkpoint plugin"
            )

        current_manager = manager_registry.get("MooncakeStore")
        current_provider = provider_registry.get("mooncakestore")
        if getattr(current_manager, _PLUGIN_MARKER, False) and getattr(
            current_provider, _PLUGIN_MARKER, False
        ):
            return
        if getattr(interface_module, "_TQ_CLIENT", None) is not None:
            raise RuntimeError(
                "Mooncake checkpoint plugin must be installed before this process initializes TQ"
            )

        original_manager = manager_module.MooncakeStorageManager
        original_provider = bootstrap_module.initialize_mooncake_storage
        if current_manager is not original_manager:
            raise RuntimeError(
                "Unexpected TransferQueue MooncakeStore manager registration; refusing to overwrite it"
            )
        if current_provider is not original_provider:
            raise RuntimeError(
                "Unexpected TransferQueue MooncakeStore bootstrap registration; refusing to overwrite it"
            )

        class NemoMooncakeCheckpointStorageManager(original_manager):
            def __init__(self, controller_info: Any, config: Any) -> None:
                super().__init__(controller_info, config)
                if _checkpoint_block(self.config).get("enabled", False) is True:
                    runtime = _RuntimeConfig.from_manager_config(self.config)
                    _install_sequential_upsert_fence(self, runtime)

            async def clear_data(self, metadata: Any) -> None:
                if _checkpoint_block(self.config).get("enabled", False) is not True:
                    await super().clear_data(metadata)
                    return

                if not metadata.field_names:
                    raise RuntimeError(
                        "Cannot clear checkpoint-enabled Mooncake data without "
                        "TQ field metadata"
                    )
                logical_keys = self._generate_keys(
                    metadata.field_names, metadata.global_indexes
                )
                _, _, custom_backend_meta = (
                    self._get_shape_type_custom_backend_meta_list(metadata)
                )
                physical_keys = _expand_physical_keys(logical_keys, custom_backend_meta)
                runtime = _RuntimeConfig.from_manager_config(self.config)
                await asyncio.to_thread(
                    _remove_physical_keys_after_write_fence,
                    self.storage_client._store,
                    physical_keys,
                    timeout_s=runtime.durability_timeout_s,
                    poll_interval_s=runtime.poll_interval_s,
                )

            async def save_checkpoint(self, checkpoint_dir: str) -> None:
                try:
                    await asyncio.to_thread(
                        _save_storage_checkpoint, self, checkpoint_dir
                    )
                except NotImplementedError:
                    raise
                except Exception as error:
                    raise RuntimeError(
                        f"Mooncake storage checkpoint save failed: {error}"
                    ) from error

            async def load_checkpoint(self, checkpoint_dir: str) -> None:
                try:
                    await asyncio.to_thread(
                        _load_storage_checkpoint, self, checkpoint_dir
                    )
                except NotImplementedError:
                    raise
                except Exception as error:
                    raise RuntimeError(
                        f"Mooncake storage checkpoint load failed: {error}"
                    ) from error

        NemoMooncakeCheckpointStorageManager.__name__ = (
            "NemoMooncakeCheckpointStorageManager"
        )
        setattr(NemoMooncakeCheckpointStorageManager, _PLUGIN_MARKER, True)

        def checkpoint_bootstrap(conf: Any) -> Any:
            if (
                _checkpoint_block(conf.backend.MooncakeStore).get("enabled", False)
                is not True
            ):
                return original_provider(conf)
            return _initialize_checkpoint_mooncake_storage(conf)

        checkpoint_bootstrap.__name__ = "initialize_nemo_checkpoint_mooncake_storage"
        setattr(checkpoint_bootstrap, _PLUGIN_MARKER, True)

        # Build and validate both replacements before touching either registry.
        # Plain assignment is the exact operation TQ's decorators perform.
        try:
            manager_registry["MooncakeStore"] = NemoMooncakeCheckpointStorageManager
            provider_registry["mooncakestore"] = checkpoint_bootstrap
        except Exception:
            manager_registry["MooncakeStore"] = original_manager
            provider_registry["mooncakestore"] = original_provider
            raise


__all__ = [
    "MOONCAKE_CHECKPOINT_SESSION_ENV",
    "enumerate_physical_keys",
    "install_tq_mooncake_checkpoint_plugin",
    "mooncake_checkpoint_enabled",
    "stop_tq_mooncake_checkpoint_master",
]
