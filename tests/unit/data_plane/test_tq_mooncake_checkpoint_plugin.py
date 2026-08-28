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
"""Unit tests for the TQ Mooncake checkpoint plugin.

This module intentionally imports neither torch nor TransferQueue. The plugin
itself is stdlib-only until its installer is called, so the persistence protocol
can also be exercised with small fakes on a cluster login node.
"""

from __future__ import annotations

import asyncio
import ctypes
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from nemo_rl.utils import tq_mooncake_checkpoint as checkpoint


class _StatusMatrix:
    def __init__(self, values: dict[tuple[int, int], int]) -> None:
        self.values = values

    def __getitem__(self, index: tuple[int, int]) -> int:
        return self.values[index]


class _Partition:
    def __init__(
        self,
        *,
        global_indexes: set[int],
        field_name_mapping: dict[str, int],
        production_status: _StatusMatrix,
        field_custom_backend_meta: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        self.global_indexes = global_indexes
        self.field_name_mapping = field_name_mapping
        self.production_status = production_status
        self.field_custom_backend_meta = field_custom_backend_meta or {}


class _ReplicaStatus:
    def __init__(self, name: str) -> None:
        self.name = name


class _DiskDescriptor:
    def __init__(self, path: Path, size: int) -> None:
        self.file_path = str(path)
        self.object_size = size


class _ReplicaDescriptor:
    def __init__(self, path: Path, size: int, *, status: str = "COMPLETE") -> None:
        self.status = _ReplicaStatus(status)
        self._disk = _DiskDescriptor(path, size)

    def is_disk_replica(self) -> bool:
        return True

    def get_disk_descriptor(self) -> _DiskDescriptor:
        return self._disk


class _StaticDescriptorStore:
    def __init__(self, descriptors: dict[str, _ReplicaDescriptor]) -> None:
        self.descriptors = descriptors

    def batch_get_replica_desc(
        self, keys: list[str]
    ) -> dict[str, list[_ReplicaDescriptor]]:
        return {key: [self.descriptors[key]] for key in keys if key in self.descriptors}

    def batch_is_exist(self, keys: list[str]) -> list[int]:
        return [int(key in self.descriptors) for key in keys]


class _RemoveStore:
    def __init__(self, responses: list[list[int]]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[list[str], bool]] = []

    def batch_remove(self, keys: list[str], *, force: bool) -> list[int]:
        self.calls.append((list(keys), force))
        if not self.responses:
            raise AssertionError("unexpected batch_remove call")
        return self.responses.pop(0)


class _FakeMooncakeClient:
    def __init__(self, store: Any) -> None:
        self._store = store
        self.replica_config = object()
        self.upsert_calls: list[tuple[list[str], list[int], list[int]]] = []

    def _batch_upsert_with_retry(
        self, keys: list[str], ptrs: list[int], sizes: list[int]
    ) -> None:
        self.upsert_calls.append((list(keys), list(ptrs), list(sizes)))


class _ClearMetadata:
    def __init__(self) -> None:
        self.field_names = ["tokens", "router_indices"]
        self.global_indexes = [7]
        self.size = 1
        self._custom_backend_meta = [
            {"router_indices": {"n_chunks": 2}, "tokens": None}
        ]


class _RestoreStore:
    def __init__(self, live_root: Path) -> None:
        self.live_root = live_root
        self.descriptors: dict[str, _ReplicaDescriptor] = {}
        self.values: dict[str, bytes] = {}
        self.put_calls = 0
        self.descriptor_calls = 0
        self.registered: dict[int, int] = {}

    def batch_get_replica_desc(
        self, keys: list[str]
    ) -> dict[str, list[_ReplicaDescriptor]]:
        self.descriptor_calls += 1
        return {key: [self.descriptors[key]] for key in keys if key in self.descriptors}

    def batch_is_exist(self, keys: list[str]) -> list[int]:
        return [int(key in self.descriptors) for key in keys]

    def register_buffer(self, ptr: int, size: int) -> int:
        self.registered[ptr] = size
        return 0

    def unregister_buffer(self, ptr: int) -> int:
        if ptr not in self.registered:
            return -1
        del self.registered[ptr]
        return 0

    def _prepare_payload(self, ptr: int, size: int) -> bytes:
        return ctypes.string_at(ptr, size)

    def _object_path(self) -> Path:
        return self.live_root / f"restored-{len(self.values):08d}.bin"

    def batch_put_from(
        self,
        keys: list[str],
        ptrs: list[int],
        sizes: list[int],
        *,
        config: object,
    ) -> list[int]:
        del config
        self.put_calls += 1
        for key, ptr, size in zip(keys, ptrs, sizes, strict=True):
            payload = self._prepare_payload(ptr, size)
            object_path = self._object_path()
            object_path.write_bytes(payload)
            self.values[key] = payload
            self.descriptors[key] = _ReplicaDescriptor(object_path, len(payload))
        return [0] * len(keys)


class _CorruptingRestoreStore(_RestoreStore):
    def _prepare_payload(self, ptr: int, size: int) -> bytes:
        payload = super()._prepare_payload(ptr, size)
        return bytes([payload[0] ^ 0xFF]) + payload[1:]


class _RejectingRestoreStore(_RestoreStore):
    def batch_put_from(
        self,
        keys: list[str],
        ptrs: list[int],
        sizes: list[int],
        *,
        config: object,
    ) -> list[int]:
        del ptrs, sizes, config
        self.put_calls += 1
        return [-123] * len(keys)


class _AliasingRestoreStore(_RestoreStore):
    def _object_path(self) -> Path:
        return self.live_root / "aliased.bin"


class _FakeProcess:
    def __init__(self) -> None:
        self.running = True
        self.terminate_calls = 0
        self.kill_calls = 0

    def poll(self) -> int | None:
        return None if self.running else 0

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.running = False

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return 0

    def kill(self) -> None:
        self.kill_calls += 1
        self.running = False


class _AttrDict(dict[str, Any]):
    def __getattr__(self, name: str) -> Any:
        return self[name]


def _runtime_config(storage_root: Path, session_id: str) -> dict[str, Any]:
    return {
        # The persistence fixture includes an oversized router-index value
        # represented by :cN keys, so its save/restore layout is genuinely GDR.
        "use_gdr": True,
        "gdr_staging_buffer_mb": 1024,
        "checkpoint": {
            "enabled": True,
            "storage_root": str(storage_root),
            "session_id": session_id,
            "durability_timeout_s": 1.0,
            "poll_interval_s": 0.001,
            "restore_batch_size": 1,
        },
    }


def _manager(config: dict[str, Any], store: Any) -> SimpleNamespace:
    return SimpleNamespace(
        config=config,
        storage_client=SimpleNamespace(_store=store, replica_config=object()),
    )


class PhysicalKeyTests(unittest.TestCase):
    def test_enumerates_only_produced_fields_and_expands_gdr_chunks(self) -> None:
        partition = _Partition(
            global_indexes={7, 2},
            field_name_mapping={"tokens": 0, "log_probs": 1, "router_indices": 2},
            production_status=_StatusMatrix(
                {
                    (2, 0): 1,
                    (2, 1): 1,
                    (2, 2): 0,
                    (7, 0): 1,
                    (7, 1): 0,
                    (7, 2): 1,
                }
            ),
            field_custom_backend_meta={7: {"tokens": {"n_chunks": 2}}},
        )

        self.assertEqual(
            checkpoint.enumerate_physical_keys({"partitions": {"train": partition}}),
            [
                "2@log_probs",
                "2@tokens",
                "7@router_indices",
                "7@tokens:c0",
                "7@tokens:c1",
            ],
        )

    def test_rejects_duplicate_physical_keys_across_partitions(self) -> None:
        partition = _Partition(
            global_indexes={1},
            field_name_mapping={"tokens": 0},
            production_status=_StatusMatrix({(1, 0): 1}),
        )
        with self.assertRaisesRegex(RuntimeError, "more than one value"):
            checkpoint.enumerate_physical_keys(
                {"partitions": {"first": partition, "second": partition}}
            )


class KeyLifecycleTests(unittest.TestCase):
    def test_clear_expands_gdr_chunks_without_local_cuda_state(self) -> None:
        self.assertEqual(
            checkpoint._expand_physical_keys(
                ["7@router_indices", "7@tokens"],
                [{"n_chunks": 2}, None],
            ),
            ["7@router_indices:c0", "7@router_indices:c1", "7@tokens"],
        )

    def test_clear_retries_processing_writes_and_accepts_missing_keys(self) -> None:
        store = _RemoveStore(
            [
                [checkpoint._MOONCAKE_REPLICA_IS_NOT_READY, -704, 0],
                [0],
            ]
        )
        keys = ["7@router_indices:c0", "7@router_indices:c1", "7@tokens"]

        with patch.object(checkpoint.time, "sleep", return_value=None):
            checkpoint._remove_physical_keys_after_write_fence(
                store,
                keys,
                timeout_s=1.0,
                poll_interval_s=0.001,
            )

        self.assertEqual(
            store.calls,
            [
                (keys, True),
                (["7@router_indices:c0"], True),
            ],
        )

    def test_clear_retries_every_nonterminal_status_idempotently(self) -> None:
        store = _RemoveStore([[-999], [checkpoint._MOONCAKE_OBJECT_NOT_FOUND]])
        with patch.object(checkpoint.time, "sleep", return_value=None):
            checkpoint._remove_physical_keys_after_write_fence(
                store,
                ["7@tokens"],
                timeout_s=1.0,
                poll_interval_s=0.001,
            )
        self.assertEqual(
            store.calls,
            [(["7@tokens"], True), (["7@tokens"], True)],
        )

    def test_clear_retries_after_transient_batch_exception(self) -> None:
        events: list[list[str]] = []

        class Store:
            def __init__(self) -> None:
                self.calls = 0

            def batch_remove(self, keys: list[str], *, force: bool) -> list[int]:
                if not force:
                    raise AssertionError("key retirement must be forced")
                self.calls += 1
                events.append(list(keys))
                if self.calls == 1:
                    raise RuntimeError("injected RPC failure")
                return [checkpoint._MOONCAKE_OBJECT_NOT_FOUND] * len(keys)

        with patch.object(checkpoint.time, "sleep", return_value=None):
            checkpoint._remove_physical_keys_after_write_fence(
                Store(),
                ["7@tokens"],
                timeout_s=1.0,
                poll_interval_s=0.001,
            )
        self.assertEqual(events, [["7@tokens"], ["7@tokens"]])

    def test_clear_times_out_fail_closed_on_permanent_status(self) -> None:
        store = _RemoveStore([[-999]])
        with (
            patch.object(checkpoint.time, "monotonic", side_effect=[0.0, 2.0]),
            self.assertRaisesRegex(RuntimeError, "Timed out converging"),
        ):
            checkpoint._remove_physical_keys_after_write_fence(
                store,
                ["7@tokens"],
                timeout_s=1.0,
                poll_interval_s=0.001,
            )

    def test_sequential_repeated_upsert_fences_before_transfer(self) -> None:
        events: list[tuple[str, list[str]]] = []

        class Store:
            def batch_upsert_from(
                self,
                keys: list[str],
                ptrs: list[int],
                sizes: list[int],
                *,
                config: object,
            ) -> list[int]:
                del ptrs, sizes
                del config
                events.append(("upsert", list(keys)))
                return [0] * len(keys)

        client = _FakeMooncakeClient(store=Store())
        manager = SimpleNamespace(storage_client=client)
        runtime = SimpleNamespace()

        def record_fence(store: Any, keys: list[str], *, runtime: Any) -> None:
            del store, runtime
            events.append(("fence", list(keys)))

        with patch.object(
            checkpoint, "_fence_existing_disk_objects", side_effect=record_fence
        ):
            checkpoint._install_sequential_upsert_fence(manager, runtime)
            client._batch_upsert_with_retry(["7@tokens"], [123], [16])

        self.assertEqual(
            events,
            [("fence", ["7@tokens"]), ("upsert", ["7@tokens"])],
        )

    def test_failed_processing_upsert_is_retired_before_raw_retry(self) -> None:
        events: list[str] = []

        class Store:
            def __init__(self) -> None:
                self.upsert_calls = 0
                self.remove_results = [
                    checkpoint._MOONCAKE_REPLICA_IS_NOT_READY,
                    0,
                ]

            def batch_is_exist(self, keys: list[str]) -> list[int]:
                events.append(f"exist:{','.join(keys)}")
                # No prior COMPLETE generation; after the failed raw attempt,
                # status 0 also covers its all-PROCESSING generation.
                return [0] * len(keys)

            def batch_upsert_from(
                self,
                keys: list[str],
                ptrs: list[int],
                sizes: list[int],
                *,
                config: object,
            ) -> list[int]:
                del ptrs, sizes, config
                self.upsert_calls += 1
                events.append(f"upsert:{self.upsert_calls}")
                return [-900] if self.upsert_calls == 1 else [0]

            def batch_remove(self, keys: list[str], *, force: bool) -> list[int]:
                if not force:
                    raise AssertionError("failed generation must be force-retired")
                events.append(f"remove:{self.remove_results[0]}")
                return [self.remove_results.pop(0)] * len(keys)

        client = _FakeMooncakeClient(Store())
        runtime = checkpoint._RuntimeConfig.from_manager_config(
            _runtime_config(Path("/tmp"), "retry-session")
        )
        with patch.object(checkpoint.time, "sleep", return_value=None):
            checkpoint._install_sequential_upsert_fence(
                SimpleNamespace(storage_client=client), runtime
            )
            client._batch_upsert_with_retry(["7@tokens"], [123], [16])

        self.assertEqual(
            events,
            [
                "exist:7@tokens",
                "upsert:1",
                "exist:7@tokens",
                f"remove:{checkpoint._MOONCAKE_REPLICA_IS_NOT_READY}",
                "remove:0",
                "upsert:2",
            ],
        )

    def test_busy_complete_generation_is_waited_not_removed_before_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            live_root = root / "live" / "busy-session"
            live_root.mkdir(parents=True)
            live_path = live_root / "existing.bin"
            live_path.write_bytes(b"existing-generation")
            events: list[str] = []

            class Store:
                def __init__(self) -> None:
                    self.upsert_calls = 0

                def batch_is_exist(self, keys: list[str]) -> list[int]:
                    events.append("exist")
                    return [1] * len(keys)

                def batch_get_replica_desc(
                    self, keys: list[str]
                ) -> dict[str, list[_ReplicaDescriptor]]:
                    events.append("disk-fence")
                    return {
                        key: [_ReplicaDescriptor(live_path, live_path.stat().st_size)]
                        for key in keys
                    }

                def batch_upsert_from(
                    self,
                    keys: list[str],
                    ptrs: list[int],
                    sizes: list[int],
                    *,
                    config: object,
                ) -> list[int]:
                    del keys, ptrs, sizes, config
                    self.upsert_calls += 1
                    events.append(f"upsert:{self.upsert_calls}")
                    return [-714] if self.upsert_calls == 1 else [0]

                def batch_remove(self, keys: list[str], *, force: bool) -> list[int]:
                    del keys, force
                    raise AssertionError("a valid COMPLETE generation was removed")

            client = _FakeMooncakeClient(Store())
            runtime = checkpoint._RuntimeConfig.from_manager_config(
                _runtime_config(root, "busy-session")
            )
            with patch.object(checkpoint.time, "sleep", return_value=None):
                checkpoint._install_sequential_upsert_fence(
                    SimpleNamespace(storage_client=client), runtime
                )
                client._batch_upsert_with_retry(["7@tokens"], [123], [16])

            self.assertEqual(
                events,
                [
                    "exist",
                    "disk-fence",
                    "upsert:1",
                    "exist",
                    "disk-fence",
                    "upsert:2",
                ],
            )

    def test_existence_rpc_error_fails_before_raw_upsert(self) -> None:
        class Store:
            def batch_is_exist(self, keys: list[str]) -> list[int]:
                return [-900] * len(keys)

            def batch_upsert_from(self, *_args: Any, **_kwargs: Any) -> list[int]:
                raise AssertionError("raw upsert must not bypass an RPC error")

        client = _FakeMooncakeClient(Store())
        checkpoint._install_sequential_upsert_fence(
            SimpleNamespace(storage_client=client), SimpleNamespace()
        )
        with self.assertRaisesRegex(RuntimeError, "existence query failed"):
            client._batch_upsert_with_retry(["7@tokens"], [123], [16])


class PersistenceProtocolTests(unittest.TestCase):
    def _make_saved_checkpoint(
        self, root: Path
    ) -> tuple[Path, dict[str, bytes], dict[str, Any]]:
        save_session = "save-session"
        live_root = root / "live" / save_session
        live_root.mkdir(parents=True)
        payloads = {
            "3@log_probs": b"log-probabilities",
            "3@router_indices:c0": b"router-part-zero",
            "3@router_indices:c1": b"router-part-one",
        }
        descriptors: dict[str, _ReplicaDescriptor] = {}
        for ordinal, (key, payload) in enumerate(payloads.items()):
            path = live_root / f"source-{ordinal}.bin"
            path.write_bytes(payload)
            descriptors[key] = _ReplicaDescriptor(path, len(payload))

        partition = _Partition(
            global_indexes={3},
            field_name_mapping={"log_probs": 0, "router_indices": 1},
            production_status=_StatusMatrix({(3, 0): 1, (3, 1): 1}),
            field_custom_backend_meta={3: {"router_indices": {"n_chunks": 2}}},
        )
        checkpoint_dir = root / "checkpoint.tmp"
        checkpoint_dir.mkdir()
        with (checkpoint_dir / "controller_state.pkl").open("wb") as output:
            pickle.dump({"partitions": {"train": partition}}, output)

        config = _runtime_config(root, save_session)
        checkpoint._save_storage_checkpoint(
            _manager(config, _StaticDescriptorStore(descriptors)),
            str(checkpoint_dir),
        )
        return checkpoint_dir, payloads, config

    def test_checkpoint_destination_cannot_replace_live_storage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            storage_root = root / "checkpoint"
            (storage_root / "live" / "save-session").mkdir(parents=True)
            checkpoint_tmp = root / "checkpoint.tmp"
            checkpoint_tmp.mkdir()
            with (checkpoint_tmp / "controller_state.pkl").open("wb") as output:
                pickle.dump({"partitions": {}}, output)

            with self.assertRaisesRegex(
                RuntimeError,
                "checkpoint destination would replace Mooncake's live storage",
            ):
                checkpoint._save_storage_checkpoint(
                    _manager(
                        _runtime_config(storage_root, "save-session"),
                        _StaticDescriptorStore({}),
                    ),
                    str(checkpoint_tmp),
                )

            self.assertFalse((checkpoint_tmp / "mooncake_storage").exists())

    def test_save_rejects_two_keys_that_resolve_to_one_live_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            live_root = root / "live" / "save-session"
            live_root.mkdir(parents=True)
            aliased_path = live_root / "aliased.bin"
            aliased_path.write_bytes(b"same-file")
            descriptors = {
                key: _ReplicaDescriptor(aliased_path, aliased_path.stat().st_size)
                for key in ("3@first", "3@second")
            }
            partition = _Partition(
                global_indexes={3},
                field_name_mapping={"first": 0, "second": 1},
                production_status=_StatusMatrix({(3, 0): 1, (3, 1): 1}),
            )
            checkpoint_dir = root / "checkpoint.tmp"
            checkpoint_dir.mkdir()
            with (checkpoint_dir / "controller_state.pkl").open("wb") as output:
                pickle.dump({"partitions": {"train": partition}}, output)

            with self.assertRaisesRegex(RuntimeError, "DISK path collision"):
                checkpoint._save_storage_checkpoint(
                    _manager(
                        _runtime_config(root, "save-session"),
                        _StaticDescriptorStore(descriptors),
                    ),
                    str(checkpoint_dir),
                )
            self.assertFalse((checkpoint_dir / "mooncake_storage").exists())

    def test_round_trips_every_controller_referenced_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint_dir, payloads, _ = self._make_saved_checkpoint(root)
            manifest = json.loads(
                (checkpoint_dir / "mooncake_storage" / "manifest.json").read_text()
            )
            self.assertTrue(manifest["committed"])
            self.assertEqual(manifest["object_count"], len(payloads))
            self.assertEqual(
                {entry["key"] for entry in manifest["objects"]}, set(payloads)
            )
            self.assertEqual(
                manifest["gdr_layout"],
                {
                    "has_chunked_objects": True,
                    "use_gdr": True,
                    "gdr_staging_buffer_mb": 1024,
                },
            )

            restore_session = "restore-session"
            restore_root = root / "live" / restore_session
            restore_root.mkdir()
            restore_store = _RestoreStore(restore_root)
            checkpoint._load_storage_checkpoint(
                _manager(_runtime_config(root, restore_session), restore_store),
                str(checkpoint_dir),
            )

            self.assertEqual(restore_store.values, payloads)
            self.assertEqual(restore_store.registered, {})
            # batch_size=1 also fences each asynchronous disk write before the
            # next PUT, which bounds Mooncake's queued restore memory.
            self.assertEqual(restore_store.put_calls, len(payloads))

    def test_chunked_restore_requires_the_saved_gdr_staging_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint_dir, _, _ = self._make_saved_checkpoint(root)
            restore_session = "restore-session"
            restore_root = root / "live" / restore_session
            restore_root.mkdir()
            restore_store = _RestoreStore(restore_root)
            restore_config = _runtime_config(root, restore_session)
            restore_config["gdr_staging_buffer_mb"] = 512

            with self.assertRaisesRegex(RuntimeError, "GDR layout mismatch"):
                checkpoint._load_storage_checkpoint(
                    _manager(restore_config, restore_store),
                    str(checkpoint_dir),
                )
            self.assertEqual(restore_store.put_calls, 0)

    def test_corruption_is_rejected_before_any_store_call(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint_dir, _, _ = self._make_saved_checkpoint(root)
            manifest_path = checkpoint_dir / "mooncake_storage" / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            first_object = (
                checkpoint_dir / "mooncake_storage" / manifest["objects"][0]["path"]
            )
            first_object.write_bytes(b"corrupt")

            restore_session = "restore-session"
            restore_root = root / "live" / restore_session
            restore_root.mkdir()
            restore_store = _RestoreStore(restore_root)
            with self.assertRaisesRegex(RuntimeError, "size|checksum"):
                checkpoint._load_storage_checkpoint(
                    _manager(_runtime_config(root, restore_session), restore_store),
                    str(checkpoint_dir),
                )
            self.assertEqual(restore_store.descriptor_calls, 0)
            self.assertEqual(restore_store.put_calls, 0)

    def test_manifest_must_exactly_match_controller_keys_before_restore(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint_dir, _, _ = self._make_saved_checkpoint(root)
            manifest_path = checkpoint_dir / "mooncake_storage" / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            removed = manifest["objects"].pop()
            manifest["object_count"] -= 1
            manifest["total_size"] -= removed["size"]
            manifest_path.write_text(json.dumps(manifest))

            restore_session = "restore-session"
            restore_root = root / "live" / restore_session
            restore_root.mkdir()
            restore_store = _RestoreStore(restore_root)
            with self.assertRaisesRegex(RuntimeError, "does not match.*controller"):
                checkpoint._load_storage_checkpoint(
                    _manager(_runtime_config(root, restore_session), restore_store),
                    str(checkpoint_dir),
                )
            self.assertEqual(restore_store.descriptor_calls, 0)
            self.assertEqual(restore_store.put_calls, 0)

    def test_all_key_collisions_are_checked_before_first_put(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint_dir, payloads, _ = self._make_saved_checkpoint(root)
            restore_session = "restore-session"
            restore_root = root / "live" / restore_session
            restore_root.mkdir()
            restore_store = _RestoreStore(restore_root)
            colliding_key = sorted(payloads)[-1]
            collision_path = restore_root / "collision.bin"
            collision_path.write_bytes(payloads[colliding_key])
            restore_store.descriptors[colliding_key] = _ReplicaDescriptor(
                collision_path, len(payloads[colliding_key])
            )

            with self.assertRaisesRegex(RuntimeError, "empty store"):
                checkpoint._load_storage_checkpoint(
                    _manager(_runtime_config(root, restore_session), restore_store),
                    str(checkpoint_dir),
                )
            self.assertEqual(restore_store.put_calls, 0)

    def test_restored_disk_payload_is_verified_after_complete_fence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint_dir, _, _ = self._make_saved_checkpoint(root)
            restore_session = "restore-session"
            restore_root = root / "live" / restore_session
            restore_root.mkdir()
            restore_store = _CorruptingRestoreStore(restore_root)

            with self.assertRaisesRegex(RuntimeError, "Restored.*checksum"):
                checkpoint._load_storage_checkpoint(
                    _manager(_runtime_config(root, restore_session), restore_store),
                    str(checkpoint_dir),
                )

    def test_failed_insert_releases_every_registered_checkpoint_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint_dir, _, _ = self._make_saved_checkpoint(root)
            restore_session = "restore-session"
            restore_root = root / "live" / restore_session
            restore_root.mkdir()
            restore_store = _RejectingRestoreStore(restore_root)

            with self.assertRaisesRegex(RuntimeError, "restore PUT failed"):
                checkpoint._load_storage_checkpoint(
                    _manager(_runtime_config(root, restore_session), restore_store),
                    str(checkpoint_dir),
                )

            self.assertEqual(restore_store.registered, {})
            self.assertEqual(restore_store.values, {})

    def test_restore_rejects_two_keys_that_resolve_to_one_live_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint_dir, _, _ = self._make_saved_checkpoint(root)
            restore_session = "restore-session"
            restore_root = root / "live" / restore_session
            restore_root.mkdir()
            restore_store = _AliasingRestoreStore(restore_root)

            with self.assertRaisesRegex(RuntimeError, "restored DISK path collision"):
                checkpoint._load_storage_checkpoint(
                    _manager(_runtime_config(root, restore_session), restore_store),
                    str(checkpoint_dir),
                )

            self.assertEqual(restore_store.registered, {})

    def test_dot_session_ids_cannot_escape_or_collapse_live_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for session_id in (".", ".."):
                with self.subTest(session_id=session_id):
                    with self.assertRaisesRegex(RuntimeError, "not valid session IDs"):
                        checkpoint._RuntimeConfig.from_manager_config(
                            _runtime_config(root, session_id)
                        )


class InstallerTests(unittest.TestCase):
    @staticmethod
    def _fake_modules(*, manager_override: type | None = None) -> dict[str, Any]:
        class OriginalManager:
            def __init__(self, controller_info: Any, config: dict[str, Any]) -> None:
                del controller_info
                self.config = config
                self.storage_client = config["storage_client"]

            @staticmethod
            def _generate_keys(
                field_names: list[str], global_indexes: list[int]
            ) -> list[str]:
                return [
                    f"{global_index}@{field_name}"
                    for field_name in sorted(field_names)
                    for global_index in global_indexes
                ]

            @staticmethod
            def _get_shape_type_custom_backend_meta_list(
                metadata: _ClearMetadata,
            ) -> tuple[list[Any], list[Any], list[Any]]:
                custom_meta = [
                    metadata._custom_backend_meta[row].get(field_name)
                    for field_name in sorted(metadata.field_names)
                    for row in range(metadata.size)
                ]
                return [], [], custom_meta

            async def clear_data(self, metadata: Any) -> None:
                del metadata

        def original_provider(conf: Any) -> Any:
            return conf

        manager_registry = {
            "MooncakeStore": manager_override or OriginalManager,
        }
        provider_registry = {"mooncakestore": original_provider}
        return {
            "transfer_queue.storage.managers.base": SimpleNamespace(
                StorageManagerFactory=SimpleNamespace(_registry=manager_registry)
            ),
            "transfer_queue.storage.managers.mooncake_manager": SimpleNamespace(
                MooncakeStorageManager=OriginalManager
            ),
            "transfer_queue.storage.bootstrap.provider": SimpleNamespace(
                StorageBootstrapProvider=SimpleNamespace(_providers=provider_registry)
            ),
            "transfer_queue.storage.bootstrap.mooncake_bootstrap": SimpleNamespace(
                initialize_mooncake_storage=original_provider
            ),
            "transfer_queue.interface": SimpleNamespace(_TQ_CLIENT=None),
        }

    def test_installs_both_registry_entries_and_is_idempotent(self) -> None:
        modules = self._fake_modules()
        importer = modules.__getitem__
        checkpoint.install_tq_mooncake_checkpoint_plugin(import_module=importer)
        manager = modules[
            "transfer_queue.storage.managers.base"
        ].StorageManagerFactory._registry["MooncakeStore"]
        provider = modules[
            "transfer_queue.storage.bootstrap.provider"
        ].StorageBootstrapProvider._providers["mooncakestore"]

        checkpoint.install_tq_mooncake_checkpoint_plugin(import_module=importer)
        self.assertIs(
            modules[
                "transfer_queue.storage.managers.base"
            ].StorageManagerFactory._registry["MooncakeStore"],
            manager,
        )
        self.assertIs(
            modules[
                "transfer_queue.storage.bootstrap.provider"
            ].StorageBootstrapProvider._providers["mooncakestore"],
            provider,
        )

    def test_installed_manager_fences_upserts_and_clear_before_index_release(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            modules = self._fake_modules()
            checkpoint.install_tq_mooncake_checkpoint_plugin(
                import_module=modules.__getitem__
            )
            manager_type = modules[
                "transfer_queue.storage.managers.base"
            ].StorageManagerFactory._registry["MooncakeStore"]
            store = _RemoveStore(
                [
                    [checkpoint._MOONCAKE_REPLICA_IS_NOT_READY, 0, 0],
                    [0],
                ]
            )
            client = _FakeMooncakeClient(store)
            config = {
                **_runtime_config(root, "clear-session"),
                "storage_client": client,
            }

            manager = manager_type(None, config)
            self.assertTrue(
                getattr(client, "_nemo_rl_checkpoint_upsert_fence_v1", False)
            )
            with patch.object(checkpoint.time, "sleep", return_value=None):
                asyncio.run(manager.clear_data(_ClearMetadata()))

            expected = [
                "7@router_indices:c0",
                "7@router_indices:c1",
                "7@tokens",
            ]
            self.assertEqual(
                store.calls,
                [(expected, True), (["7@router_indices:c0"], True)],
            )

    def test_refuses_to_overwrite_registry_drift(self) -> None:
        class UnexpectedManager:
            pass

        modules = self._fake_modules(manager_override=UnexpectedManager)
        with self.assertRaisesRegex(RuntimeError, "Unexpected.*manager registration"):
            checkpoint.install_tq_mooncake_checkpoint_plugin(
                import_module=modules.__getitem__
            )


class OwnedMasterTests(unittest.TestCase):
    def test_bootstrap_enables_only_direct_shared_filesystem_replicas(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mooncake = _AttrDict(
                {
                    **_runtime_config(root, "fresh-session"),
                    "auto_init": True,
                    "offload": {"enabled": False},
                    "metadata_server": "head:50050",
                    "master_server_address": "head:50051",
                }
            )
            conf = SimpleNamespace(backend=SimpleNamespace(MooncakeStore=mooncake))
            process = _FakeProcess()

            with (
                patch.object(
                    checkpoint.subprocess,
                    "Popen",
                    return_value=process,
                ) as popen,
                patch.object(checkpoint.time, "sleep", return_value=None),
            ):
                returned = checkpoint._initialize_checkpoint_mooncake_storage(conf)

            self.assertIs(returned, process)
            command = popen.call_args.args[0]
            self.assertIn(f"--root_fs_dir={root / 'live'}", command)
            self.assertIn("--cluster_id=fresh-session", command)
            self.assertIn("--enable_offload=false", command)
            self.assertIn("--enable_disk_eviction=false", command)
            self.assertNotIn("mooncake_client", command)
            self.assertTrue((root / "live" / "fresh-session").is_dir())
            checkpoint.stop_tq_mooncake_checkpoint_master()

    def test_teardown_terminates_only_the_recorded_process(self) -> None:
        process = _FakeProcess()
        checkpoint._claim_owned_master(process)
        checkpoint.stop_tq_mooncake_checkpoint_master()

        self.assertEqual(process.terminate_calls, 1)
        self.assertEqual(process.kill_calls, 0)
        # Idempotent and, unlike upstream TQ, never scans or kills by name.
        checkpoint.stop_tq_mooncake_checkpoint_master()

    def test_forked_child_cannot_terminate_parent_owned_master(self) -> None:
        process = _FakeProcess()
        checkpoint._claim_owned_master(process)
        checkpoint._OWNED_MASTER_PID = checkpoint.os.getpid() + 1
        try:
            checkpoint.stop_tq_mooncake_checkpoint_master()
            self.assertEqual(process.terminate_calls, 0)
            self.assertTrue(process.running)
        finally:
            # The fake foreign owner has no actual process to stop. Clear test
            # state directly so the module's atexit hook remains a no-op.
            checkpoint._OWNED_MASTER_PROCESS = None
            checkpoint._OWNED_MASTER_PID = None


class OptInTests(unittest.TestCase):
    def test_checkpointing_requires_explicit_mooncake_opt_in(self) -> None:
        self.assertFalse(
            checkpoint.mooncake_checkpoint_enabled(
                {"backend": "mooncake_cpu", "mooncake_cpu": {}}
            )
        )
        self.assertFalse(
            checkpoint.mooncake_checkpoint_enabled(
                {
                    "backend": "simple",
                    "mooncake_cpu": {"checkpoint": {"enabled": True}},
                }
            )
        )
        self.assertTrue(
            checkpoint.mooncake_checkpoint_enabled(
                {
                    "backend": "mooncake_cpu",
                    "mooncake_cpu": {"checkpoint": {"enabled": True}},
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
