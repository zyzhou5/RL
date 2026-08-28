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

from types import SimpleNamespace

import pytest

from nemo_rl.models.value.lm_value import Value
from nemo_rl.models.value.tq_value import TQValue


def test_shutdown_closes_attached_data_plane_after_workers_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    value = object.__new__(TQValue)
    value.dp_client = SimpleNamespace(close=lambda: events.append("data-plane"))

    def shutdown_workers(_self: Value) -> bool:
        events.append("workers")
        return True

    monkeypatch.setattr(Value, "shutdown", shutdown_workers)

    assert value.shutdown() is True
    assert events == ["workers", "data-plane"]


def test_shutdown_preserves_attached_data_plane_when_workers_remain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    value = object.__new__(TQValue)
    value.dp_client = SimpleNamespace(close=lambda: events.append("data-plane"))

    def shutdown_workers(_self: Value) -> bool:
        events.append("workers")
        return False

    monkeypatch.setattr(Value, "shutdown", shutdown_workers)

    with pytest.warns(RuntimeWarning, match="Value workers did not shut down"):
        assert value.shutdown() is False
    assert value.shutdown() is False
    assert events == ["workers"]


def test_shutdown_preserves_attached_data_plane_when_worker_teardown_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    value = object.__new__(TQValue)
    value.dp_client = SimpleNamespace(close=lambda: events.append("data-plane"))

    def fail_worker_shutdown(_self: Value) -> bool:
        events.append("workers")
        raise RuntimeError("injected worker shutdown failure")

    monkeypatch.setattr(Value, "shutdown", fail_worker_shutdown)

    with pytest.raises(RuntimeError, match="injected worker shutdown failure"):
        value.shutdown()
    assert value.shutdown() is False
    assert events == ["workers"]
