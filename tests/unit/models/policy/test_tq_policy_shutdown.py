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

from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.models.policy.tq_policy import TQPolicy


def test_shutdown_closes_owner_data_plane_after_workers_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    policy = object.__new__(TQPolicy)
    policy.dp_client = SimpleNamespace(close=lambda: events.append("data-plane"))

    def shutdown_workers(_self: Policy) -> bool:
        events.append("workers")
        return True

    monkeypatch.setattr(Policy, "shutdown", shutdown_workers)

    assert policy.shutdown() is True
    assert events == ["workers", "data-plane"]


def test_shutdown_preserves_owner_data_plane_when_workers_remain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    policy = object.__new__(TQPolicy)
    policy.dp_client = SimpleNamespace(close=lambda: events.append("data-plane"))

    def shutdown_workers(_self: Policy) -> bool:
        events.append("workers")
        return False

    monkeypatch.setattr(Policy, "shutdown", shutdown_workers)

    with pytest.warns(RuntimeWarning, match="Policy workers did not shut down"):
        assert policy.shutdown() is False
    # WorkerGroup shutdown can lose its actor references after a failed kill.
    # A later explicit call or Policy.__del__ must not reinterpret that as a
    # successful teardown and close the owner.
    assert policy.shutdown() is False
    assert events == ["workers"]


def test_shutdown_preserves_owner_data_plane_when_worker_teardown_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    policy = object.__new__(TQPolicy)
    policy.dp_client = SimpleNamespace(close=lambda: events.append("data-plane"))

    def fail_worker_shutdown(_self: Policy) -> bool:
        events.append("workers")
        raise RuntimeError("injected worker shutdown failure")

    monkeypatch.setattr(Policy, "shutdown", fail_worker_shutdown)

    with pytest.raises(RuntimeError, match="injected worker shutdown failure"):
        policy.shutdown()
    assert policy.shutdown() is False
    assert events == ["workers"]


def test_driver_can_block_destructor_driven_owner_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    policy = object.__new__(TQPolicy)
    policy.dp_client = SimpleNamespace(close=lambda: events.append("data-plane"))
    monkeypatch.setattr(
        Policy,
        "shutdown",
        lambda _self: events.append("workers") or True,
    )

    with pytest.warns(RuntimeWarning, match="controller may still be alive"):
        policy.preserve_data_plane_on_shutdown("controller may still be alive")

    assert policy.shutdown() is False
    assert events == []
