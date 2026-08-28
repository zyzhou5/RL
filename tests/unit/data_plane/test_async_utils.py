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

"""Tests for local and Ray-style async data-plane dispatch."""

import asyncio
import threading

import pytest

from nemo_rl.data_plane.async_utils import call_data_plane


class _LocalClient:
    def thread_id(self) -> int:
        return threading.get_ident()

    async def async_value(self, *, value: int) -> int:
        return value


class _RemoteMethod:
    def __init__(self) -> None:
        self.calls: list[int] = []

    async def remote(self, *, value: int) -> int:
        self.calls.append(value)
        return value


class _RemoteClient:
    def __init__(self) -> None:
        self.value = _RemoteMethod()


def test_sync_call_stays_inline_by_default() -> None:
    caller_thread_id = threading.get_ident()

    result = asyncio.run(call_data_plane(_LocalClient(), "thread_id"))

    assert result == caller_thread_id


def test_sync_call_can_be_offloaded() -> None:
    caller_thread_id = threading.get_ident()

    result = asyncio.run(
        call_data_plane(_LocalClient(), "thread_id", offload_sync=True)
    )

    assert result != caller_thread_id


def test_cancelled_offload_drains_worker_thread_before_propagating() -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    class _BlockingClient:
        def mutate(self) -> None:
            started.set()
            try:
                assert release.wait(timeout=30.0), "test never released mutation"
            finally:
                finished.set()

    async def _main() -> tuple[bool, bool]:
        task = asyncio.create_task(
            call_data_plane(_BlockingClient(), "mutate", offload_sync=True)
        )
        assert await asyncio.to_thread(started.wait, 30.0)
        task.cancel()
        await asyncio.sleep(0.05)
        done_before_release = task.done()
        finished_before_release = finished.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        return done_before_release, finished_before_release

    try:
        done_before_release, finished_before_release = asyncio.run(_main())
    finally:
        release.set()

    assert not done_before_release
    assert not finished_before_release
    assert finished.is_set()


def test_repeated_cancellation_cannot_detach_offloaded_worker_thread() -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    class _BlockingClient:
        def mutate(self) -> None:
            started.set()
            try:
                assert release.wait(timeout=30.0), "test never released mutation"
            finally:
                finished.set()

    async def _main() -> tuple[bool, bool]:
        task = asyncio.create_task(
            call_data_plane(_BlockingClient(), "mutate", offload_sync=True)
        )
        assert await asyncio.to_thread(started.wait, 30.0)

        task.cancel()
        await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.sleep(0.05)
        done_before_release = task.done()
        finished_before_release = finished.is_set()

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        return done_before_release, finished_before_release

    try:
        done_before_release, finished_before_release = asyncio.run(_main())
    finally:
        release.set()

    assert not done_before_release
    assert not finished_before_release
    assert finished.is_set()


def test_local_coroutine_result_is_awaited() -> None:
    result = asyncio.run(call_data_plane(_LocalClient(), "async_value", value=7))

    assert result == 7


def test_ray_style_remote_result_is_awaited() -> None:
    client = _RemoteClient()

    result = asyncio.run(call_data_plane(client, "value", value=11))

    assert result == 11
    assert client.value.calls == [11]
