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

"""Async dispatch helpers for local and Ray data-plane clients."""

from __future__ import annotations

import asyncio
from typing import Any


async def drain_task_until_done(task: asyncio.Future[Any]) -> bool:
    """Wait for ``task`` to finish without letting caller cancellation detach it.

    ``asyncio.shield`` protects a child from one cancellation, but the await used
    to drain that child can itself be cancelled again. Keep shielding until the
    child is actually done. Child exceptions are observed here and remain
    available through ``task.result()`` to callers that need to propagate them.

    Returns:
        True when at least one cancellation of the waiting task was deferred.
    """
    cancellation_deferred = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            # A shielded child is not cancelled when its waiter is cancelled.
            # If the child cancelled itself, it is already terminal; otherwise
            # this was another cancellation of the waiter and must be deferred.
            if task.cancelled():
                break
            cancellation_deferred = True
        except BaseException:
            # The await observes the child exception. The caller can inspect or
            # re-raise it with task.result() after this helper returns.
            break

    if task.done() and not task.cancelled():
        # Also observe a result/exception when the task completed between the
        # loop condition and the shielded await.
        task.exception()
    return cancellation_deferred


async def call_data_plane(
    client: Any,
    method_name: str,
    *,
    offload_sync: bool = False,
    **kwargs: Any,
) -> Any:
    """Call a local data-plane client or a Ray actor exposing its methods.

    Synchronous offloading is opt-in because it allows the actor event loop to
    issue other calls while this one is running. Callers should enable it only
    when that concurrency is supported or externally serialized.

    Args:
        client: Local ``DataPlaneClient`` or Ray actor handle.
        method_name: Data-plane method to invoke.
        offload_sync: Run a synchronous local implementation in a worker
            thread. Ray methods are already asynchronous and ignore this flag.
        **kwargs: Keyword arguments forwarded to the data-plane method.

    Returns:
        The method result after awaiting Ray or coroutine results.
    """
    method = getattr(client, method_name)
    remote = getattr(method, "remote", None)
    if remote is not None:
        return await remote(**kwargs)
    if offload_sync:
        # Cancelling an asyncio.to_thread await does not stop its worker
        # thread. Side-effecting data-plane operations (notably Mooncake save
        # and durability-fenced clear) must finish before their process-local
        # TQ client can be torn down, so preserve cancellation while draining
        # the actual synchronous call.
        async def _run_offloaded() -> Any:
            offloaded_result = await asyncio.to_thread(method, **kwargs)
            if asyncio.iscoroutine(offloaded_result):
                return await offloaded_result
            return offloaded_result

        offloaded_call = asyncio.create_task(
            _run_offloaded(),
            name=f"data-plane-{method_name}",
        )
        try:
            result = await asyncio.shield(offloaded_call)
        except asyncio.CancelledError:
            await drain_task_until_done(offloaded_call)
            raise
    else:
        result = method(**kwargs)
    if asyncio.iscoroutine(result):
        return await result
    return result
