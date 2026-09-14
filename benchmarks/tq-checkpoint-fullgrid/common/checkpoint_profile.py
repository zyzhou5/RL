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

"""Benchmark-only aggregate profiling, kept outside published project code.

Records remain in memory until the bootstrap collects them after the public
checkpoint API timer. Timings are inclusive; concurrent owners are not summed
to estimate elapsed time. No per-object events or trace files are emitted.
"""

import functools
import inspect
import os
import socket
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator

G_ENABLED = os.environ.get("BENCH_PROFILE") == "1"
G_LOCK = threading.Lock()
G_RECORDS: dict[str, dict[str, int]] = {}


def record(
    name: str, elapsed_ns: int, *, calls: int = 1, items: int = 0, nbytes: int = 0
) -> None:
    """Accumulate a coarse region or locally accumulated hot-loop durations."""
    if not G_ENABLED:
        return
    with G_LOCK:
        result = G_RECORDS.setdefault(
            name, {"duration_ns": 0, "calls": 0, "items": 0, "bytes": 0}
        )
        result["duration_ns"] += int(elapsed_ns)
        result["calls"] += int(calls)
        result["items"] += int(items)
        result["bytes"] += int(nbytes)


@contextmanager
def span(name: str) -> Iterator[None]:
    """Measure one coarse synchronous or async-body region."""
    if not G_ENABLED:
        yield
        return
    started = time.perf_counter_ns()
    try:
        yield
    finally:
        record(name, time.perf_counter_ns() - started)


def timed(name: str) -> Callable:
    """Decorate ordinary or coroutine functions without changing their result."""
    def decorate(function: Callable) -> Callable:
        if not G_ENABLED:
            return function
        if inspect.iscoroutinefunction(function):
            @functools.wraps(function)
            async def asynchronous(*args: Any, **kwargs: Any) -> Any:
                with span(name):
                    return await function(*args, **kwargs)
            return asynchronous

        @functools.wraps(function)
        def synchronous(*args: Any, **kwargs: Any) -> Any:
            with span(name):
                return function(*args, **kwargs)
        return synchronous
    return decorate


def snapshot() -> dict[str, Any]:
    """Return a stable copy for collection after the outer benchmark timer."""
    with G_LOCK:
        values = {name: dict(value) for name, value in G_RECORDS.items()}
    return {
        "enabled": G_ENABLED,
        "hostname": socket.gethostname().split(".")[0],
        "pid": os.getpid(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "clock": "perf_counter_ns",
        "semantics": "inclusive aggregate durations; no cross-host clock subtraction",
        "stages": values,
    }
