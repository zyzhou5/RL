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

"""Route typed Gym tasks to session harnesses without changing prompt groups."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from nemo_rl.environments.nemo_gym_task import get_nemo_gym_task_input
from nemo_rl.experience.failures import RolloutDataFailure
from nemo_rl.experience.interfaces import (
    NEMO_GYM_GROUP_ID_KEY,
    NEMO_GYM_ROLLOUT_INDEX_KEY,
)


HarnessPool = tuple["HarnessRoute", ...]


@dataclass(frozen=True)
class HarnessRoute:
    """One resolved Gym environment, the agent it binds, and the resources server it shares.

    A Harbor environment grades with its tasks' own verifiers, so it names no resources server.
    """

    environment_name: str
    agent_name: str
    resources_name: str | None


def _parse_names(value: object) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(name, str) or not name.strip() for name in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(
            "env.nemo_gym.harness_pool must be a nonempty list of distinct environment names"
        )
    return list(value)


def parse_harness_pool(value: object) -> list[str] | dict[str, list[str]] | None:
    """Validate the optional pool: one ordered list for every taskset, or one list per taskset."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        if not value or any(
            not isinstance(taskset, str) or not taskset.strip() for taskset in value
        ):
            raise ValueError(
                "env.nemo_gym.harness_pool must map nonempty taskset names to environment lists"
            )
        return {taskset: _parse_names(names) for taskset, names in value.items()}
    return _parse_names(value)


def resolve_harness_pool(
    names: list[str],
    global_config: Mapping[str, Any],
    *,
    taskset: str | None = None,
) -> HarnessPool:
    """Resolve a pool of environments that differ only in their harness.

    Single-agent environments must share one resources server and verifier. Harbor environments
    must serve the same tasksets, or, for a pool of one taskset, each serve that taskset.

    Args:
        names: Ordered, distinct environment instance names.
        global_config: Gym's resolved global configuration.
        taskset: The taskset this pool serves, or None for a pool that serves every taskset.

    Returns:
        Validated routes in the configured order.
    """
    _parse_names(names)
    routes = []
    kinds = set()
    harbor_tasksets = set()
    for name in names:
        entry = global_config.get(name, {})
        servers = entry.get("environment_servers", {})
        if set(servers) not in ({"single_agent_turn"}, {"harbor"}):
            raise ValueError(
                f"Harness environment {name!r} must use typed single_agent_turn or harbor"
            )
        kind = next(iter(servers))
        kinds.add(kind)
        config = servers[kind]
        if config.get("agent_server_pool") is not None:
            raise ValueError(
                f"Harness environment {name!r} must bind exactly one agent"
            )
        agent = config.get("agent_server") or {}
        if agent.get("type") != "responses_api_agents":
            raise ValueError(
                f"Harness environment {name!r} has invalid agent/resources references"
            )
        resources_name = None
        if kind == "single_agent_turn":
            resources = config.get("resources_server") or {}
            if resources.get("type") != "resources_servers":
                raise ValueError(
                    f"Harness environment {name!r} has invalid agent/resources references"
                )
            resources_name = resources.get("name")
            if not isinstance(resources_name, str):
                raise ValueError(
                    f"Harness environment {name!r} must name its agent and resources server"
                )
            if not global_config.get(resources_name, {}).get("resources_servers"):
                raise ValueError(
                    f"Harness environment {name!r} references missing resources {resources_name!r}"
                )
        else:
            served = config.get("tasksets") or {}
            if not isinstance(served, Mapping) or not served:
                raise ValueError(
                    f"Harness environment {name!r} must serve at least one taskset"
                )
            if taskset is not None and taskset not in served:
                raise ValueError(
                    f"Harness environment {name!r} does not serve taskset {taskset!r}"
                )
            harbor_tasksets.add(frozenset(served))
        agent_name = agent.get("name")
        if not isinstance(agent_name, str):
            raise ValueError(
                f"Harness environment {name!r} must name its agent and resources server"
            )
        agent_servers = global_config.get(agent_name, {}).get("responses_api_agents")
        if not agent_servers:
            raise ValueError(
                f"Harness environment {name!r} references missing agent {agent_name!r}"
            )
        if len(agent_servers) != 1 or next(iter(agent_servers.values())).get(
            "model_server"
        ) != {
            "type": "responses_api_models",
            "name": "policy_model",
        }:
            raise ValueError(
                f"Harness agent {agent_name!r} must have exactly one configuration "
                "using model_server responses_api_models:policy_model for training capture"
            )
        routes.append(HarnessRoute(name, agent_name, resources_name))
    if len(kinds) != 1:
        raise ValueError(
            "Harness environments must all use single_agent_turn or all use harbor"
        )
    if kinds == {"single_agent_turn"}:
        if len({route.resources_name for route in routes}) != 1:
            raise ValueError(
                "Harness environments must share one resources server and verifier"
            )
    elif taskset is None and len(harbor_tasksets) != 1:
        raise ValueError("Harbor harness environments must serve the same tasksets")
    if len({route.agent_name for route in routes}) != len(routes):
        raise ValueError("Harness environments must bind distinct agent instances")
    return tuple(routes)


def resolve_harness_pools(
    pool: list[str] | dict[str, list[str]], global_config: Mapping[str, Any]
) -> dict[str | None, HarnessPool]:
    """Resolve a list pool, or one pool per taskset, keyed by taskset (None serves every taskset)."""
    parsed = parse_harness_pool(pool)
    if parsed is None:
        raise ValueError("Harness routing requires a nonempty resolved pool")
    if isinstance(parsed, list):
        return {None: resolve_harness_pool(parsed, global_config)}
    return {
        taskset: resolve_harness_pool(names, global_config, taskset=taskset)
        for taskset, names in parsed.items()
    }


def responses_create_params(row: dict[str, Any]) -> dict[str, Any]:
    """Return mutable model parameters from a flat row or typed Gym task."""
    params = get_nemo_gym_task_input(row).get("responses_create_params")
    if not isinstance(params, dict):
        raise ValueError("Each Gym task must contain a responses_create_params object")
    return params


def route_harness_rows(
    rows: list[dict[str, Any]],
    routes: HarnessPool | Mapping[str | None, HarnessPool],
) -> None:
    """Stamp per-rollout destinations using durable identity, never physical attempts.

    The group ID distinguishes repeated visits to a task. Its generation index
    survives sibling retries, while the capture ID can change on each attempt.
    Pool order must remain fixed when resuming a training run.
    """
    pools = routes if isinstance(routes, Mapping) else {None: routes}
    if not pools or not all(pools.values()):
        raise ValueError("Harness routing requires a nonempty resolved pool")
    for row in rows:
        task = row.get("task_id")
        if not isinstance(task, dict) or not isinstance(row.get("task_input"), dict):
            raise ValueError("harness_pool requires typed task_id and task_input rows")
        taskset, task_id = task.get("taskset"), task.get("task_id")
        group = row.get(NEMO_GYM_GROUP_ID_KEY)
        index = row.get(NEMO_GYM_ROLLOUT_INDEX_KEY)
        if any(
            not isinstance(value, str) or not value
            for value in (taskset, task_id, group)
        ):
            raise ValueError(
                "Harness routing requires nonempty taskset, task_id, and durable group ID"
            )
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError(
                "Harness routing requires a nonnegative logical rollout index"
            )
        capture_id = row.get("_ng_rollout_id")
        if not isinstance(capture_id, str) or not capture_id:
            raise ValueError(
                "Harness routing requires the controller's token-capture rollout ID"
            )
        responses_create_params(row)
        key = ["nemo-rl-harness-v1", taskset, task_id, group, index]
        digest = hashlib.sha256(
            json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode()
        ).digest()
        pool = pools.get(taskset, pools.get(None))
        if pool is None:
            raise ValueError(f"No harness pool serves taskset {taskset!r}")
        route = pool[int.from_bytes(digest, "big") % len(pool)]
        if route.resources_name is not None:
            source = row.get("task_source")
            if source is not None and source != route.resources_name:
                raise ValueError(
                    f"Task source {source!r} does not match harness resources {route.resources_name!r}"
                )
            row["task_source"] = route.resources_name
        row["_ng_environment_server"] = route.environment_name
        row["agent_ref"] = {"type": "responses_api_agents", "name": route.agent_name}


def attribute_harness_result(
    row: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    """Attach the selected harness to an already validated episode result."""
    agent_name = row["agent_ref"]["name"]
    if result.get("ng_agent_server") not in (None, agent_name):
        raise RolloutDataFailure("Harness result disagrees with the selected agent")
    return {
        **result,
        "ng_agent_server": agent_name,
        "ng_environment_server": row["_ng_environment_server"],
    }
