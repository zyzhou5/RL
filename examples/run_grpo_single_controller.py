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

"""Async GRPO / PPO launcher driven by the SingleController actor.

Builds the full SC actor args driver-side via setup_single_controller and hands them
to SingleControllerActor. Mirrors run_grpo.py for config loading so the same YAML
files apply. data_plane.enabled=true is mandatory. A config carrying a `ppo:` block
additionally brings up the PPO critic and trains it alongside the policy.
"""

import argparse
import os
import pprint
import sys
import time
from typing import Any

import ray
from omegaconf import OmegaConf

from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.algorithms.single_controller_utils import (
    MasterConfig,
    WatchdogConfig,
    is_ppo_run,
    setup_single_controller,
)
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data_plane.factory import maybe_configure_data_plane_env
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.nemo_gym import setup_nemo_gym_config
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir

# Teardown must be bounded: it runs in a finally block, so a hung shutdown would
# replace a real training error with an indefinite hang.
_SHUTDOWN_TIMEOUT_S = 10


class _ControllerTerminationError(RuntimeError):
    """The driver could not prove that Single Controller stopped."""


# Fail-closed references used only when teardown cannot prove dependents are
# terminal. Keeping the owner reachable prevents an eager __del__ retry while
# the driver reports the failure.
_PRESERVED_TQ_OWNERS: list[Any] = []


# Drop examples/ from sys.path so examples/nemo_gym/ (no __init__.py) doesn't
# shadow the real nemo_gym package as a namespace package.
current_dir = os.path.dirname(os.path.abspath(__file__))
while current_dir in sys.path:
    sys.path.remove(current_dir)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run async GRPO / PPO training via SingleController"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def main() -> None:
    """Main entry point."""
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__),
            "configs",
            "grpo_math_1B_megatron_single_controller.yaml",
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config = OmegaConf.to_container(config, resolve=True)
    config = MasterConfig(**config)
    print("Applied CLI overrides")

    if is_ppo_run(config):
        legacy_async_block, legacy_async = "ppo.async_ppo", config.ppo.async_ppo
    else:
        legacy_async_block, legacy_async = "grpo.async_grpo", config.grpo.async_grpo
    if legacy_async is not None:
        raise ValueError(
            f"SC requires `{legacy_async_block}: null`; use `async_rl.*` instead. "
            "See docs/guides/single-controller.md#migrating-a-legacy-async-config."
        )

    dp_cfg = config.data_plane
    if not dp_cfg.get("enabled", False):
        raise ValueError(
            "run_grpo_single_controller requires data_plane.enabled=true. "
            "Use examples/run_grpo.py for the legacy / sync paths."
        )

    print("Final config:")
    pprint.pprint(config)

    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    print(f"📊 Using log directory: {config.logger['log_dir']}")
    if config.checkpointing["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config.checkpointing['checkpoint_dir']}"
        )

    # Must precede init_ray() — see maybe_configure_data_plane_env's docstring.
    maybe_configure_data_plane_env(config.data_plane)
    init_ray()

    tokenizer = get_tokenizer(config.policy["tokenizer"])
    assert config.policy["generation"] is not None, (
        "A generation config is required for SC-driven async GRPO"
    )
    has_refit_draft_weights = bool(config.policy["draft"]["enabled"])
    megatron_cfg = config.policy.get("megatron_cfg") or {}
    trains_mtp = bool(megatron_cfg.get("mtp_num_layers"))
    config.policy["generation"] = configure_generation_config(
        config.policy["generation"],
        tokenizer,
        has_refit_draft_weights=has_refit_draft_weights,
        trains_mtp=trains_mtp,
    )

    # NeMo-Gym specific config setup.
    if bool(config.env.get("should_use_nemo_gym")):
        setup_nemo_gym_config(config, tokenizer)

    actor_args, setup_timing_metrics = setup_single_controller(config, tokenizer)

    print("🚀 Launching SingleControllerActor")
    sc = SingleControllerActor.remote(
        master_config=config,
        actor_args=actor_args,
        setup_timing_metrics=setup_timing_metrics,
    )
    controller_stopped = False
    try:
        try:
            result = _run_with_controller_liveness_watch(
                sc, config.async_rl.stall_watchdog
            )
        except _ControllerTerminationError:
            raise
        except BaseException:
            # _run_with_controller_liveness_watch only re-raises the run error
            # after its cancellation/kill path has proved the task terminal.
            controller_stopped = True
            raise
        else:
            controller_stopped = True
            print(f"SC run complete: {result}")
    finally:
        if controller_stopped:
            _shutdown_driver_resources(actor_args)
        else:
            _preserve_trainer_tq_owner(
                actor_args,
                "Single Controller could not be proven terminal",
            )
            print(
                "Skipping driver resource teardown because Single Controller "
                "may still be alive; preserving its workers and TQ owner.",
                flush=True,
            )


def _shutdown_driver_resources(actor_args: Any) -> None:
    """Stop dependents in ownership order after Single Controller is terminal."""
    # Drain env actors before generation to avoid in-flight requests during shutdown.
    for env_name, handle in actor_args.env_handles.items():
        try:
            ray.get(handle.shutdown.remote(), timeout=_SHUTDOWN_TIMEOUT_S)
        except Exception as e:
            print(f"Env {env_name!r} shutdown failed: {e}")
            try:
                ray.kill(handle)
            except Exception as kill_error:
                print(f"Env {env_name!r} kill failed: {kill_error}")

    for resource_name, resource in (
        ("Generation", actor_args.gen_handle),
        ("Value", actor_args.value_handle),
        # The trainer owns the bootstrap TQ client/controller. Stop every
        # attached model first so their process-local clients can detach.
        ("Trainer", actor_args.trainer_handle),
    ):
        if resource is None:
            continue
        try:
            stopped = resource.shutdown()
        except Exception as e:
            print(f"{resource_name} shutdown failed: {e}")
            if resource_name == "Value":
                _preserve_trainer_tq_owner(
                    actor_args,
                    "value workers may still be attached",
                )
                print(
                    "Trainer shutdown skipped because value workers may still "
                    "be attached to its TQ owner."
                )
                break
        else:
            if stopped is False and resource_name == "Value":
                _preserve_trainer_tq_owner(
                    actor_args,
                    "value workers may still be attached",
                )
                print(
                    "Trainer shutdown skipped because value workers may still "
                    "be attached to its TQ owner."
                )
                break


def _preserve_trainer_tq_owner(actor_args: Any, reason: str) -> None:
    """Block explicit and destructor-driven owner teardown after uncertainty."""
    trainer = actor_args.trainer_handle
    if trainer is None:
        return
    if not any(owner is trainer for owner in _PRESERVED_TQ_OWNERS):
        _PRESERVED_TQ_OWNERS.append(trainer)

    preserve = getattr(trainer, "preserve_data_plane_on_shutdown", None)
    if preserve is not None:
        preserve(reason)


def _run_with_controller_liveness_watch(
    sc: ray.actor.ActorHandle, watchdog_config: WatchdogConfig
) -> dict[str, Any]:
    """Await the SC run, polling ping() so a frozen event loop cannot hide.

    The in-actor watchdog is an asyncio task on the SC's own event loop, so it cannot
    observe that loop being blocked -- by a synchronous Ray call into a wedged worker,
    say. The driver is a separate process that already holds the handle, which makes it
    the cheapest possible external observer; no supervisor actor required.

    ping() returning is the liveness signal. A slow reply is not a freeze, so the check
    only escalates once the loop has been unresponsive for the same budget the in-actor
    watchdog uses to call a stall.
    """
    run_ref = sc.run.remote()
    last_pong_at = time.monotonic()

    try:
        while True:
            ready, _ = ray.wait([run_ref], timeout=watchdog_config.interval_s)
            if ready:
                return ray.get(run_ref)

            try:
                ray.get(sc.ping.remote(), timeout=watchdog_config.interval_s)
            except Exception as error:
                unresponsive_s = time.monotonic() - last_pong_at
                print(
                    f"SingleController ping failed after {unresponsive_s:.0f}s "
                    f"unresponsive: {type(error).__name__}: {error}",
                    flush=True,
                )
                if unresponsive_s > watchdog_config.stall_timeout_s:
                    raise RuntimeError(
                        "SingleController event loop has been unresponsive for "
                        f"{unresponsive_s:.0f}s (stall_timeout_s="
                        f"{watchdog_config.stall_timeout_s}); its in-actor watchdog "
                        "runs on that loop and cannot report this."
                    ) from error
            else:
                last_pong_at = time.monotonic()
    except BaseException:
        _cancel_and_drain_controller_run(sc, run_ref)
        raise


def _cancel_and_drain_controller_run(
    sc: ray.actor.ActorHandle, run_ref: ray.ObjectRef
) -> None:
    """Stop ``sc.run`` before the driver tears down attached TQ resources."""
    try:
        # SingleControllerActor is async, so Ray maps this to asyncio task
        # cancellation and lets run() execute its cancellation-safe finally.
        ray.cancel(run_ref)
    except Exception as error:
        print(f"SingleController run cancellation request failed: {error}", flush=True)

    try:
        ray.get(run_ref, timeout=_SHUTDOWN_TIMEOUT_S)
    except ray.exceptions.GetTimeoutError:
        # A blocked actor event loop cannot accept graceful cancellation. Kill
        # it before any driver-side worker or bootstrap TQ owner is shut down.
        try:
            ray.kill(sc)
        except Exception as error:
            raise _ControllerTerminationError(
                "SingleController force-kill failed; dependent resources must "
                "remain alive"
            ) from error

        try:
            # ray.kill submits the forceful termination. Prove the original run
            # reference became terminal before releasing dependent resources.
            ray.get(run_ref, timeout=_SHUTDOWN_TIMEOUT_S)
        except ray.exceptions.GetTimeoutError as error:
            raise _ControllerTerminationError(
                "SingleController run remained nonterminal after force-kill; "
                "dependent resources must remain alive"
            ) from error
        except Exception:
            # RayActorError (or the run's terminal error) proves the task ended.
            pass
    except Exception:
        # TaskCancelledError, RayActorError, or the run's original exception all
        # mean the actor task is terminal and its finally has finished.
        pass


if __name__ == "__main__":
    main()
