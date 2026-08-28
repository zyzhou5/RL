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

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from examples import run_grpo_single_controller
from nemo_rl.algorithms.grpo import GRPOConfig
from nemo_rl.algorithms.metric_utils import SetupTimingMetrics
from nemo_rl.algorithms.single_controller_utils.config import MasterConfig


@pytest.fixture
def main_context(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    generation_config = {"backend": "vllm"}
    config = MasterConfig.model_construct(
        policy={
            "tokenizer": {},
            "generation": generation_config,
            "draft": {"enabled": False},
            "megatron_cfg": {"mtp_num_layers": 2},
        },
        env={},
        data_plane={"enabled": True, "impl": "transfer_queue", "backend": "simple"},
        logger={"log_dir": "/tmp/logs"},
        checkpointing={"enabled": False},
        async_rl=SimpleNamespace(
            stall_watchdog=SimpleNamespace(interval_s=30.0, stall_timeout_s=600.0)
        ),
        grpo=GRPOConfig(async_grpo=None),
    )
    configured_generation = {"backend": "vllm", "_mtp_weights_from_refit": True}
    configure_generation = MagicMock(return_value=configured_generation)
    actor = SimpleNamespace(run=SimpleNamespace(remote=MagicMock(return_value="run")))
    actor_args = SimpleNamespace(
        env_handles={},
        gen_handle=SimpleNamespace(shutdown=MagicMock()),
        trainer_handle=SimpleNamespace(shutdown=MagicMock()),
        value_handle=None,
    )
    ray_get = MagicMock(return_value={})
    # The driver now polls ping() around the run. Report the run as ready on the first
    # check so these tests keep exercising the same path they always did.
    ray_wait = MagicMock(side_effect=lambda refs, timeout=None: (list(refs), []))
    ray_cancel = MagicMock()
    ray_kill = MagicMock()
    monkeypatch.setattr(run_grpo_single_controller, "_PRESERVED_TQ_OWNERS", [])

    monkeypatch.setattr(
        run_grpo_single_controller,
        "parse_args",
        lambda: (Namespace(config="config.yaml"), []),
    )
    monkeypatch.setattr(run_grpo_single_controller, "load_config", lambda _: {})
    monkeypatch.setattr(
        run_grpo_single_controller.OmegaConf,
        "to_container",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(run_grpo_single_controller, "MasterConfig", lambda **_: config)
    monkeypatch.setattr(run_grpo_single_controller, "init_ray", lambda: None)
    monkeypatch.setattr(
        run_grpo_single_controller,
        "get_tokenizer",
        lambda _: "tokenizer",
    )
    monkeypatch.setattr(
        run_grpo_single_controller,
        "get_next_experiment_dir",
        lambda _: "/tmp/logs/0",
    )
    monkeypatch.setattr(
        run_grpo_single_controller,
        "configure_generation_config",
        configure_generation,
    )
    monkeypatch.setattr(
        run_grpo_single_controller,
        "setup_single_controller",
        lambda *_args: (actor_args, SetupTimingMetrics()),
    )
    monkeypatch.setattr(
        run_grpo_single_controller.SingleControllerActor,
        "remote",
        MagicMock(return_value=actor),
    )
    monkeypatch.setattr(run_grpo_single_controller.ray, "get", ray_get)
    monkeypatch.setattr(run_grpo_single_controller.ray, "wait", ray_wait)
    monkeypatch.setattr(run_grpo_single_controller.ray, "cancel", ray_cancel)
    monkeypatch.setattr(run_grpo_single_controller.ray, "kill", ray_kill)

    return SimpleNamespace(
        actor=actor,
        actor_args=actor_args,
        config=config,
        configure_generation=configure_generation,
        configured_generation=configured_generation,
        generation_config=generation_config,
        ray_get=ray_get,
        ray_wait=ray_wait,
        ray_cancel=ray_cancel,
        ray_kill=ray_kill,
    )


def test_cleanup_is_best_effort_and_preserves_run_error(
    main_context: SimpleNamespace,
    capsys: pytest.CaptureFixture[str],
) -> None:
    failing_env = SimpleNamespace(
        shutdown=SimpleNamespace(remote=MagicMock(return_value="failing-env"))
    )
    healthy_env = SimpleNamespace(
        shutdown=SimpleNamespace(remote=MagicMock(return_value="healthy-env"))
    )
    generation = SimpleNamespace(
        shutdown=MagicMock(side_effect=RuntimeError("generation cleanup failed"))
    )
    trainer = SimpleNamespace(shutdown=MagicMock())
    main_context.actor_args.env_handles = {
        "failing": failing_env,
        "healthy": healthy_env,
    }
    main_context.actor_args.gen_handle = generation
    main_context.actor_args.trainer_handle = trainer

    def get(ref: object, timeout: float | None = None) -> None:
        del timeout
        if ref == "run":
            raise RuntimeError("training failed")
        if ref == "failing-env":
            raise RuntimeError("env cleanup failed")
        return None

    main_context.ray_get.side_effect = get

    with pytest.raises(RuntimeError, match="training failed"):
        run_grpo_single_controller.main()

    healthy_env.shutdown.remote.assert_called_once_with()
    # A hung env must not replace the training error with an indefinite wait.
    main_context.ray_kill.assert_called_once_with(failing_env)
    generation.shutdown.assert_called_once_with()
    trainer.shutdown.assert_called_once_with()
    output = capsys.readouterr().out
    assert "Env 'failing' shutdown failed: env cleanup failed" in output
    assert "Generation shutdown failed: generation cleanup failed" in output


def test_main_configures_generation_for_trained_mtp(
    main_context: SimpleNamespace,
) -> None:
    run_grpo_single_controller.main()

    main_context.configure_generation.assert_called_once_with(
        main_context.generation_config,
        "tokenizer",
        has_refit_draft_weights=False,
        trains_mtp=True,
    )
    assert (
        main_context.config.policy["generation"] is main_context.configured_generation
    )


def test_main_shuts_down_value_before_tq_owner_trainer(
    main_context: SimpleNamespace,
) -> None:
    events: list[str] = []
    trainer_live = True

    def shutdown_generation() -> None:
        events.append("Generation")

    def shutdown_value() -> None:
        assert trainer_live
        events.append("Value")

    def shutdown_trainer() -> None:
        nonlocal trainer_live
        events.append("Trainer")
        trainer_live = False

    main_context.actor_args.gen_handle = SimpleNamespace(shutdown=shutdown_generation)
    main_context.actor_args.value_handle = SimpleNamespace(shutdown=shutdown_value)
    main_context.actor_args.trainer_handle = SimpleNamespace(shutdown=shutdown_trainer)

    run_grpo_single_controller.main()

    assert events == ["Generation", "Value", "Trainer"]
    assert not trainer_live


def test_value_shutdown_failure_preserves_tq_owner_trainer(
    main_context: SimpleNamespace,
    capsys: pytest.CaptureFixture[str],
) -> None:
    events: list[str] = []
    main_context.actor_args.gen_handle = SimpleNamespace(
        shutdown=lambda: events.append("Generation")
    )

    def fail_value_shutdown() -> bool:
        events.append("Value")
        return False

    main_context.actor_args.value_handle = SimpleNamespace(shutdown=fail_value_shutdown)

    def preserve_owner(reason: str) -> None:
        events.append(f"preserve:{reason}")

    main_context.actor_args.trainer_handle = SimpleNamespace(
        shutdown=lambda: events.append("Trainer"),
        preserve_data_plane_on_shutdown=preserve_owner,
    )

    run_grpo_single_controller.main()

    assert events == [
        "Generation",
        "Value",
        "preserve:value workers may still be attached",
    ]
    assert run_grpo_single_controller._PRESERVED_TQ_OWNERS == [
        main_context.actor_args.trainer_handle
    ]
    assert "Trainer shutdown skipped" in capsys.readouterr().out


def test_liveness_failure_stops_controller_before_tq_resources(
    main_context: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    main_context.config.async_rl.stall_watchdog = SimpleNamespace(
        interval_s=1.0,
        stall_timeout_s=0.0,
    )
    main_context.actor.ping = SimpleNamespace(remote=MagicMock(return_value="ping"))
    main_context.ray_wait.side_effect = lambda refs, timeout=None: ([], list(refs))
    controller_killed = False

    def ray_get(ref: object, timeout: float | None = None) -> None:
        if ref == "ping":
            raise RuntimeError("injected ping timeout")
        if ref == "run" and timeout == run_grpo_single_controller._SHUTDOWN_TIMEOUT_S:
            if controller_killed:
                return None
            raise run_grpo_single_controller.ray.exceptions.GetTimeoutError(
                "injected cancellation timeout"
            )
        raise AssertionError(f"unexpected ray.get({ref!r}, timeout={timeout!r})")

    def ray_kill(actor: object) -> None:
        nonlocal controller_killed
        events.append(
            "kill:controller" if actor is main_context.actor else "kill:other"
        )
        controller_killed = True

    main_context.ray_get.side_effect = ray_get
    main_context.ray_cancel.side_effect = lambda ref: events.append(f"cancel:{ref}")
    main_context.ray_kill.side_effect = ray_kill
    main_context.actor_args.gen_handle = SimpleNamespace(
        shutdown=lambda: events.append("Generation")
    )
    main_context.actor_args.value_handle = SimpleNamespace(
        shutdown=lambda: events.append("Value")
    )

    def preserve_owner(reason: str) -> None:
        events.append(f"preserve:{reason}")

    main_context.actor_args.trainer_handle = SimpleNamespace(
        shutdown=lambda: events.append("Trainer"),
        preserve_data_plane_on_shutdown=preserve_owner,
    )
    monotonic_values = iter([0.0, 1.0])
    monkeypatch.setattr(
        run_grpo_single_controller.time,
        "monotonic",
        lambda: next(monotonic_values),
    )

    with pytest.raises(RuntimeError, match="event loop has been unresponsive"):
        run_grpo_single_controller.main()

    assert events == [
        "cancel:run",
        "kill:controller",
        "Generation",
        "Value",
        "Trainer",
    ]


def test_force_kill_failure_preserves_all_controller_dependencies(
    main_context: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    events: list[str] = []
    main_context.config.async_rl.stall_watchdog = SimpleNamespace(
        interval_s=1.0,
        stall_timeout_s=0.0,
    )
    main_context.actor.ping = SimpleNamespace(remote=MagicMock(return_value="ping"))
    main_context.ray_wait.side_effect = lambda refs, timeout=None: ([], list(refs))

    def ray_get(ref: object, timeout: float | None = None) -> None:
        if ref == "ping":
            raise RuntimeError("injected ping timeout")
        if ref == "run" and timeout == run_grpo_single_controller._SHUTDOWN_TIMEOUT_S:
            raise run_grpo_single_controller.ray.exceptions.GetTimeoutError(
                "injected cancellation timeout"
            )
        raise AssertionError(f"unexpected ray.get({ref!r}, timeout={timeout!r})")

    def ray_kill(actor: object) -> None:
        assert actor is main_context.actor
        events.append("kill:controller")
        raise RuntimeError("injected force-kill failure")

    main_context.ray_get.side_effect = ray_get
    main_context.ray_cancel.side_effect = lambda ref: events.append(f"cancel:{ref}")
    main_context.ray_kill.side_effect = ray_kill
    main_context.actor_args.gen_handle = SimpleNamespace(
        shutdown=lambda: events.append("Generation")
    )
    main_context.actor_args.value_handle = SimpleNamespace(
        shutdown=lambda: events.append("Value")
    )

    def preserve_owner(reason: str) -> None:
        events.append(f"preserve:{reason}")

    main_context.actor_args.trainer_handle = SimpleNamespace(
        shutdown=lambda: events.append("Trainer"),
        preserve_data_plane_on_shutdown=preserve_owner,
    )
    monotonic_values = iter([0.0, 1.0])
    monkeypatch.setattr(
        run_grpo_single_controller.time,
        "monotonic",
        lambda: next(monotonic_values),
    )

    with pytest.raises(
        run_grpo_single_controller._ControllerTerminationError,
        match="force-kill failed",
    ):
        run_grpo_single_controller.main()

    assert events == [
        "cancel:run",
        "kill:controller",
        "preserve:Single Controller could not be proven terminal",
    ]
    assert run_grpo_single_controller._PRESERVED_TQ_OWNERS == [
        main_context.actor_args.trainer_handle
    ]
    assert "Skipping driver resource teardown" in capsys.readouterr().out
