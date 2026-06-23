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

from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_rl.algorithms.grpo import (
    MasterConfig,
    _default_grpo_save_state,
    _preserve_router_replay_routed_experts,
    async_grpo_train,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


@pytest.fixture(scope="session", autouse=True)
def init_ray_cluster():
    yield


@pytest.fixture(scope="session", autouse=True)
def ray_gpu_monitor():
    class _NoopGpuMonitor:
        def _collect_metrics(self):
            return {}

        def stop(self):
            pass

    yield _NoopGpuMonitor()


@pytest.fixture(scope="session", autouse=True)
def session_data(_unit_test_data):
    yield _unit_test_data


class _StubReplayBuffer:
    def __init__(self, batch: BatchedDataDict, rollout_metrics: dict[str, float]):
        self.batch = batch
        self.rollout_metrics = rollout_metrics

    @property
    def size(self):
        mock = MagicMock()
        mock.remote = MagicMock(return_value=1)
        return mock

    @property
    def has_complete_batch(self):
        mock = MagicMock()
        mock.remote = MagicMock(return_value=True)
        return mock

    @property
    def get_trajectories_needed(self):
        mock = MagicMock()
        mock.remote = MagicMock(return_value=0)
        return mock

    @property
    def sample(self):
        def _sample(num_prompt_groups, *_args, **_kwargs):
            return {
                "trajectories": [
                    {
                        "batch": self.batch,
                        "rollout_metrics": self.rollout_metrics,
                    }
                    for _ in range(num_prompt_groups)
                ],
                "avg_trajectory_age": 0.0,
            }

        mock = MagicMock()
        mock.remote = MagicMock(side_effect=_sample)
        return mock


class _StubAsyncTrajectoryCollector:
    def _remote_none(self):
        mock = MagicMock()
        mock.remote = MagicMock(return_value=None)
        return mock

    @property
    def start_collection(self):
        return self._remote_none()

    @property
    def set_weight_version(self):
        return self._remote_none()

    @property
    def pause(self):
        return self._remote_none()

    @property
    def resume(self):
        return self._remote_none()

    @property
    def prepare_for_refit(self):
        return self._remote_none()

    @property
    def resume_after_refit(self):
        return self._remote_none()

    @property
    def get_dataloader_state(self):
        return self._remote_none()


def _mock_ray_get(value):
    if isinstance(value, (bool, int, float, str, dict, list, tuple)):
        return value
    return None


@contextmanager
def _mock_async_infrastructure(
    batch: BatchedDataDict,
    rollout_metrics: dict[str, float],
):
    stack = ExitStack()
    replay_buffer = _StubReplayBuffer(batch, rollout_metrics)
    collector = _StubAsyncTrajectoryCollector()

    replay_cls = MagicMock()
    replay_cls.options.return_value.remote.return_value = replay_buffer
    collector_cls = MagicMock()
    collector_cls.options.return_value.remote.return_value = collector

    stack.enter_context(
        patch("nemo_rl.algorithms.async_utils.ReplayBuffer", replay_cls)
    )
    stack.enter_context(
        patch("nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector", collector_cls)
    )
    stack.enter_context(
        patch("nemo_rl.algorithms.grpo.get_actor_python_env", return_value="/fake/python")
    )
    stack.enter_context(
        patch(
            "nemo_rl.algorithms.grpo.create_local_venv_on_each_node",
            return_value="/fake/python",
        )
    )
    stack.enter_context(patch("nemo_rl.algorithms.grpo.ray.get", side_effect=_mock_ray_get))
    stack.enter_context(patch("nemo_rl.algorithms.grpo.ray.kill", return_value=None))
    stack.enter_context(
        patch("nemo_rl.algorithms.grpo.refit_policy_generation", return_value=None)
    )
    stack.enter_context(
        patch("nemo_rl.algorithms.grpo.print_performance_metrics", return_value={})
    )
    stack.enter_context(
        patch("nemo_rl.algorithms.grpo.maybe_gpu_profile_step", return_value=None)
    )
    stack.enter_context(
        patch(
            "nemo_rl.algorithms.grpo.compute_and_apply_seq_logprob_error_masking",
            return_value={
                "max_seq_mult_prob_error": 1.0,
                "mean_seq_mult_prob_error": 1.0,
                "min_seq_mult_prob_error": 1.0,
                "max_seq_mult_prob_error_after_mask": 1.0,
                "mean_seq_mult_prob_error_after_mask": 1.0,
                "min_seq_mult_prob_error_after_mask": 1.0,
                "num_masked_seqs": 0,
                "masked_correct_pct": 0.0,
            },
        )
    )
    try:
        yield
    finally:
        stack.close()


def _make_master_config() -> MasterConfig:
    return MasterConfig.model_construct(
        **{
            "grpo": {
                "max_num_steps": 1,
                "max_num_epochs": 1,
                "num_prompts_per_step": 1,
                "num_generations_per_prompt": 1,
                "max_rollout_turns": 1,
                "val_period": 0,
                "val_batch_size": 1,
                "val_at_start": False,
                "val_at_end": False,
                "max_val_samples": 0,
                "seed": 42,
                "use_dynamic_sampling": False,
                "overlong_filtering": False,
                "advantage_clip_low": None,
                "advantage_clip_high": None,
                "async_grpo": {
                    "enabled": True,
                    "max_trajectory_age_steps": 1,
                },
                "seq_logprob_error_threshold": None,
                "skip_reference_policy_logprobs_calculation": False,
            },
            "policy": {
                "train_global_batch_size": 1,
                "train_micro_batch_size": 1,
                "max_total_sequence_length": 16,
                "make_sequence_length_divisible_by": 1,
                "router_replay": {"enabled": True},
                "generation": {
                    "backend": "vllm",
                    "colocated": {"enabled": False},
                    "vllm_cfg": {
                        "async_engine": True,
                        "enable_vllm_metrics_logger": False,
                    },
                },
            },
            "loss_fn": SimpleNamespace(
                use_importance_sampling_correction=True,
                force_on_policy_ratio=False,
            ),
            "checkpointing": {
                "enabled": False,
                "checkpoint_must_save_by": None,
                "save_period": 10,
            },
            "cluster": {"num_nodes": 1, "gpus_per_node": 1},
            "logger": {"wandb_enabled": False},
            "data_plane": None,
        }
    )


def _make_policy() -> MagicMock:
    policy = MagicMock()
    policy.sharding_annotations.get_axis_size.return_value = 1
    policy.get_logprobs.return_value = {"logprobs": torch.zeros(1, 3)}
    policy.get_reference_policy_logprobs.return_value = {
        "reference_logprobs": torch.zeros(1, 3)
    }
    policy.train.return_value = {
        "loss": torch.tensor(0.5),
        "grad_norm": torch.tensor(1.0),
        "all_mb_metrics": {
            "global_valid_toks": [2],
            "gen_kl_error": [0.0],
            "token_mult_prob_error": [1.0],
        },
    }
    return policy


def _make_policy_generation() -> MagicMock:
    policy_generation = MagicMock()
    policy_generation.get_logger_metrics.return_value = {}
    return policy_generation


def _make_replay_batch() -> BatchedDataDict:
    return BatchedDataDict(
        {
            "message_log": [
                [
                    {
                        "role": "user",
                        "content": "prompt",
                        "token_ids": torch.tensor([1]),
                    },
                    {
                        "role": "assistant",
                        "content": "answer",
                        "token_ids": torch.tensor([2, 3]),
                        "generation_logprobs": torch.zeros(2),
                    },
                ]
            ],
            "task_name": ["math"],
            "extra_env_info": [{}],
            "loss_multiplier": torch.tensor([1.0]),
            "length": torch.tensor([1]),
            "total_reward": torch.tensor([1.0]),
        }
    )


def test_preserve_router_replay_routed_experts_is_gated():
    routes = torch.arange(1 * 3 * 2 * 4, dtype=torch.int32).reshape(1, 3, 2, 4)
    flat_messages = BatchedDataDict({"routed_experts": routes})

    enabled_target = BatchedDataDict({"input_ids": torch.ones(1, 3, dtype=torch.long)})
    _preserve_router_replay_routed_experts(
        enabled_target,
        flat_messages,
        {"router_replay": {"enabled": True}},
    )
    assert torch.equal(enabled_target["routed_experts"], routes)

    disabled_target = BatchedDataDict({"input_ids": torch.ones(1, 3, dtype=torch.long)})
    _preserve_router_replay_routed_experts(
        disabled_target,
        flat_messages,
        {"router_replay": {"enabled": False}},
    )
    assert "routed_experts" not in disabled_target


def test_async_grpo_train_preserves_routed_experts_for_r3(monkeypatch):
    routes = torch.arange(1 * 3 * 2 * 4, dtype=torch.int32).reshape(1, 3, 2, 4)
    fake_flat = BatchedDataDict(
        {
            "token_ids": torch.tensor([[1, 2, 3]]),
            "generation_logprobs": torch.zeros(1, 3),
            "token_loss_mask": torch.tensor([[0, 1, 1]]),
            "content": ["ok"],
            "routed_experts": routes,
        }
    )
    policy = _make_policy()
    policy_generation = _make_policy_generation()
    master_config = _make_master_config()
    replay_batch = _make_replay_batch()
    mock_adv_estimator = MagicMock()
    mock_adv_estimator.compute_advantage.return_value = torch.ones(1, 3)

    monkeypatch.setattr(
        "nemo_rl.algorithms.grpo.batched_message_log_to_flat_message",
        lambda *_args, **_kwargs: (fake_flat, torch.tensor([3])),
    )
    monkeypatch.setattr(
        "nemo_rl.algorithms.grpo._create_advantage_estimator",
        lambda _cfg: mock_adv_estimator,
    )

    checkpointer = MagicMock()
    checkpointer.get_latest_checkpoint_path.return_value = None

    with _mock_async_infrastructure(
        replay_batch,
        {"mean_gen_tokens_per_sample": 2.0, "gen_kl_error": 0.0},
    ):
        async_grpo_train(
            policy,
            policy_generation,
            MagicMock(),
            None,
            MagicMock(pad_token_id=0),
            MagicMock(),
            {"math": MagicMock()},
            None,
            MagicMock(),
            checkpointer,
            _default_grpo_save_state(),
            master_config,
        )

    assert torch.equal(policy.get_logprobs.call_args[0][0]["routed_experts"], routes)
    assert torch.equal(policy.train.call_args[0][0]["routed_experts"], routes)


def test_async_grpo_r3_rejects_data_plane_until_async_tq_exists():
    master_config = _make_master_config()
    master_config.data_plane = {"enabled": True}

    with pytest.raises(NotImplementedError, match="data_plane.enabled=false"):
        async_grpo_train(
            _make_policy(),
            _make_policy_generation(),
            MagicMock(),
            None,
            MagicMock(),
            MagicMock(),
            {"math": MagicMock()},
            None,
            MagicMock(),
            MagicMock(),
            _default_grpo_save_state(),
            master_config,
        )
