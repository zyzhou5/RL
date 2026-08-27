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

from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("nemo_automodel")

from nemo_rl.algorithms.logits_sampling_utils import TrainingSamplingParams
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.automodel.data import ProcessedInputs, ProcessedMicrobatch
from nemo_rl.models.automodel.diffusion_train import (
    DiffusionLogprobsPostProcessor,
    DiffusionLossPostProcessor,
    same_position_logprobs,
)
from nemo_rl.models.automodel.train import forward_with_post_processing_fn


def _processor_kwargs(sampling_params: TrainingSamplingParams) -> dict:
    return {
        "cfg": {},
        "device_mesh": MagicMock(),
        "cp_mesh": MagicMock(),
        "tp_mesh": MagicMock(),
        "cp_size": 1,
        "enable_seq_packing": False,
        "sampling_params": sampling_params,
    }


def test_same_position_logprobs_scores_target_at_same_position():
    logits = torch.tensor([[[8.0, 1.0, 0.0], [0.0, 7.0, 1.0], [0.0, 1.0, 6.0]]])
    target_ids = torch.tensor([[0, 1, 2]])

    actual = same_position_logprobs(logits, target_ids)
    expected = (
        torch.log_softmax(logits.float(), dim=-1)
        .gather(-1, target_ids.unsqueeze(-1))
        .squeeze(-1)
    )

    assert torch.allclose(actual, expected)
    assert torch.all(actual > -0.01)


def test_logprob_processor_applies_temperature_exactly_once():
    raw_logits = torch.tensor([[[4.0, 1.0, -2.0], [1.0, 5.0, 0.0]]])
    target_ids = torch.tensor([[0, 1]])
    sampling_params = TrainingSamplingParams(temperature=2.0)
    post_processor = DiffusionLogprobsPostProcessor(
        **_processor_kwargs(sampling_params)
    )
    processed_inputs = ProcessedInputs(
        input_ids=torch.tensor([[2, 2]]),
        seq_len=2,
        attention_mask=torch.ones(1, 2),
        position_ids=torch.arange(2).unsqueeze(0),
    )
    processed_microbatch = ProcessedMicrobatch(
        data_dict=BatchedDataDict(
            {
                "input_ids": processed_inputs.input_ids,
                "diffu_grpo_target_ids": target_ids,
            }
        ),
        processed_inputs=processed_inputs,
        original_batch_size=1,
        original_seq_len=2,
    )
    model = MagicMock(return_value=raw_logits.clone())

    actual, _metrics, _ = forward_with_post_processing_fn(
        model=model,
        post_processing_fn=post_processor,
        processed_mb=processed_microbatch,
        sampling_params=sampling_params,
    )
    expected = (
        torch.log_softmax(raw_logits.float() / 2.0, dim=-1)
        .gather(-1, target_ids.unsqueeze(-1))
        .squeeze(-1)
    )

    assert torch.allclose(actual, expected)


def test_mask_token_exclusion_and_top_k_are_applied_before_softmax():
    logits = torch.tensor([[[9.0, 8.0, 1.0]]])
    target_ids = torch.tensor([[1]])
    sampling_params = TrainingSamplingParams(top_k=1, top_p=1.0, temperature=1.0)

    actual = same_position_logprobs(
        logits,
        target_ids,
        exclude_token_id=0,
        sampling_params=sampling_params,
    )

    assert torch.equal(actual, torch.zeros_like(actual))


def test_loss_processor_uses_noisy_segment_and_valid_token_override():
    loss_fn = MagicMock()
    loss_fn.compute_from_aligned_tensors.return_value = (
        torch.tensor(0.25),
        {"num_valid_samples": 1},
    )
    valid_toks_override = torch.tensor(3.0)
    post_processor = DiffusionLossPostProcessor(
        loss_fn=loss_fn,
        valid_toks_override=valid_toks_override,
        dp_size=1,
        **_processor_kwargs(TrainingSamplingParams()),
    )
    logits = torch.randn(1, 5, 7)
    data = BatchedDataDict(
        {
            "diffu_grpo_target_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "diffu_grpo_noisy_lengths": torch.tensor([2]),
            "diffu_grpo_loss_mask": torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]]),
            "sample_mask": torch.ones(1),
            "advantages": torch.ones(1, 5),
            "prev_logprobs": torch.zeros(1, 5),
            "generation_logprobs": torch.zeros(1, 5),
        }
    )
    processed_inputs = ProcessedInputs(input_ids=torch.ones(1, 5).long(), seq_len=5)

    loss, metrics = post_processor(
        logits=logits,
        data_dict=data,
        processed_inputs=processed_inputs,
        global_valid_seqs=torch.tensor(1.0),
        global_valid_toks=torch.tensor(99.0),
    )

    assert loss.item() == 0.25
    assert metrics["num_valid_samples"] == 1
    call = loss_fn.compute_from_aligned_tensors.call_args.kwargs
    assert call["curr_logprobs"].shape == (1, 2)
    assert call["token_mask"].shape == (1, 2)
    assert call["global_valid_toks"] is valid_toks_override
