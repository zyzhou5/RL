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

import torch

from nemo_rl.models.automodel.diffusion_attention import (
    asymmetric_semi_ar_mask_mod,
    build_asymmetric_position_ids,
    clear_hf_nld_asymmetric_mask,
    install_hf_nld_asymmetric_mask,
)


def _dense_mask(mask_mod, batch_size: int, sequence_length: int) -> torch.Tensor:
    dense = torch.zeros(batch_size, sequence_length, sequence_length, dtype=torch.bool)
    for batch_idx in range(batch_size):
        for query_idx in range(sequence_length):
            for kv_idx in range(sequence_length):
                dense[batch_idx, query_idx, kv_idx] = mask_mod(
                    torch.tensor(batch_idx),
                    torch.tensor(0),
                    torch.tensor(query_idx),
                    torch.tensor(kv_idx),
                )
    return dense


def test_asymmetric_mask_uses_prompt_and_per_sample_valid_lengths():
    prompt_lengths = torch.tensor([2, 3])
    noisy_valid_lengths = torch.tensor([4, 2])
    clean_lengths = torch.tensor([6, 5])
    mask_mod = asymmetric_semi_ar_mask_mod(
        block_size=2,
        noisy_length=4,
        noisy_response_offset=0,
        prompt_lengths=prompt_lengths,
        noisy_valid_lengths=noisy_valid_lengths,
        clean_lengths=clean_lengths,
    )
    mask = _dense_mask(mask_mod, batch_size=2, sequence_length=10)

    # Sample 0, first noisy block: its own noisy block + the two-token prompt.
    assert torch.equal(
        torch.nonzero(mask[0, 0], as_tuple=False).flatten(),
        torch.tensor([0, 1, 4, 5]),
    )
    # Second noisy block also sees the previous clean response block.
    assert torch.equal(
        torch.nonzero(mask[0, 2], as_tuple=False).flatten(),
        torch.tensor([2, 3, 4, 5, 6, 7]),
    )
    # Sample 1 has only two valid noisy positions. Invalid query rows retain a
    # self edge so flex attention never receives a fully masked row.
    assert torch.equal(
        torch.nonzero(mask[1, 2], as_tuple=False).flatten(), torch.tensor([2])
    )
    # Clean padding is excluded per sample (sample 1 clean length is five).
    assert not mask[1, :9, 9].any()


def test_asymmetric_position_ids_align_noisy_response_with_clean_sequence():
    position_ids = build_asymmetric_position_ids(
        noisy_length=4,
        clean_length=6,
        noisy_response_offset=0,
        prompt_lengths=torch.tensor([2, 3]),
        noisy_valid_lengths=torch.tensor([4, 2]),
    )

    assert torch.equal(position_ids[0], torch.tensor([2, 3, 4, 5, 0, 1, 2, 3, 4, 5]))
    assert torch.equal(position_ids[1], torch.tensor([3, 4, 0, 0, 0, 1, 2, 3, 4, 5]))


def test_hf_nld_adapter_is_narrow_and_clears_cache():
    class MinistralFlexAttention:
        def __init__(self):
            self.sbd_block_diff_mask = None
            self.block_size_orig = 16

        def set_attention_mode(self, mode, block_size=None):
            self.mode = mode
            self.block_size = block_size

    module = MinistralFlexAttention()
    sentinel = object()
    assert install_hf_nld_asymmetric_mask(module, block_mask=sentinel)
    assert module.sbd_block_diff_mask is sentinel
    assert clear_hf_nld_asymmetric_mask(module)
    assert module.sbd_block_diff_mask is None

    class UnknownAttention:
        pass

    assert not install_hf_nld_asymmetric_mask(UnknownAttention(), block_mask=sentinel)
