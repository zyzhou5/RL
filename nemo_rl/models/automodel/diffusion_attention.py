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

"""AutoModel attention helpers for completion-only diffusion replay.

The replay layout is ``[noisy_response | clean_prompt_response]``. The published
Nemotron-Labs-Diffusion Hugging Face attention implements only the older symmetric
``[x_t | x_0]`` layout: it assumes equal halves, uses one scalar split point, and
does not mask per-sample padding. These helpers provide the asymmetric mask and
segmented position ids used by the Megatron-Bridge reference implementation.

The worker installs the resulting ``BlockMask`` through the model's existing
``sbd_block_diff_mask`` cache. This is a narrowly checked compatibility adapter;
models that expose a native ``set_asymmetric_ar_metadata`` API use that instead.
"""

from collections.abc import Callable
from typing import Any

import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask


def asymmetric_semi_ar_mask_mod(
    *,
    block_size: int,
    noisy_length: int,
    noisy_response_offset: int,
    prompt_lengths: torch.Tensor,
    noisy_valid_lengths: torch.Tensor,
    clean_lengths: torch.Tensor,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return the batch-aware asymmetric semi-AR attention predicate."""

    def asymmetric_semi_ar_mask(
        batch_idx: torch.Tensor,
        head_idx: torch.Tensor,
        query_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        del head_idx
        prompt_length = prompt_lengths[batch_idx]
        noisy_valid_length = noisy_valid_lengths[batch_idx]
        clean_length = clean_lengths[batch_idx]

        query_is_noisy = query_idx < noisy_length
        kv_is_noisy = kv_idx < noisy_length
        query_noisy_relative = query_idx - noisy_response_offset
        kv_noisy_relative = kv_idx - noisy_response_offset
        query_noisy_valid = (
            query_is_noisy
            & (query_noisy_relative >= 0)
            & (query_noisy_relative < noisy_valid_length)
        )
        kv_noisy_valid = (
            kv_is_noisy
            & (kv_noisy_relative >= 0)
            & (kv_noisy_relative < noisy_valid_length)
        )

        query_block = torch.div(query_noisy_relative, block_size, rounding_mode="floor")
        kv_block = torch.div(kv_noisy_relative, block_size, rounding_mode="floor")
        noisy_same_block = (
            query_noisy_valid & kv_noisy_valid & (query_block == kv_block)
        )

        query_clean_idx = query_idx - noisy_length
        kv_clean_idx = kv_idx - noisy_length
        query_clean_valid = (
            (~query_is_noisy)
            & (query_clean_idx >= 0)
            & (query_clean_idx < clean_length)
        )
        kv_clean_valid = (
            (~kv_is_noisy) & (kv_clean_idx >= 0) & (kv_clean_idx < clean_length)
        )

        clean_prompt = kv_clean_valid & (kv_clean_idx < prompt_length)
        clean_response_relative = kv_clean_idx - prompt_length
        clean_previous_response_blocks = (
            kv_clean_valid
            & (clean_response_relative >= 0)
            & (clean_response_relative < query_block * block_size)
        )
        noisy_query = query_noisy_valid & (
            noisy_same_block | clean_prompt | clean_previous_response_blocks
        )

        clean_causal = (
            query_clean_valid & kv_clean_valid & (kv_clean_idx <= query_clean_idx)
        )
        valid_query = query_noisy_valid | query_clean_valid
        invalid_query_self = (~valid_query) & (query_idx == kv_idx)
        return noisy_query | clean_causal | invalid_query_self

    return asymmetric_semi_ar_mask


def build_asymmetric_semi_ar_block_mask(
    *,
    block_size: int,
    noisy_length: int,
    clean_length: int,
    noisy_response_offset: int,
    prompt_lengths: torch.Tensor,
    noisy_valid_lengths: torch.Tensor,
    clean_lengths: torch.Tensor,
) -> BlockMask:
    """Build the compact flex-attention mask for one processed microbatch."""
    _validate_metadata(
        prompt_lengths=prompt_lengths,
        noisy_valid_lengths=noisy_valid_lengths,
        clean_lengths=clean_lengths,
    )
    total_length = noisy_length + clean_length
    mask_mod = asymmetric_semi_ar_mask_mod(
        block_size=block_size,
        noisy_length=noisy_length,
        noisy_response_offset=noisy_response_offset,
        prompt_lengths=prompt_lengths,
        noisy_valid_lengths=noisy_valid_lengths,
        clean_lengths=clean_lengths,
    )
    return create_block_mask(
        mask_mod,
        B=prompt_lengths.shape[0],
        H=None,
        Q_LEN=total_length,
        KV_LEN=total_length,
        device=prompt_lengths.device,
    )


def build_asymmetric_position_ids(
    *,
    noisy_length: int,
    clean_length: int,
    noisy_response_offset: int,
    prompt_lengths: torch.Tensor,
    noisy_valid_lengths: torch.Tensor,
) -> torch.Tensor:
    """Build per-sample RoPE positions for ``[noisy | clean]`` replay."""
    batch_size = prompt_lengths.shape[0]
    device = prompt_lengths.device
    noisy_positions = torch.arange(noisy_length, device=device).unsqueeze(0)
    noisy_positions = noisy_positions.expand(batch_size, -1)
    noisy_relative = noisy_positions - noisy_response_offset
    noisy_valid = (noisy_relative >= 0) & (
        noisy_relative < noisy_valid_lengths.unsqueeze(1)
    )
    noisy_position_ids = prompt_lengths.unsqueeze(1) + noisy_relative.clamp_min(0)
    noisy_position_ids = torch.where(
        noisy_valid, noisy_position_ids, torch.zeros_like(noisy_position_ids)
    )
    clean_position_ids = torch.arange(clean_length, device=device).unsqueeze(0)
    clean_position_ids = clean_position_ids.expand(batch_size, -1)
    return torch.cat((noisy_position_ids, clean_position_ids), dim=1)


def install_hf_nld_asymmetric_mask(
    module: Any,
    *,
    block_mask: BlockMask,
) -> bool:
    """Install a mask into the published HF NLD flex-attention cache.

    Returns ``False`` for an unknown attention implementation so callers can fail
    instead of silently falling back to its symmetric mask.
    """
    if module.__class__.__name__ != "MinistralFlexAttention":
        return False
    required_attributes = (
        "sbd_block_diff_mask",
        "block_size_orig",
        "set_attention_mode",
    )
    if not all(hasattr(module, name) for name in required_attributes):
        return False
    module.sbd_block_diff_mask = block_mask
    return True


def clear_hf_nld_asymmetric_mask(module: Any) -> bool:
    """Clear a mask installed by :func:`install_hf_nld_asymmetric_mask`."""
    if module.__class__.__name__ != "MinistralFlexAttention" or not hasattr(
        module, "sbd_block_diff_mask"
    ):
        return False
    module.sbd_block_diff_mask = None
    return True


def _validate_metadata(
    *,
    prompt_lengths: torch.Tensor,
    noisy_valid_lengths: torch.Tensor,
    clean_lengths: torch.Tensor,
) -> None:
    tensors = (prompt_lengths, noisy_valid_lengths, clean_lengths)
    if any(tensor.ndim != 1 for tensor in tensors):
        raise ValueError("Asymmetric semi-AR attention metadata must be 1D")
    if not (prompt_lengths.shape == noisy_valid_lengths.shape == clean_lengths.shape):
        raise ValueError(
            "Asymmetric semi-AR attention metadata tensors must have matching shapes"
        )
