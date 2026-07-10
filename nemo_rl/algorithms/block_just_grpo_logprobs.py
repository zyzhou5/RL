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

"""JustGRPO *block-reveal* logprob estimation.

Computes the same leftmost-reveal token logprobs as
``just_grpo_logprobs.build_leftmost_reveal_batch`` but in ``block_size`` forward
passes instead of one-pass-per-response-token, by reusing DiffuGRPO's asymmetric
``[noisy | clean]`` block-diffusion layout (see
``diffu_grpo_logprobs.build_fully_masked_completion_batch`` and
``NemotronLabsDiffusionAttention.set_asymmetric_ar_metadata``).

Design: a single fully-masked completion ``base`` (N rows) is built once. Each
*reveal level* ``l`` is then a cheap derived **view** of that base in which every
block reveals its first ``l * k`` tokens (the rest stay MASK); the view
contributes the logprobs of that block's next ``k`` tokens -- the "harvest"
window at within-block offsets ``[l*k, (l+1)*k)``. The worker iterates levels
``l = 0 .. num_levels-1`` (a Python for-loop), so only one level (N rows) is
materialized at a time -- the number of forward passes is ``ceil(block_size /
k)`` (capped by ``max_reveal_levels``), independent of sequence length.

``k`` (``reveal_tokens_per_level``) is the semi-autoregressive reveal width: how
many tokens each block reveals (and harvests) per forward. ``k = 1`` is the exact
per-token leftmost-reveal objective. ``k > 1`` is the block-parallel
generalization: the ``k`` tokens harvested at a level are each conditioned only
on the ``l*k`` real tokens revealed before the window (the others stay MASK in
this pass), matching generation that unmasks ``k`` tokens per diffusion step
(SGLang ``max_steps = block_size / k``); it no longer reproduces the per-token
leftmost-reveal logprobs once ``k > 1``.

The worker uses ``build_block_reveal_base`` (one fully-masked base) +
``make_reveal_level_view`` (per-level reveal) in an explicit loop for logprobs,
and ``BlockJustGRPORevealSchedule`` (which emits those level views as Megatron
microbatches) for training. ``scatter_block_reveal_logprobs`` maps the harvested
per-level logprobs back to NemoRL's ``[N, S]`` layout.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Iterator

import torch

from nemo_rl.algorithms.diffu_grpo_logprobs import (
    build_fully_masked_completion_batch,
    build_fully_masked_completion_loss_batch,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

if TYPE_CHECKING:
    from nemo_rl.models.policy import BlockJustGRPOLogprobEstimationConfig


def get_block_reveal_logprob_estimation_cfg(
    cfg: dict[str, Any],
) -> "BlockJustGRPOLogprobEstimationConfig":
    estimator_cfg = cfg.get("logprob_estimation", None)
    if estimator_cfg is None:
        raise ValueError("policy.logprob_estimation must be set")
    if estimator_cfg["type"] != "just_grpo_block_reveal":
        raise ValueError(
            "policy.logprob_estimation.type must be 'just_grpo_block_reveal'"
        )
    return estimator_cfg


def count_reveal_levels(
    block_size: int,
    response_lengths: torch.Tensor,
    max_reveal_levels: int | None,
    reveal_tokens_per_level: int = 1,
) -> int:
    """Number of reveal-level passes.

    MUST be identical across all data-parallel ranks. The worker issues one
    forward pass (and its collectives) per reveal level, so if ranks computed
    different counts they would desync and deadlock at the DP process group
    (observed as a 60-min NCCL collective timeout at 16-node scale). It is
    therefore purely ``block_size`` (capped by ``max_reveal_levels``) and does
    NOT depend on ``response_lengths`` — a per-rank, batch-dependent quantity.
    Levels past a sample's response length simply harvest nothing (zero mask),
    which is correct and keeps all ranks in lockstep.

    ``response_lengths`` is accepted for signature stability but only used to
    short-circuit the empty case.
    """
    if response_lengths.numel() == 0:
        return 0
    k = max(1, int(reveal_tokens_per_level))
    num_levels = (int(block_size) + k - 1) // k  # ceil(block_size / k)
    if max_reveal_levels is not None:
        num_levels = min(num_levels, int(max_reveal_levels))
    return int(num_levels)


def require_generation_entropy(message_logs: list[Any]) -> None:
    """Fail fast when JustGRPO-Fast is active but generation produced no entropy.

    ``fast_entropy_level_ratio`` ranks tokens by the SGLang rollout per-token
    entropy. If the dLLM server ran with ``return_entropy: false`` the entropy
    channel is absent and would otherwise be silently zero-filled -> sparsification
    on a non-entropy signal. Raises unless at least one message carries a real
    ``generation_entropy``.
    """
    has_entropy = any(
        "generation_entropy" in message
        for message_log in message_logs
        for message in message_log
    )
    if not has_entropy:
        raise ValueError(
            "fast_entropy_level_ratio is set but the SGLang generation returned no "
            "per-token entropy; enable return_entropy: true in the dLLM algorithm "
            "config (FastDiffuser) so generation_entropy is produced."
        )


def build_block_topk_offsets(
    entropy: torch.Tensor,
    response_lengths: torch.Tensor,
    completion_starts: torch.Tensor,
    block_size: int,
    m: int,
) -> torch.Tensor:
    """Per-(sample, block) top-``m`` highest-entropy within-block offsets.

    ``entropy`` is the rollout per-token entropy in the original ``[N, S]``
    layout (aligned to ``data['input_ids']`` exactly like ``generation_logprobs``);
    ``completion_starts`` / ``response_lengths`` (from the block-reveal base) give
    each sample's response span. Response token ``r`` lives in block
    ``r // block_size`` at within-block offset ``r % block_size``; for each
    (sample, block) the ``m`` valid offsets with highest entropy are kept and
    returned **sorted ascending**. Blocks with fewer than ``m`` valid offsets
    (partial trailing blocks) are padded with the sentinel ``-1``, which reveals
    and harvests nothing in ``make_reveal_level_view``. Purely local per sample --
    no all-reduce, since only the level count ``m`` (not the offsets) must be
    DP-uniform, and it is a config-derived constant.
    """
    device = entropy.device
    num_samples, seq_len = entropy.shape
    response_lengths = response_lengths.to(device=device, dtype=torch.long)
    completion_starts = completion_starts.to(device=device, dtype=torch.long)
    k = int(block_size)
    max_resp = int(response_lengths.max().item()) if num_samples else 0
    num_blocks = max(1, (max_resp + k - 1) // k)

    rel = torch.arange(num_blocks * k, device=device)  # response-relative index
    valid = rel.unsqueeze(0) < response_lengths.unsqueeze(1)  # [N, num_blocks*k]
    pos = (completion_starts.unsqueeze(1) + rel.unsqueeze(0)).clamp_(0, seq_len - 1)
    gathered = entropy.gather(1, pos)
    # Entropy is >= 0, so any negative fill ranks strictly below every valid
    # offset in the top-``m`` and is recoverable as padding.
    gathered = torch.where(valid, gathered, gathered.new_full((), -1.0))
    grid = gathered.view(num_samples, num_blocks, k)  # [N, num_blocks, block_size]

    topk_vals, topk_idx = torch.topk(grid, m, dim=-1)  # [N, num_blocks, m]
    invalid = topk_vals < 0
    # ``k`` sorts after every real offset (0..k-1); restored to sentinel -1 below.
    sel = torch.where(invalid, torch.full_like(topk_idx, k), topk_idx)
    sel, _ = torch.sort(sel, dim=-1)  # ascending; padded offsets land last
    sel = torch.where(sel == k, torch.full_like(sel, -1), sel)
    return sel.to(torch.long)


def build_block_reveal_base(
    data: BatchedDataDict[Any],
    mask_token_id: int,
    pad_token_id: int,
    block_size: int,
    pad_to_length: int | None = None,
    include_loss: bool = False,
    max_reveal_levels: int | None = None,
    reveal_tokens_per_level: int = 1,
    fast_entropy_level_ratio: float | None = None,
) -> tuple[BatchedDataDict[Any], int, int]:
    """Build the fully-masked completion ``base`` (N rows) + level count.

    The noisy side is *not* block-padded (``block_size=None`` is forwarded to the
    completion-batch builder) so the final partial block is truncated at the last
    response token -- matching the per-token leftmost-reveal attention length.
    Block structure for the attention mask comes from the model module's
    ``block_size``, not from noisy padding. Returns ``(base, num_samples,
    num_levels)``.
    """
    if include_loss:
        base = build_fully_masked_completion_loss_batch(
            data,
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
            pad_to_length=pad_to_length,
            block_size=None,
        )
    else:
        base = build_fully_masked_completion_batch(
            data,
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
            pad_to_length=pad_to_length,
            block_size=None,
        )
    num_samples = base["input_ids"].shape[0]
    num_levels = count_reveal_levels(
        block_size,
        base["diffu_grpo_response_lengths"],
        max_reveal_levels,
        reveal_tokens_per_level=reveal_tokens_per_level,
    )
    # JustGRPO-Fast: keep only the top-``ratio`` highest-entropy offsets per
    # block. ``m`` is ``ceil(ratio * num_levels)`` -- a config-derived constant,
    # so it stays DP-uniform like the full level count (no collective desync).
    # ``ratio`` >= 1 (or None) leaves the full block untouched -> uniform reveal
    # path -> byte-for-byte identical to base Block JustGRPO.
    if fast_entropy_level_ratio is not None and int(reveal_tokens_per_level) != 1:
        # Fast reveal is width-1 per selected offset and ignores k, so with k > 1 it
        # would reveal same-step neighbors that were masked during generation ->
        # wrong conditioning. Disallow the combination outright.
        raise ValueError(
            "JustGRPO-Fast entropy sparsification (fast_entropy_level_ratio) "
            "currently supports reveal_tokens_per_level == 1 only; got "
            f"k={int(reveal_tokens_per_level)}."
        )
    if fast_entropy_level_ratio is not None and num_levels > 0:
        m = max(1, math.ceil(float(fast_entropy_level_ratio) * num_levels))
        if m < num_levels:
            if "generation_entropy" not in data:
                raise KeyError(
                    "BlockJustGRPO-Fast (fast_entropy_level_ratio) needs "
                    "'generation_entropy' in the batch passed to get_logprobs/train; "
                    "it was not threaded in (see grpo.py logprob_data construction)."
                )
            entropy = data["generation_entropy"].to(base["input_ids"].device)
            base["block_reveal_selected_offsets"] = build_block_topk_offsets(
                entropy=entropy,
                response_lengths=base["diffu_grpo_response_lengths"],
                completion_starts=base["diffu_grpo_completion_starts"],
                block_size=block_size,
                m=m,
            )
            num_levels = m
    return base, num_samples, num_levels


def make_reveal_level_view(
    base: BatchedDataDict[Any],
    level: int,
    block_size: int,
    harvest_keys: tuple[str, ...],
    reveal_tokens_per_level: int = 1,
) -> BatchedDataDict[Any]:
    """Derive the reveal-level-``level`` view (N rows) from a fully-masked base.

    Reveals the first ``level`` tokens of every block (real tokens) and leaves the
    rest MASK; ``harvest_keys`` (score and/or loss mask) are set to the per-row
    harvest mask (1 only at within-block offset ``level`` of valid response
    tokens). All other base fields (asymmetric-AR metadata, target ids, scattered
    advantages, ...) are passed through unchanged.
    """
    device = base["input_ids"].device
    num_samples, total_len = base["input_ids"].shape
    target_ids = base["diffu_grpo_target_ids"].to(device)
    score_mask = base["diffu_grpo_score_mask"].to(device)
    response_lengths = base["diffu_grpo_response_lengths"].to(device)
    noisy_offset = (
        int(base["diffu_grpo_noisy_response_offsets"][0].item())
        if num_samples
        else 0
    )

    col = torch.arange(total_len, device=device).unsqueeze(0)
    rel = col - noisy_offset
    within_block_off = torch.remainder(rel.clamp_min(0), int(block_size))
    in_response = (rel >= 0) & (rel < response_lengths.unsqueeze(1))

    selected = base.get("block_reveal_selected_offsets", None)
    if selected is None:
        # Uniform reveal (default / full block): every block reveals its first
        # ``level * k`` tokens and harvests the length-``k`` window after them.
        k = max(1, int(reveal_tokens_per_level))
        reveal_upto = level * k
        reveal = in_response & (within_block_off < reveal_upto)
        harvest = (
            in_response
            & (within_block_off >= reveal_upto)
            & (within_block_off < reveal_upto + k)
            & (score_mask > 0.5)
        )
    else:
        # JustGRPO-Fast: at rank ``level`` each block reveals its own selected
        # prefix ``0..o-1`` and harvests exactly its ``level``-th kept offset
        # ``o`` (sentinel -1 reveals/harvests nothing). Conditioning is
        # byte-identical to the full block-reveal view at absolute level ``o``, so
        # each kept token's logprob is unchanged; only which offsets enter the
        # loss differs.
        selected = selected.to(device=device)
        num_blocks = selected.shape[1]
        block_idx = torch.div(
            rel.clamp_min(0), int(block_size), rounding_mode="floor"
        ).clamp_(0, num_blocks - 1)
        block_idx = block_idx.expand(num_samples, -1)
        reveal_upto = torch.gather(selected[:, :, level], 1, block_idx)
        reveal = in_response & (within_block_off < reveal_upto)
        harvest = (
            in_response & (within_block_off == reveal_upto) & (score_mask > 0.5)
        )
    ids = torch.where(reveal, target_ids, base["input_ids"].to(device))
    harvest = harvest.to(dtype=score_mask.dtype)

    view = BatchedDataDict[Any]()
    for key, value in base.items():
        view[key] = value
    # The [N, num_blocks, m] selection tensor is only needed to build this view;
    # keep it out of the emitted microbatch, which carries 2-D [N, S] fields.
    if "block_reveal_selected_offsets" in view:
        del view["block_reveal_selected_offsets"]
    view["input_ids"] = ids
    for key in harvest_keys:
        view[key] = harvest
    view["token_mask"] = harvest
    view["block_reveal_harvest_mask"] = harvest
    view["block_reveal_reveal_level"] = torch.full(
        (num_samples,), int(level), device=device, dtype=torch.long
    )
    view["block_reveal_sample_index"] = torch.arange(
        num_samples, device=device, dtype=torch.long
    )
    return view


def build_block_kept_mask(
    base: BatchedDataDict[Any],
    block_size: int,
) -> torch.Tensor:
    """The ``[N, total_len]`` mask of tokens harvested across all Fast reveal levels.

    Equals the union of the ``m`` per-level harvest masks -- each selected offset is
    harvested at exactly one level, so the union is also their sum -- intersected with
    the score mask. Used to normalize the JustGRPO-Fast loss by the *kept* token count
    rather than the full response length. Requires ``block_reveal_selected_offsets`` on
    ``base`` (Fast path only). Mirrors ``make_reveal_level_view``'s harvest primitives
    so the two cannot drift.
    """
    device = base["input_ids"].device
    selected = base["block_reveal_selected_offsets"].to(device)
    num_samples, total_len = base["input_ids"].shape
    score_mask = base["diffu_grpo_score_mask"].to(device)
    response_lengths = base["diffu_grpo_response_lengths"].to(device)
    noisy_offset = (
        int(base["diffu_grpo_noisy_response_offsets"][0].item())
        if num_samples
        else 0
    )
    col = torch.arange(total_len, device=device).unsqueeze(0)
    rel = col - noisy_offset
    within_block_off = torch.remainder(rel.clamp_min(0), int(block_size))
    in_response = (rel >= 0) & (rel < response_lengths.unsqueeze(1))
    num_blocks = selected.shape[1]
    block_idx = (
        torch.div(rel.clamp_min(0), int(block_size), rounding_mode="floor")
        .clamp_(0, num_blocks - 1)
        .expand(num_samples, -1)
    )
    kept = torch.zeros_like(in_response)
    for j in range(selected.shape[2]):
        sel_j = torch.gather(selected[:, :, j], 1, block_idx)
        kept = kept | (within_block_off == sel_j)
    kept = in_response & kept & (score_mask > 0.5)
    return kept.to(dtype=score_mask.dtype)


class BlockJustGRPORevealSchedule(BatchedDataDict[Any]):
    """The reveal-level schedule, presented to Megatron as a microbatch source.

    Holds the fully-masked ``base`` (N samples) and emits the model inputs for
    each reveal level in turn: for level ``j`` it reveals the first ``j`` tokens of
    every block, then microbatches the N samples the *standard* way (delegating to
    ``BatchedDataDict.make_microbatch_iterator``). The only block-reveal-specific
    structure is the outer loop over reveal levels; sample microbatching is
    ordinary.

    Used as the training "batch" so a single ``megatron_forward_backward``
    accumulates gradients across all reveal levels before one optimizer step. Only
    one reveal level is materialized at a time.
    """

    def configure(
        self,
        *,
        num_levels: int,
        block_size: int,
        harvest_keys: tuple[str, ...],
        reveal_tokens_per_level: int = 1,
    ) -> "BlockJustGRPORevealSchedule":
        self._br_num_levels = int(num_levels)
        self._br_block_size = int(block_size)
        self._br_harvest_keys = tuple(harvest_keys)
        self._br_reveal_tokens_per_level = max(1, int(reveal_tokens_per_level))
        return self

    def _sample_count(self) -> int:
        if not self.data:
            return 0
        value = self.data[next(iter(self.data))]
        return value.shape[0] if torch.is_tensor(value) else len(value)

    @property
    def size(self) -> int:
        # Total microbatch-equivalents Megatron will see: one full sample batch
        # per reveal level. get_microbatch_iterator divides this by the microbatch
        # size to get num_microbatches.
        return self._br_num_levels * self._sample_count()

    def make_microbatch_iterator(
        self, microbatch_size: int
    ) -> Iterator[BatchedDataDict[Any]]:
        for level in range(self._br_num_levels):
            level_view = make_reveal_level_view(
                self,
                level,
                self._br_block_size,
                self._br_harvest_keys,
                self._br_reveal_tokens_per_level,
            )
            # Sample microbatching is the standard BatchedDataDict mechanism.
            yield from level_view.make_microbatch_iterator(microbatch_size)


def scatter_block_reveal_logprobs(
    flat_logprobs: torch.Tensor,
    harvest_mask: torch.Tensor,
    sample_index: torch.Tensor,
    completion_starts: torch.Tensor,
    noisy_response_offset: int,
    original_seq_len: int,
    num_samples: int,
) -> torch.Tensor:
    """Scatter per-row harvested logprobs back to NemoRL's ``[N, S]`` convention.

    ``flat_logprobs`` is ``[rows, noisy_len]`` (logprob of the target token at each
    noisy position). Only harvest positions are kept; each ``(sample, original
    position)`` is harvested by exactly one row, so a scatter-add reconstructs the
    full per-token logprob vector without double counting.
    """
    device = flat_logprobs.device
    noisy_len = flat_logprobs.shape[1]
    output = flat_logprobs.new_zeros((num_samples, original_seq_len))
    if flat_logprobs.numel() == 0:
        return output

    harvest = harvest_mask.to(device=device)[:, :noisy_len]
    contrib = flat_logprobs * harvest
    rel = torch.arange(noisy_len, device=device) - int(noisy_response_offset)
    sample_index = sample_index.to(device=device)
    completion_starts = completion_starts.to(device=device)

    for row in range(flat_logprobs.shape[0]):
        keep = harvest[row] > 0.5
        if not bool(keep.any()):
            continue
        sample = int(sample_index[row].item())
        if sample < 0 or sample >= num_samples:
            continue
        start = int(completion_starts[row].item())
        positions = (start + rel[keep]).to(dtype=torch.long)
        valid = (positions >= 0) & (positions < original_seq_len)
        if not bool(valid.any()):
            continue
        output[sample, positions[valid]] += contrib[row, keep][valid]
    return output
