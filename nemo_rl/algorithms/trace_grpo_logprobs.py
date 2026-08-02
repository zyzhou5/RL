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
"""TraceGRPO logprob estimation (inference-trajectory replay).

Same multi-level reveal machinery as
``block_just_grpo_logprobs.py`` (a single fully-masked ``[noisy | clean]`` base,
scored one reveal *level* per forward pass, harvested logprobs scattered back to
NemoRL's ``[N, S]`` layout via ``block_just_grpo_logprobs.scatter_block_reveal_logprobs``),
with one change: the reveal order is taken from the *actual* inference denoising
trajectory instead of the deterministic leftmost-within-block schedule.

During rollout, SGLang FastDiffuser (``logprob_mode="trajectory"``) records, for
each generated token, the block-relative denoising step (``commit_step``) that
committed it. Here we replay that trajectory: at reveal level ``L`` every token
committed *before* ``L`` is revealed (real target token) and every token
committed *at* ``L`` is harvested (scored on exactly the context inference
conditioned on). Because tokens committed together in one confidence step were
genuinely produced in a single parallel forward from the same context, harvesting
them together in one training forward is exact (unlike block-JustGRPO's ``k > 1``
approximation).

Within-block reveal levels, blocks in parallel (see ``compute_reveal_levels``): the
reveal level of a token is its denoising step within its block (``commit_step``,
dense-ranked per sample), NOT a cross-block ordering. At level ``i`` every block
reveals its step-``<i`` tokens and harvests its step-``==i`` tokens in one forward;
blocks advance together because previous-block context comes from the clean side of
the asymmetric semi-AR attention mask, so no block loop is needed. This is
block-JustGRPO's structure with a trajectory-driven reveal instead of the leftmost
one. Dense-ranking the steps drops the force-commit gap, so ``num_levels`` = max
denoising steps rather than ``max_steps + 1``.

The number of levels is data-dependent (unlike block-JustGRPO's pure-config
count), so it is agreed across data-parallel ranks with a single
``all_reduce(MAX)`` (clamped by ``max_reveal_levels``); every rank then runs the
identical number of forwards, samples that finished early riding along with a zero
harvest mask. This keeps all ranks in collective lockstep -- the same
deadlock-safety contract as block-JustGRPO.

Structurally this is a ``RevealLevelSchedule`` plug-in (see
``diffu_grpo_logprobs.py``), mirroring ``coupled_grpo_logprobs.py``: a
``build_trace_base`` builder, a ``make_trace_level_view`` per-level view, and a
``TraceGRPORevealSchedule`` subclass whose ``_make_level_view`` is the trajectory
reveal predicate. Reuses DiffuGRPO's layout/attention/post-processors wholesale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from nemo_rl.algorithms.diffu_grpo_logprobs import (
    RevealLevelSchedule,
    _scatter_original_response_values,
    build_fully_masked_completion_batch,
    build_fully_masked_completion_loss_batch,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

if TYPE_CHECKING:
    from nemo_rl.models.policy import TraceGRPOLogprobEstimationConfig

__all__ = [
    "get_trace_grpo_logprob_estimation_cfg",
    "compute_reveal_levels",
    "count_reveal_levels",
    "build_trace_base",
    "maybe_set_trace_level_seed",
    "per_sample_level_depths",
    "sample_trace_levels",
    "make_trace_level_view",
    "TraceGRPORevealSchedule",
]


# Sentinel reveal level for positions outside the response (never revealed or
# harvested). Larger than any real dense rank, so ``reveal_level < L`` is always
# False and ``reveal_level == L`` is never True for these positions.
_NON_RESPONSE_LEVEL = 1 << 30


def get_trace_grpo_logprob_estimation_cfg(
    cfg: dict[str, Any],
) -> "TraceGRPOLogprobEstimationConfig":
    estimator_cfg = cfg.get("logprob_estimation", None)
    if estimator_cfg is None:
        raise ValueError("policy.logprob_estimation must be set")
    if estimator_cfg["type"] != "trace_grpo":
        raise ValueError("policy.logprob_estimation.type must be 'trace_grpo'")
    return estimator_cfg


def compute_reveal_levels(
    reveal_steps: torch.Tensor,
    completion_starts: torch.Tensor,
    response_lengths: torch.Tensor,
) -> torch.Tensor:
    """Per-token reveal levels ``[N, S]``: each sample's commit steps made gapless-ordinal.

    ``reveal_steps`` is the response-aligned per-token denoising step (block-relative,
    from SGLang). Per sample, the distinct steps in its response slice
    ``[start : start + response_len]`` are dense-ranked ``0, 1, 2, ...``. The only value
    this moves is the force-commit sentinel (``== max_steps``), which collapses onto the
    next contiguous level instead of leaving a gap -- so ``num_levels`` is the number of
    distinct denoising steps, not ``max_steps + 1``. The level is the within-block step,
    so blocks stay parallel: at level ``i`` every block reveals its step-``<i`` tokens
    and harvests step-``==i`` in one forward (previous-block context comes from the clean
    side of the semi-AR mask, so no block ordering is needed here). Ranks are written back
    at the response slice; non-response positions stay 0 here and are sentinel-marked
    after the scatter into the noisy layout (see ``build_trace_base``).
    """
    reveal_level = torch.zeros_like(reveal_steps, dtype=torch.long)
    for sample in range(reveal_steps.shape[0]):
        start = int(completion_starts[sample].item())
        rlen = int(response_lengths[sample].item())
        if rlen <= 0:
            continue
        steps = reveal_steps[sample, start : start + rlen].to(torch.long)
        # Every scored response token must be committed (commit_step >= 0).
        # FastDiffuser only leaves commit_step == -1 on EOS-frozen / uncommitted tail
        # positions, which must be trimmed before scoring; a -1 here would dense-rank
        # to a phantom level 0 and shift every real level up.
        assert bool((steps >= 0).all()), (
            f"TraceGRPO: uncommitted reveal step (-1) in sample {sample}'s scored "
            "response; the EOS-frozen tail must be trimmed before scoring."
        )
        distinct = torch.unique(steps, sorted=True)
        reveal_level[sample, start : start + rlen] = torch.searchsorted(distinct, steps)
    return reveal_level


def count_reveal_levels(
    reveal_level: torch.Tensor,
    max_reveal_levels: int | None,
) -> int:
    """Globally-agreed number of reveal-level passes.

    The per-sample level count is data-dependent, so it MUST be reconciled across
    all data-parallel ranks before the per-level loop -- otherwise ranks would
    issue a different number of forward passes and deadlock at the DP process
    group (the 60-min NCCL collective timeout). This uses a single unconditional
    ``all_reduce(MAX)`` of the per-rank maximum per-sample denoising-step count
    (blocks advance in parallel, so this is the max steps over blocks),
    which every rank calls, so all ranks agree and then run the same number of
    forwards. Levels past a sample's trajectory harvest nothing (zero mask), which
    is correct and keeps every rank in lockstep. Optionally clamped by
    ``max_reveal_levels``.
    """
    in_response = reveal_level < _NON_RESPONSE_LEVEL
    if bool(in_response.any()):
        # dense ranks are 0-based, so level count = max rank + 1
        local_max = int(reveal_level[in_response].max().item()) + 1
    else:
        local_max = 0

    num_levels = torch.tensor(local_max, dtype=torch.long)
    if dist.is_available() and dist.is_initialized():
        num_levels = num_levels.cuda() if torch.cuda.is_available() else num_levels
        dist.all_reduce(num_levels, op=dist.ReduceOp.MAX)
    num_levels = int(num_levels.item())

    if max_reveal_levels is not None:
        num_levels = min(num_levels, int(max_reveal_levels))
    return num_levels


def build_trace_base(
    data: BatchedDataDict[Any],
    mask_token_id: int,
    pad_token_id: int,
    noisy_block_size: int | None = None,
    noisy_tail_mode: str = "mask",
    eos_token_id: int | None = None,
    pad_to_length: int | None = None,
    include_loss: bool = False,
    max_reveal_levels: int | None = None,
) -> tuple[BatchedDataDict[Any], int, int]:
    """Build the fully-masked completion base (N rows) + per-token reveal levels.

    One fully-masked ``[noisy | clean]`` base is built. With ``noisy_block_size``
    set, the noisy side is padded to full blocks so the final block keeps its
    trailing positions -- SGLang trims the returned response at the first EOS, but
    generation scored the EOS commit inside a FULL block (trailing positions MASK
    at commit time, EOS after propagation). ``noisy_tail_mode`` (the mainline
    diffu_grpo knob: "mask" / "eos" / "none") chooses what those trailing pad
    positions hold during replay. Neither static fill is exact for every token
    in the final block (the tail state changed mid-block at the EOS step);
    measured with no padding the EOS-token replay bias was -0.4 nats, and only
    emit_full_blocks generation is exact (see analyze_trace_eos_parity.py). The
    recorded per-token ``data["reveal_steps"]`` are dense-ranked into gapless-ordinal
    levels and ``num_levels`` is agreed across ranks (both layout-independent), then the
    levels are scattered into the noisy layout and stashed as ``trace_reveal_level``.
    Returns ``(base, num_samples, num_levels)``.
    """
    if include_loss:
        base = build_fully_masked_completion_loss_batch(
            data,
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
            pad_to_length=pad_to_length,
            block_size=noisy_block_size,
            noisy_tail_mode=noisy_tail_mode,
            eos_token_id=eos_token_id,
        )
    else:
        base = build_fully_masked_completion_batch(
            data,
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
            pad_to_length=pad_to_length,
            block_size=noisy_block_size,
            noisy_tail_mode=noisy_tail_mode,
            eos_token_id=eos_token_id,
        )
    num_samples = base["input_ids"].shape[0]
    total_len = base["input_ids"].shape[1]

    if "reveal_steps" not in data:
        raise RuntimeError(
            "TraceGRPO requires per-token inference reveal steps, but "
            "reveal_steps is absent from the rollout data. The SGLang rollout must "
            "run FastDiffuser with logprob_mode: trajectory AND return_reveal_steps: "
            "true (tools/nemotron_diffusion/trace_fastdiffuser.yaml); point "
            "policy.generation.sglang_cfg.dllm_algorithm_config at that file."
        )
    reveal_steps = data["reveal_steps"].to(base["input_ids"].device)
    completion_starts = base["diffu_grpo_completion_starts"]
    response_lengths = base["diffu_grpo_response_lengths"]

    # Fail fast on the config-mismatch signature: when the rollout did not opt into
    # the reveal-step channel, the SGLang worker zero-fills reveal_steps, so every
    # response token reports step 0. A real confidence trajectory over block_size>1
    # always has later-step / force-commit tokens, so all-zero reveal_steps means the
    # FastDiffuser dllm config is missing logprob_mode: trajectory /
    # return_reveal_steps: true -- guard rather than train a degenerate schedule.
    if num_samples > 0 and not bool((reveal_steps != 0).any()):
        raise RuntimeError(
            "TraceGRPO received all-zero reveal steps: the SGLang rollout "
            "did not emit the inference trajectory. Set logprob_mode: trajectory "
            "AND return_reveal_steps: true in the FastDiffuser dllm_algorithm_config "
            "(tools/nemotron_diffusion/trace_fastdiffuser.yaml)."
        )

    # Trajectory -> gapless-ordinal levels -> num_levels (both layout-independent),
    # then place the levels into the noisy layout.
    reveal_level_orig = compute_reveal_levels(
        reveal_steps, completion_starts, response_lengths
    )
    num_levels = count_reveal_levels(reveal_level_orig, max_reveal_levels)

    reveal_level = _scatter_original_response_values(
        values=reveal_level_orig,
        total_length=total_len,
        completion_starts=completion_starts,
        response_lengths=response_lengths,
    )
    # Non-response gets the sentinel so it never reveals/harvests (the scatter
    # zero-fills, which would otherwise read as level 0). Response occupies
    # [0:response_len] in the noisy layout (NOISY_RESPONSE_OFFSET == 0).
    in_response = torch.arange(
        total_len, device=reveal_level.device
    ).unsqueeze(0) < response_lengths.to(reveal_level.device).unsqueeze(1)
    reveal_level = torch.where(
        in_response, reveal_level, torch.full_like(reveal_level, _NON_RESPONSE_LEVEL)
    )
    base["trace_reveal_level"] = reveal_level

    if eos_token_id is not None and num_samples > 0:
        # With emit_full_blocks generation, the response includes the decoder's
        # post-EOS trajectory tokens (real commits + propagated EOS). They are
        # context-only for the GRADIENT: zero their LOSS channel so they are
        # never trained on, but KEEP the score channel so their logprobs are
        # still computed (post-EOS parity). reveal uses target_ids + reveal_level,
        # so the tail stays revealed as context for the terminal EOS. No-op when
        # responses are trimmed at the first EOS.
        dev = base["input_ids"].device
        tgt = base["diffu_grpo_target_ids"].to(dev)
        col2 = torch.arange(total_len, device=dev).unsqueeze(0)
        in_resp = col2 < response_lengths.to(dev).unsqueeze(1)
        eos_hits = ((tgt == int(eos_token_id)) & in_resp).to(torch.int32)
        after_first_eos = (eos_hits.cumsum(dim=1) - eos_hits) > 0
        drop = after_first_eos & in_resp
        if bool(drop.any()) and "diffu_grpo_loss_mask" in base:
            keep = (~drop).to(base["diffu_grpo_loss_mask"].dtype)
            base["diffu_grpo_loss_mask"] = (
                base["diffu_grpo_loss_mask"].to(dev) * keep
            )
    return base, num_samples, num_levels


def maybe_set_trace_level_seed(
    data: BatchedDataDict[Any],
    policy_cfg: dict[str, Any],
    step: int,
) -> None:
    """Attach the per-row stochastic level-sampling seed to ``data`` in place.

    No-op unless ``policy_cfg`` selects the ``trace_grpo`` estimator with
    ``num_level_samples`` set (stochastic level sampling). The seed is shared
    across a step's prev / reference / training forwards so every pass draws the
    SAME trajectory levels per row (required for a valid GRPO ratio) and varies
    by ``(row, step)`` so levels resample each step. Mirrors
    ``maybe_set_coupled_grpo_seed`` (same ``1_000_003`` per-step stride
    contract; distinct data key, so the two never collide).
    """
    estimation_cfg = policy_cfg.get("logprob_estimation", {})
    if estimation_cfg.get("type") != "trace_grpo":
        return
    if estimation_cfg.get("num_level_samples") is None:
        return
    seed_base = int(estimation_cfg.get("seed_base", 0))
    data["trace_level_sample_seed"] = (
        seed_base
        + step * 1_000_003
        + torch.arange(data["input_ids"].shape[0], dtype=torch.long)
    )


def per_sample_level_depths(reveal_level: torch.Tensor) -> torch.Tensor:
    """Per-sample trajectory depth ``[N]``: how many reveal levels each sample needs.

    Dense ranks are contiguous ``0..depth-1``, so depth = (max in-response rank)+1;
    samples with no response tokens get 0.
    """
    in_response = reveal_level < _NON_RESPONSE_LEVEL
    masked = torch.where(
        in_response, reveal_level, torch.full_like(reveal_level, -1)
    )
    return (masked.max(dim=1).values + 1).clamp_min(0)


def sample_trace_levels(
    base: BatchedDataDict[Any],
    num_level_samples: int,
    seeds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw per-row trajectory levels for stochastic level sampling.

    For each row, draws ``k_s = min(num_level_samples, depth_s)`` DISTINCT levels
    uniformly from ``[0, depth_s)`` (without replacement, so no token is ever
    harvested twice and the per-draw logprob scatter-sum stays exact on the
    sampled set). Draws ``j >= k_s`` are padded with ``depth_s`` (reveals the
    whole response, harvests nothing) so every row participates in exactly
    ``num_level_samples`` lockstep forwards regardless of its depth.

    Returns ``(levels [N, k] long, loss_weights [N] float)``.
    ``loss_weights[s] = depth_s / k_s`` is the inverse-inclusion-probability
    importance weight: a token of row ``s`` (at any of its levels) is harvested
    with probability ``k_s / depth_s`` per step, so the weighted per-token
    policy-gradient sum stays unbiased (self-normalized under the loss's masked
    normalization). Zero for empty rows. Applied to the LOSS mask channel only
    (see ``make_trace_level_view``) -- score/harvest masks stay binary.
    """
    depths = per_sample_level_depths(base["trace_reveal_level"])
    num_samples = int(depths.shape[0])
    k = int(num_level_samples)
    if k <= 0:
        raise ValueError(f"num_level_samples must be positive, got {k}")
    if seeds.shape[0] != num_samples:
        raise ValueError(
            f"trace_level_sample_seed has {seeds.shape[0]} rows for "
            f"{num_samples} samples"
        )
    device = base["input_ids"].device
    levels = torch.empty((num_samples, k), dtype=torch.long)
    weights = torch.zeros((num_samples,), dtype=torch.float32)
    seeds_list = seeds.tolist()
    depths_list = depths.tolist()
    for s in range(num_samples):
        depth = int(depths_list[s])
        levels[s, :] = depth  # pad draws: reveal-all / harvest-none
        if depth <= 0:
            continue
        k_s = min(k, depth)
        gen = torch.Generator()
        gen.manual_seed(int(seeds_list[s]) & 0x7FFFFFFFFFFFFFFF)
        levels[s, :k_s] = torch.randperm(depth, generator=gen)[:k_s]
        weights[s] = depth / k_s
    return levels.to(device), weights.to(device)


def make_trace_level_view(
    base: BatchedDataDict[Any],
    level: int | torch.Tensor,
    harvest_keys: tuple[str, ...],
    loss_weights: torch.Tensor | None = None,
) -> BatchedDataDict[Any]:
    """Derive a reveal-level view (N rows) from a fully-masked trace base.

    ``level`` is either a scalar (exhaustive schedule: same level for every row)
    or a per-row ``[N]`` tensor (stochastic level sampling: each row is rendered
    at its own drawn level in the same forward). Trajectory reveal predicate,
    rowwise:

    * reveal  = in_response AND ``reveal_level <  level``  -> put real target token
    * harvest = in_response AND ``reveal_level == level`` AND score_mask -> score it

    ``loss_weights`` (per-row, optional) is the stochastic-sampling
    inverse-inclusion-probability weight; it multiplies ONLY the
    ``diffu_grpo_loss_mask`` channel -- score/harvest/token masks stay binary so
    logprob extraction and the level scatter-sum remain exact.
    Everything else (asymmetric-AR metadata, target ids, scattered advantages, ...)
    is passed through unchanged, exactly like the block-reveal / coupled views.
    Reuses block-reveal's ``block_reveal_*`` field names so
    ``scatter_block_reveal_logprobs`` and the post-processors work untouched.
    """
    device = base["input_ids"].device
    num_samples = base["input_ids"].shape[0]
    target_ids = base["diffu_grpo_target_ids"].to(device)
    score_mask = base["diffu_grpo_score_mask"].to(device)
    reveal_level = base["trace_reveal_level"].to(device)

    if torch.is_tensor(level):
        level_rows = level.to(device=device, dtype=torch.long)
    else:
        level_rows = torch.full(
            (num_samples,), int(level), device=device, dtype=torch.long
        )
    level_col = level_rows.unsqueeze(1)

    reveal = reveal_level < level_col
    ids = torch.where(reveal, target_ids, base["input_ids"].to(device))
    score_harvest = ((reveal_level == level_col) & (score_mask > 0.5)).to(
        dtype=score_mask.dtype
    )
    # Loss harvest respects the post-first-EOS-zeroed loss mask, so the gradient
    # excludes the tail while score_harvest still harvests it for logprobs.
    _loss_mask_base = base.get("diffu_grpo_loss_mask", None)
    if _loss_mask_base is not None:
        loss_harvest = (
            (reveal_level == level_col) & (_loss_mask_base.to(device) > 0.5)
        ).to(dtype=score_mask.dtype)
    else:
        loss_harvest = score_harvest

    view = BatchedDataDict[Any]()
    for key, value in base.items():
        view[key] = value
    view["input_ids"] = ids
    for key in harvest_keys:
        if key == "diffu_grpo_loss_mask":
            h = loss_harvest
            if loss_weights is not None:
                h = h * loss_weights.to(
                    device=device, dtype=h.dtype
                ).unsqueeze(1)
            view[key] = h
        else:
            view[key] = score_harvest
    view["token_mask"] = score_harvest
    view["block_reveal_harvest_mask"] = score_harvest
    view["block_reveal_reveal_level"] = level_rows
    view["block_reveal_sample_index"] = torch.arange(
        num_samples, device=device, dtype=torch.long
    )
    return view


class TraceGRPORevealSchedule(RevealLevelSchedule):
    """The trajectory reveal-level schedule, presented to Megatron as a microbatch source.

    Holds the fully-masked base (N samples, with per-token ``trace_reveal_level``
    already computed) and emits the model inputs for each reveal level in turn,
    microbatching each level the *standard* way (see ``RevealLevelSchedule``). Only
    the per-level view predicate differs from coupled's schedule. Used as the
    training "batch" so a single ``megatron_forward_backward`` accumulates gradients
    across all reveal levels before one optimizer step.
    """

    def configure(
        self,
        *,
        num_levels: int,
        harvest_keys: tuple[str, ...],
        sampled_levels: torch.Tensor | None = None,
        loss_weights: torch.Tensor | None = None,
    ) -> "TraceGRPORevealSchedule":
        self._configure_levels(num_levels=num_levels, harvest_keys=harvest_keys)
        # Stochastic level sampling: ``sampled_levels [N, k]`` gives each row its
        # own level per draw; ``num_levels`` is then the (DP-uniform) draw count k.
        self._tr_sampled_levels = sampled_levels
        self._tr_loss_weights = loss_weights
        return self

    def _make_level_view(self, level: int) -> BatchedDataDict[Any]:
        if self._tr_sampled_levels is not None:
            return make_trace_level_view(
                self,
                self._tr_sampled_levels[:, level],
                self._rl_harvest_keys,
                loss_weights=self._tr_loss_weights,
            )
        return make_trace_level_view(self, level, self._rl_harvest_keys)
