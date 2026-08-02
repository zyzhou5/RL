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

from typing import Any, Optional

import ray
import torch
from megatron.core import parallel_state

from nemo_rl.algorithms.block_just_grpo_logprobs import scatter_block_reveal_logprobs
from nemo_rl.algorithms.trace_grpo_logprobs import (
    TraceGRPORevealSchedule,
    build_trace_base,
    per_sample_level_depths,
    sample_trace_levels,
    get_trace_grpo_logprob_estimation_cfg,
    make_trace_level_view,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import LogprobOutputSpec
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.diffu_grpo_megatron_policy_worker import (
    DiffuGRPOMegatronPolicyWorkerImpl,
)


class TraceGRPOMegatronPolicyWorkerImpl(DiffuGRPOMegatronPolicyWorkerImpl):
    """TraceGRPO logprobs that replay the inference denoising trajectory.

    Reuses DiffuGRPO's asymmetric ``[noisy | clean]`` layout, attention metadata,
    and same-position post-processors. The reveal level of each token is taken from
    the recorded inference ``commit_step`` (dense-ranked per sample) instead of the
    leftmost-within-block schedule. At reveal level ``L`` every token committed
    before ``L`` is revealed as real context and every token committed at ``L`` is
    harvested (see ``make_trace_level_view``). So:

    * logprobs: an outer Python for-loop runs one forward per level and sums the
      scattered per-level logprobs into the full ``[N, S]`` vector.
    * training: a ``TraceGRPORevealSchedule`` yields one level per forward, so a
      single ``forward_backward`` accumulates gradients across all levels before one
      optimizer step.

    The number of levels is data-dependent, so it is agreed across data-parallel
    ranks with a single ``all_reduce(MAX)`` in ``build_trace_base`` before the
    per-level loop -- every rank runs the identical number of forwards, samples that
    finished early riding along with a zero harvest mask.
    """

    def _validate_diffusion_algorithm_support(self) -> None:
        self._validate_diffusion_support("TraceGRPO")
        get_trace_grpo_logprob_estimation_cfg(self.cfg)

    def _trace_cfg(self):
        return get_trace_grpo_logprob_estimation_cfg(self.cfg)

    # DiffuGRPO's inherited helpers read mask_token_id via ``_diffu_grpo_cfg``; the
    # trace estimator carries the same key, so route them to our config.
    def _diffu_grpo_cfg(self):
        return self._trace_cfg()

    def _trace_block_size(self) -> int:
        cfg = self._trace_cfg()
        return int(cfg.get("block_size") or self._diffusion_block_size())

    def _noisy_tail_mode(self) -> str:
        mode = str(self._trace_cfg().get("noisy_tail_mode", "mask"))
        if mode not in ("mask", "eos", "none"):
            raise ValueError(
                f"noisy_tail_mode must be one of mask, eos, none; got {mode!r}"
            )
        return mode

    def _print_trace_num_levels(
        self,
        source: str,
        num_levels: int,
        base: Any | None = None,
        num_level_samples: int | None = None,
    ) -> None:
        # num_levels is the per-pass forward count (DP-uniform: all_reduce MAX in
        # exhaustive mode, the fixed draw count k in sampled mode). Also log this
        # rank's shard depth distribution -- the gap between mean depth and the
        # loop count is what stochastic sampling / compaction saves. Rank-0 only.
        try:
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else self.rank
            )
        except Exception:
            rank = self.rank
        if rank != 0:
            return
        msg = f"[trace_grpo] {source}: num_levels={num_levels}"
        if num_level_samples is not None:
            msg += f" (sampled k={num_level_samples})"
        if base is not None and "trace_reveal_level" in base:
            depths = per_sample_level_depths(base["trace_reveal_level"]).float()
            if depths.numel():
                msg += (
                    f" shard_depths(mean={depths.mean().item():.1f}"
                    f" p50={depths.median().item():.0f}"
                    f" max={depths.max().item():.0f}"
                    f" n={depths.numel()})"
                )
        print(msg)

    def _trace_level_seeds(self, data: Any) -> torch.Tensor:
        if "trace_level_sample_seed" not in data:
            raise RuntimeError(
                "TraceGRPO stochastic level sampling (num_level_samples) requires "
                "the per-row trace_level_sample_seed on the batch so the prev and "
                "training passes draw the SAME levels. It is set once per step by "
                "maybe_set_trace_level_seed in nemo_rl/algorithms/grpo.py."
            )
        return data["trace_level_sample_seed"]

    def _valid_toks_correction(self, batch):
        """DP-reduced count of post-first-EOS tail tokens to subtract from
        global_valid_toks (applied at the base-worker normalizer seam).

        emit_full_blocks keeps the post-EOS tail in the response -- revealed and
        score-harvested so its logprobs are computed (parity) -- but excludes it
        from the loss (diffu_grpo_loss_mask). The loss normalizer must match the
        loss coverage, so the tail tokens are removed from the token count. Uses
        the same ``token_mask[:, 1:] * sample_mask`` convention as
        ``process_global_batch``, so this equals a pre-first-EOS trim.
        """
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None or "token_mask" not in batch or "input_ids" not in batch:
            return 0
        tm = batch["token_mask"]
        ids = batch["input_ids"]
        sm = batch["sample_mask"]
        resp = tm > 0.5
        eos_hits = ((ids == int(eos_id)) & resp).to(torch.int32)
        after_first = (eos_hits.cumsum(dim=1) - eos_hits) > 0
        drop = (after_first & resp).to(tm.dtype)
        local = torch.sum(drop[:, 1:] * sm.unsqueeze(-1))
        local = local.cuda() if torch.cuda.is_available() else local
        torch.distributed.all_reduce(
            local, group=parallel_state.get_data_parallel_group()
        )
        return local

    # ---- training: lazy reveal-level schedule (one optimizer step) ----------
    def _build_training_megatron_batch(
        self,
        data: BatchedDataDict[Any],
        mbs: int,
    ) -> tuple[BatchedDataDict[Any], PolicyConfig, int, dict[str, Any]]:
        cfg = self._trace_cfg()
        block_size = self._trace_block_size()
        self._maybe_print_diffusion_block_size("trace_grpo_train", block_size)
        base, _num_samples, num_levels = build_trace_base(
            data,
            mask_token_id=cfg["mask_token_id"],
            pad_token_id=self.tokenizer.pad_token_id,
            noisy_block_size=self._trace_block_size(),
            noisy_tail_mode=self._noisy_tail_mode(),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_to_length=self._diffu_grpo_sequence_length_round(),
            include_loss=True,
            max_reveal_levels=cfg.get("max_reveal_levels"),
        )
        # One forward per level; samples within a level are microbatched the
        # standard way by train_micro_batch_size (passed in as ``mbs``). A single
        # forward_backward over the whole schedule accumulates gradients across
        # all levels before one optimizer step. With ``num_level_samples`` set,
        # only k per-row sampled levels run instead of every level; the draws are
        # shared with the prev-logprob pass via trace_level_sample_seed and the
        # loss mask carries the depth/k inverse-inclusion weight.
        num_level_samples = cfg.get("num_level_samples")
        sampled_levels = loss_weights = None
        loop_levels = num_levels
        if num_level_samples is not None and num_levels > 0:
            sampled_levels, loss_weights = sample_trace_levels(
                base, int(num_level_samples), self._trace_level_seeds(data)
            )
            loop_levels = int(num_level_samples)
        self._print_trace_num_levels(
            "train", loop_levels, base=base, num_level_samples=num_level_samples
        )
        schedule = TraceGRPORevealSchedule(base).configure(
            num_levels=loop_levels,
            harvest_keys=("diffu_grpo_score_mask", "diffu_grpo_loss_mask"),
            sampled_levels=sampled_levels,
            loss_weights=loss_weights,
        )
        return (
            schedule,
            self._cfg_for_diffu_grpo_sequence(base["input_ids"].shape[1]),
            mbs,
            {},
        )

    # ---- logprobs: explicit for-loop over reveal levels ---------------------
    def get_logprobs(
        self,
        *,
        data: BatchedDataDict[Any],
        micro_batch_size: Optional[int] = None,
    ) -> BatchedDataDict[LogprobOutputSpec]:
        self._validate_diffusion_algorithm_support()
        cfg = self._trace_cfg()
        reveal_mbs = (
            micro_batch_size
            if micro_batch_size is not None
            else self.cfg["logprob_batch_size"]
        )
        # No post-EOS token_mask trim here: build_trace_base builds the layout
        # from the FULL response (tail revealed + score-harvested for logprobs)
        # and excludes the tail from the loss via diffu_grpo_loss_mask. Both the
        # prev/reference base (here) and the training base use the full response,
        # so exp(curr - prev) stays 1 on-policy.
        base, num_samples, num_levels = build_trace_base(
            data,
            mask_token_id=cfg["mask_token_id"],
            pad_token_id=self.tokenizer.pad_token_id,
            noisy_block_size=self._trace_block_size(),
            noisy_tail_mode=self._noisy_tail_mode(),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_to_length=self._diffu_grpo_sequence_length_round(),
            include_loss=False,
            max_reveal_levels=cfg.get("max_reveal_levels"),
        )
        num_level_samples = cfg.get("num_level_samples")
        sampled_levels = None
        loop_levels = num_levels
        if num_level_samples is not None and num_levels > 0:
            # Same seeded draws as the training pass (no loss weights here: the
            # scattered logprobs must stay exact on the sampled positions).
            sampled_levels, _ = sample_trace_levels(
                base, int(num_level_samples), self._trace_level_seeds(data)
            )
            loop_levels = int(num_level_samples)
        self._print_trace_num_levels(
            "logprobs", loop_levels, base=base, num_level_samples=num_level_samples
        )
        original_seq_len = int(data["input_ids"].shape[1])
        if num_levels == 0:
            empty = torch.zeros_like(data["input_ids"], dtype=torch.float32)
            return BatchedDataDict[LogprobOutputSpec](logprobs=empty).to("cpu")

        # Each forward selects its level view through the ``self._tr_*`` instance
        # state that ``_build_logprob_megatron_batch`` reads back; ``super().
        # get_logprobs`` (the diffusion per-level path) is called once per level and
        # the scattered per-level logprobs are summed into the [N, S] vector. The
        # pass count is ``num_levels`` on every rank (agreed in build_trace_base),
        # so this stays DP-uniform.
        self._tr_base = base
        self._tr_sampled_levels = sampled_levels
        self._tr_reveal_mbs = reveal_mbs
        self._tr_num_samples = num_samples
        self._tr_original_seq_len = original_seq_len
        accumulated: Optional[torch.Tensor] = None
        try:
            for level in range(loop_levels):
                self._tr_level = level
                level_out = super().get_logprobs(
                    data=data, micro_batch_size=reveal_mbs
                )["logprobs"]
                accumulated = (
                    level_out if accumulated is None else accumulated + level_out
                )
        finally:
            self._tr_base = None
            self._tr_sampled_levels = None
        return BatchedDataDict[LogprobOutputSpec](logprobs=accumulated).to("cpu")

    def _build_logprob_megatron_batch(
        self,
        data: BatchedDataDict[Any],
        micro_batch_size: Optional[int],
    ) -> tuple[
        BatchedDataDict[Any] | None,
        PolicyConfig,
        int,
        dict[str, Any],
    ]:
        # Called by super().get_logprobs once per reveal level; builds the view for
        # the level currently selected by the outer loop.
        base = self._tr_base
        level_arg = (
            self._tr_sampled_levels[:, self._tr_level]
            if getattr(self, "_tr_sampled_levels", None) is not None
            else self._tr_level
        )
        view = make_trace_level_view(
            base, level_arg, ("diffu_grpo_score_mask",)
        )
        level_mbs = (
            micro_batch_size if micro_batch_size is not None else self._tr_reveal_mbs
        )
        noisy_response_offset = int(
            view["diffu_grpo_noisy_response_offsets"][0].item()
        )
        return (
            view,
            self._cfg_for_diffu_grpo_sequence(view["input_ids"].shape[1]),
            level_mbs,
            {
                "num_samples": self._tr_num_samples,
                "original_seq_len": self._tr_original_seq_len,
                "noisy_response_offset": noisy_response_offset,
            },
        )

    def _finalize_logprobs_from_outputs(
        self,
        list_of_logprobs: list[dict[str, torch.Tensor]],
        *,
        original_data: BatchedDataDict[Any],
        transformed_data: BatchedDataDict[Any],
        metadata: dict[str, Any],
    ) -> torch.Tensor:
        flat_logprobs = torch.cat(
            [lp["logprobs"] for lp in list_of_logprobs], dim=0
        )
        return scatter_block_reveal_logprobs(
            flat_logprobs=flat_logprobs,
            harvest_mask=transformed_data["block_reveal_harvest_mask"],
            sample_index=transformed_data["block_reveal_sample_index"],
            completion_starts=transformed_data["diffu_grpo_completion_starts"],
            noisy_response_offset=metadata["noisy_response_offset"],
            original_seq_len=metadata["original_seq_len"],
            num_samples=metadata["num_samples"],
        )


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker(
        "trace_grpo_megatron_policy_worker"
    )
)  # pragma: no cover
class TraceGRPOMegatronPolicyWorker(TraceGRPOMegatronPolicyWorkerImpl):
    pass
