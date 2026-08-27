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

"""BlockJustGRPO(-Fast) on the Automodel (DTensor) backend.

Automodel counterpart of ``block_just_grpo_megatron_policy_worker``. The estimator
math is imported unchanged from ``nemo_rl.algorithms.block_just_grpo_logprobs``,
which has no backend imports -- only the driving of forwards differs between the two
backends, which is the whole point of the hook set.

Scoring cost, for orientation (block_size=16, 512-token response, 32 blocks):

    naive   512 forwards   one per token
    Block    16 forwards   score offset j of EVERY block at once (exact)
    Fast      4 forwards   keep the top ceil(0.25*16) highest-entropy offsets

"Fast" needs per-token entropy from the generation engine. Its absence is a
fail-fast (``require_generation_entropy``) rather than a silent zero-fill, because
sparsifying on zeros would rank positions arbitrarily and still train.
"""

from typing import Any, Optional

import ray
import torch

from nemo_rl.algorithms.block_just_grpo_logprobs import (
    BlockJustGRPORevealSchedule,
    build_block_kept_mask,
    build_block_reveal_base,
    get_block_reveal_logprob_estimation_cfg,
    make_reveal_level_view,
    scatter_block_reveal_logprobs,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import LogprobOutputSpec
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.diffusion_dtensor_policy_worker import (
    DiffusionDTensorPolicyWorkerImpl,
)


class BlockJustGRPODTensorPolicyWorkerImpl(DiffusionDTensorPolicyWorkerImpl):
    """Leftmost-reveal logprobs computed in ``block_size`` (or top-m) passes."""

    # ---- config ------------------------------------------------------------

    def _validate_diffusion_algorithm_support(self) -> None:
        self._validate_diffusion_support("BlockJustGRPO")
        paradigm = (self.cfg.get("hf_config_overrides") or {}).get(
            "dlm_paradigm"
        ) or getattr(getattr(self, "model_config", None), "dlm_paradigm", None)
        if paradigm != "sbd_block_diff":
            raise NotImplementedError(
                "BlockJustGRPO completion-only replay on AutoModel requires "
                "dlm_paradigm='sbd_block_diff'; the block_diff cache follows a "
                "different symmetric attention contract."
            )
        get_block_reveal_logprob_estimation_cfg(self.cfg)

    def _block_reveal_cfg(self) -> Any:
        return get_block_reveal_logprob_estimation_cfg(self.cfg)

    def _block_reveal_block_size(self) -> int:
        cfg = self._block_reveal_cfg()
        return int(cfg.get("block_size") or self._diffusion_block_size())

    def _block_reveal_tokens_per_level(self) -> int:
        return max(1, int(self._block_reveal_cfg().get("reveal_tokens_per_level") or 1))

    def _mask_token_id(self) -> int:
        return int(self._block_reveal_cfg()["mask_token_id"])

    def _exclude_mask_token(self) -> bool:
        return bool(
            self._block_reveal_cfg().get("exclude_mask_token_from_logits", False)
        )

    # ---- post-processors ---------------------------------------------------

    def _make_logprobs_post_processor(self, **kwargs: Any) -> Any:
        from nemo_rl.models.automodel.diffusion_train import (
            DiffusionLogprobsPostProcessor,
        )

        return DiffusionLogprobsPostProcessor(
            exclude_token_id=self._mask_token_id()
            if self._exclude_mask_token()
            else None,
            **kwargs,
        )

    def _make_loss_post_processor(self, **kwargs: Any) -> Any:
        from nemo_rl.models.automodel.diffusion_train import DiffusionLossPostProcessor

        return DiffusionLossPostProcessor(
            exclude_token_id=self._mask_token_id()
            if self._exclude_mask_token()
            else None,
            valid_toks_override=getattr(self, "_fast_valid_toks", None),
            **kwargs,
        )

    # ---- training ----------------------------------------------------------

    def _build_training_batch(
        self, data: BatchedDataDict[Any], mbs: int
    ) -> tuple[BatchedDataDict[Any], PolicyConfig, int, dict[str, Any]]:
        cfg = self._block_reveal_cfg()
        block_size = self._block_reveal_block_size()
        reveal_k = self._block_reveal_tokens_per_level()

        base, _n_samples, num_levels, selected_offsets = build_block_reveal_base(
            data,
            mask_token_id=cfg["mask_token_id"],
            pad_token_id=self.tokenizer.pad_token_id,
            block_size=block_size,
            pad_to_length=None,
            include_loss=True,
            max_reveal_levels=cfg.get("max_reveal_levels"),
            reveal_tokens_per_level=reveal_k,
            fast_entropy_level_ratio=cfg.get("fast_entropy_level_ratio"),
            eos_token_id=self.tokenizer.eos_token_id,
            force_eos=cfg.get("fast_force_eos", True),
        )

        # A schedule, not a batch. It overrides only .size and
        # .make_microbatch_iterator -- exactly the two members the Automodel
        # microbatch iterator uses -- so one logical batch expands into
        # num_levels x microbatches forwards, all accumulating into ONE optimizer
        # step. This is where Automodel is easier than Megatron: its
        # forward/backward is a plain Python loop rather than an mcore schedule.
        schedule = BlockJustGRPORevealSchedule(base).configure(
            num_levels=num_levels,
            block_size=block_size,
            harvest_keys=("diffu_grpo_score_mask", "diffu_grpo_loss_mask"),
            reveal_tokens_per_level=reveal_k,
            selected_offsets=selected_offsets,
        )

        metadata: dict[str, Any] = {"num_levels": num_levels}
        if selected_offsets is not None:
            # Fast harvests only a subset of positions, so the loss must be
            # normalised by the KEPT count. Using the full response length would
            # silently scale the gradient with fast_entropy_level_ratio.
            metadata["fast_global_valid_toks"] = self._fast_global_valid_toks(
                base, selected_offsets, block_size
            )
            self._fast_valid_toks = metadata["fast_global_valid_toks"]
        return schedule, self.cfg, mbs, metadata

    def _fast_global_valid_toks(
        self,
        base: BatchedDataDict[Any],
        selected_offsets: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        """DP-reduced count of positions Fast actually harvests.

        The Megatron worker's single mcore call
        (``parallel_state.get_data_parallel_group()``) becomes ``self.dp_mesh``
        here -- the only backend-specific line in this class.
        """
        local_kept = build_block_kept_mask(base, selected_offsets, block_size).sum()
        to_reduce = local_kept.detach().to(dtype=torch.float32).reshape(1).cuda()
        torch.distributed.all_reduce(to_reduce, group=self._dp_group())
        return to_reduce[0]

    # ---- logprobs ----------------------------------------------------------

    def get_logprobs(
        self, data: BatchedDataDict[Any], micro_batch_size: Optional[int] = None
    ) -> BatchedDataDict[LogprobOutputSpec]:
        """One forward per reveal level; per-level logprobs are summed.

        Only ONE reveal level is resident at a time, which is what keeps long
        contexts tractable.
        """
        self._validate_diffusion_algorithm_support()
        cfg = self._block_reveal_cfg()
        block_size = self._block_reveal_block_size()
        reveal_k = self._block_reveal_tokens_per_level()
        reveal_mbs = micro_batch_size or self.cfg["logprob_batch_size"]

        base, num_samples, num_levels, selected_offsets = build_block_reveal_base(
            data,
            mask_token_id=cfg["mask_token_id"],
            pad_token_id=self.tokenizer.pad_token_id,
            block_size=block_size,
            pad_to_length=None,
            include_loss=False,
            max_reveal_levels=cfg.get("max_reveal_levels"),
            reveal_tokens_per_level=reveal_k,
            fast_entropy_level_ratio=cfg.get("fast_entropy_level_ratio"),
            eos_token_id=self.tokenizer.eos_token_id,
            force_eos=cfg.get("fast_force_eos", True),
        )

        if num_levels == 0:
            empty = torch.zeros_like(data["input_ids"], dtype=torch.float32)
            return BatchedDataDict[LogprobOutputSpec](logprobs=empty).to("cpu")

        # Per-level state. NOTE: this makes get_logprobs non-reentrant -- safe
        # only with one in-flight call per worker. The reference-logprob pass
        # re-enters this method under a weight swap, but sequentially.
        self._br_base = base
        self._br_selected_offsets = selected_offsets
        self._br_block_size_cur = block_size
        self._br_reveal_k = reveal_k
        self._br_reveal_mbs = reveal_mbs
        self._br_num_samples = num_samples
        self._br_original_seq_len = int(data["input_ids"].shape[1])

        accumulated: Optional[torch.Tensor] = None
        try:
            for level in range(num_levels):
                self._br_level = level
                level_out = super().get_logprobs(
                    data=data, micro_batch_size=reveal_mbs
                )["logprobs"]
                accumulated = (
                    level_out if accumulated is None else accumulated + level_out
                )
        finally:
            self._br_base = None

        return BatchedDataDict[LogprobOutputSpec](logprobs=accumulated).to("cpu")

    def _build_logprob_batch(
        self, data: BatchedDataDict[Any], micro_batch_size: Optional[int]
    ) -> tuple[Optional[BatchedDataDict[Any]], PolicyConfig, int, dict[str, Any]]:
        """Build the view for the reveal level the outer loop is currently on."""
        view = make_reveal_level_view(
            self._br_base,
            self._br_level,
            self._br_block_size_cur,
            ("diffu_grpo_score_mask",),
            self._br_reveal_k,
            selected_offsets=self._br_selected_offsets,
        )
        return (
            view,
            self.cfg,
            micro_batch_size or self._br_reveal_mbs,
            {
                "num_samples": self._br_num_samples,
                "original_seq_len": self._br_original_seq_len,
                "noisy_response_offset": int(
                    view["diffu_grpo_noisy_response_offsets"][0].item()
                ),
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
        flat = torch.cat([lp["logprobs"] for lp in list_of_logprobs], dim=0)
        return scatter_block_reveal_logprobs(
            flat_logprobs=flat,
            harvest_mask=transformed_data["block_reveal_harvest_mask"],
            sample_index=transformed_data["block_reveal_sample_index"],
            completion_starts=transformed_data["diffu_grpo_completion_starts"],
            noisy_response_offset=metadata["noisy_response_offset"],
            original_seq_len=metadata["original_seq_len"],
            num_samples=metadata["num_samples"],
        )


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker(
        "block_just_grpo_dtensor_policy_worker"
    )
)  # pragma: no cover
class BlockJustGRPODTensorPolicyWorker(BlockJustGRPODTensorPolicyWorkerImpl):
    pass
