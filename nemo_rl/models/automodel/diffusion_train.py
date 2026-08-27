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

"""Automodel (DTensor) post-processors for diffusion-LLM RL.

This is the Automodel counterpart of ``nemo_rl/models/megatron/diffu_grpo_train.py``.
It supplies the two post-processors a diffusion policy worker needs, differing from
the autoregressive ones in exactly one respect:

    An AR model predicts token ``t+1`` at position ``t``, so its logprobs are read
    with a one-position shift. A diffusion model predicts token ``t`` AT position
    ``t``, so the target is read at the SAME position -- no shift.

Both classes SUBCLASS the stock Automodel post-processors rather than replacing
them. That matters: ``forward_with_post_processing_fn`` dispatches with
``isinstance``, so a subclass matches the existing branch and needs no change to
the ``PostProcessingFunction`` union or the dispatch chain.
"""

from typing import Any, Optional

import torch

from nemo_rl.algorithms.logits_sampling_utils import TrainingSamplingParams
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.automodel.data import ProcessedInputs
from nemo_rl.models.automodel.train import (
    LogprobsPostProcessor,
    LossPostProcessor,
    apply_top_k_top_p_filtering_for_local_logits,
)


def same_position_logprobs(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    exclude_token_id: Optional[int] = None,
    temperature: Optional[float] = None,
    sampling_params: Optional[TrainingSamplingParams] = None,
) -> torch.Tensor:
    """Log-probability of ``target_ids[b, t]`` under ``logits[b, t]`` -- no shift.

    The autoregressive counterpart rolls the targets left by one and drops a column.
    Here position ``t`` scores the token that belongs at position ``t``, which is the
    whole point of a masked-diffusion objective.

    Args:
        logits: ``[B, S, V]`` model output.
        target_ids: ``[B, S]`` tokens to score, aligned to ``logits``.
        exclude_token_id: If given, that vocabulary entry is removed from the
            softmax before normalising. Used to drop the MASK token, which the
            SGLang FastDiffuser decoder also excludes from its x0 distribution --
            train and generate must agree or generation-KL blows up.
        temperature: If given, logits are divided by it first. Must match whatever
            was applied when ``prev_logprobs`` were computed.
        sampling_params: Optional top-k/top-p filtering parameters. Temperature in
            this object is intentionally ignored; callers scale exactly once before
            invoking this helper.

    Returns:
        ``[B, S]`` log probabilities.
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must be [B, S, V], got {tuple(logits.shape)}")
    if target_ids.shape != logits.shape[:2]:
        raise ValueError(
            f"target_ids {tuple(target_ids.shape)} must match logits prefix "
            f"{tuple(logits.shape[:2])}"
        )

    if temperature is not None:
        logits = logits / temperature

    if exclude_token_id is not None:
        # Mask in a copy; -inf drops the entry from the softmax denominator.
        logits = logits.clone()
        logits[..., exclude_token_id] = float("-inf")

    logits = apply_top_k_top_p_filtering_for_local_logits(logits, sampling_params)

    logprobs = torch.log_softmax(logits.to(torch.float32), dim=-1)
    return logprobs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)


class DiffusionLogprobsPostProcessor(LogprobsPostProcessor):
    """Same-position logprobs for a diffusion objective.

    Reads the tokens to score from ``data_dict['diffu_grpo_target_ids']`` when
    present -- a diffusion batch feeds MASKED tokens to the model but scores the
    CLEAN tokens underneath, so input and target genuinely differ. Falls back to
    ``input_ids`` otherwise.
    """

    def __init__(
        self, *args: Any, exclude_token_id: Optional[int] = None, **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self.exclude_token_id = exclude_token_id

    def __call__(
        self,
        logits: torch.Tensor,
        data_dict: BatchedDataDict[Any],
        processed_inputs: ProcessedInputs,
        original_batch_size: int,
        original_seq_len: int,
        sequence_dim: int = 1,
    ) -> torch.Tensor:
        if self.cp_size > 1:
            raise NotImplementedError(
                "Context parallelism is not supported for the diffusion logprob "
                "path. The default Automodel CP route requires is_causal=True and "
                "nulls the attention mask, which a block-diffusion objective cannot "
                "use. See the DSV4 manual-CP precedent for how a model-specific CP "
                "path would be added."
            )

        target_ids = data_dict.get("diffu_grpo_target_ids", None)
        if target_ids is None:
            target_ids = processed_inputs.input_ids

        # ``forward_with_post_processing_fn`` applies temperature scaling before
        # dispatching every LogprobsPostProcessor subclass. Applying it again here
        # would score at temperature squared when T != 1.
        return same_position_logprobs(
            logits,
            target_ids,
            exclude_token_id=self.exclude_token_id,
            sampling_params=self.sampling_params,
        )


class DiffusionLossPostProcessor(LossPostProcessor):
    """Clipped-PG loss over position-aligned (unshifted) diffusion logprobs.

    The stock LossPostProcessor hands raw logits to ``ClippedPGLossFn.__call__``,
    which internally slices five tensors by ``[:, 1:]`` to meet the next-token
    contract. That is wrong here: a diffusion model scores token ``t`` AT position
    ``t``, so nothing should be shifted.

    We therefore compute the logprobs ourselves and call the loss's
    ``compute_from_aligned_tensors`` entry point, which takes already-aligned
    tensors. That method lives on the SHARED loss class, so the Megatron and
    Automodel diffusion paths use exactly the same objective code.

    Note the denominator: ``global_valid_toks`` as computed by
    ``process_global_batch`` counts ``token_mask[:, 1:]`` -- an AR next-token
    count. A diffusion objective supervises a different set of positions, so
    callers pass the harvested-token count through ``metadata`` instead. If it is
    absent we fall back to the AR count and warn, rather than silently scaling the
    gradient wrong.
    """

    def __init__(
        self,
        *args: Any,
        exclude_token_id: Optional[int] = None,
        valid_toks_override: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.exclude_token_id = exclude_token_id
        self.valid_toks_override = valid_toks_override

    def __call__(
        self,
        logits: torch.Tensor,
        data_dict: BatchedDataDict[Any],
        processed_inputs: ProcessedInputs,
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
        sequence_dim: int = 1,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.cp_size > 1:
            raise NotImplementedError(
                "Context parallelism is not supported for the diffusion loss path."
            )

        target_ids = data_dict.get("diffu_grpo_target_ids", None)
        if target_ids is None:
            target_ids = processed_inputs.input_ids

        # Temperature was already applied by ``forward_with_post_processing_fn``.
        curr_logprobs = same_position_logprobs(
            logits,
            target_ids,
            exclude_token_id=self.exclude_token_id,
            sampling_params=self.sampling_params,
        )

        # Only the NOISY half is scored. The clean half exists solely to supply
        # previous-block context, so every per-token tensor is truncated to it --
        # mirroring nemo_rl/models/megatron/diffu_grpo_train.py so both backends
        # feed the shared loss identical inputs.
        noisy_length = int(data_dict["diffu_grpo_noisy_lengths"][0].item())

        def _clip(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return None if t is None else t[:, :noisy_length]

        curr_logprobs = curr_logprobs[:, :noisy_length]

        # Note this is diffu_grpo_loss_mask, NOT the generic token_mask: the set of
        # supervised positions is determined by the reveal schedule, not by which
        # tokens are non-padding.
        loss_mask = _clip(data_dict["diffu_grpo_loss_mask"])

        valid_toks = (
            self.valid_toks_override
            if self.valid_toks_override is not None
            else global_valid_toks
        )

        loss, metrics = self.loss_fn.compute_from_aligned_tensors(
            curr_logprobs=curr_logprobs,
            token_mask=loss_mask,
            sample_mask=data_dict["sample_mask"],
            advantages=_clip(data_dict["advantages"]),
            prev_logprobs=_clip(data_dict["prev_logprobs"]),
            generation_logprobs=_clip(data_dict["generation_logprobs"]),
            reference_policy_logprobs=_clip(data_dict.get("reference_policy_logprobs")),
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=valid_toks,
        )
        return loss, metrics
