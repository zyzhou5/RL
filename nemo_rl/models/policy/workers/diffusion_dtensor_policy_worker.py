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

"""Automodel (DTensor) base worker for diffusion-LLM RL.

Counterpart of ``diffusion_megatron_policy_worker.DiffusionMegatronPolicyWorkerImpl``,
re-parented onto ``DTensorPolicyWorkerV2Impl``.

The hook set is deliberately IDENTICAL in shape to the Megatron ABC's, because that
hook set is already validated: seven estimators (JustGRPO, DiffuGRPO, BlockJustGRPO,
CoupledGRPO, ESPO, TraceGRPO, hybrid AR+diffusion) sit on it without the base class
knowing about any of them. Names drop the ``megatron`` infix since they are no
longer backend-specific.

What differs from the Megatron ABC, and why:

* No pipeline-parallel handling. Automodel hardcodes ``pp_size=1``, so the
  last-stage broadcasts and cross-stage plumbing simply do not exist here. Net
  deletion.
* No ``rerun_state_machine``, no ``StragglerDetector``, no ``zero_grad_buffer``.
* ``stream_weights_via_http`` is NOT overridden. The Megatron ABC overrides it to
  call ``_iter_params_with_optional_kv_scales()``, which exists only on the
  Megatron worker -- inheriting that override here would AttributeError on the
  first refit. The DTensor base already implements refit correctly.
* Attention control is a hook rather than a duck-typed walk over ``model.modules()``.

Subclasses implement the six abstract hooks. Everything else is inherited.
"""

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any, Iterator, Optional
import warnings

import torch
from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm

from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.automodel.data import (
    ProcessedMicrobatch,
    check_sequence_dim,
    get_microbatch_iterator,
    process_global_batch,
)
from nemo_rl.models.automodel.diffusion_attention import (
    build_asymmetric_position_ids,
    build_asymmetric_semi_ar_block_mask,
    clear_hf_nld_asymmetric_mask,
    install_hf_nld_asymmetric_mask,
)
from nemo_rl.models.automodel.train import (
    aggregate_training_statistics,
    automodel_forward_backward,
    forward_with_post_processing_fn,
)
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import LogprobOutputSpec
from nemo_rl.models.policy.workers.dtensor_policy_worker_v2 import (
    DTensorPolicyWorkerV2Impl,
    get_train_context,
)
from nemo_rl.utils.nsys import wrap_with_nvtx_name


class DiffusionDTensorPolicyWorkerImpl(DTensorPolicyWorkerV2Impl, ABC):
    """Shared machinery for masked-diffusion RL estimators on the Automodel backend."""

    # ---- capability guards ------------------------------------------------

    def _validate_diffusion_support(self, algorithm_name: str) -> None:
        """Reject configurations this path cannot honour, loudly and early."""
        dtensor_cfg = self.cfg.get("dtensor_cfg", {})

        if dtensor_cfg.get("context_parallel_size", 1) != 1:
            raise NotImplementedError(
                f"{algorithm_name}: context_parallel_size > 1 is not supported on the "
                "Automodel backend. The default CP route requires is_causal=True and "
                "nulls the attention mask (automodel/data.py), which a block-diffusion "
                "objective cannot use. A model-specific manual-CP path would be needed "
                "-- see NRL_DSV4_MANUAL_CP for the precedent."
            )

        if dtensor_cfg.get("tensor_parallel_size", 1) != 1:
            raise NotImplementedError(
                f"{algorithm_name}: tensor_parallel_size > 1 is not supported yet. "
                "The diffusion post-processors currently require a full local "
                "vocabulary for same-position logprobs."
            )

        if self.cfg.get("dynamic_batching", {}).get("enabled", False):
            raise NotImplementedError(
                f"{algorithm_name}: dynamic batching is not supported. Reveal-level "
                "schedules currently implement the fixed-size microbatch iterator only."
            )

        self._validate_attention_paradigm()

        seq_packing = self.cfg.get("sequence_packing", {})
        if seq_packing.get("enabled", False):
            raise NotImplementedError(
                f"{algorithm_name}: sequence packing is not supported. Packing keeps "
                "documents apart via cu_seqlens precisely BECAUSE attention is causal; "
                "bidirectional attention inside a packed buffer leaks across document "
                "boundaries unless the mask is rebuilt block-diagonally."
            )

    @abstractmethod
    def _validate_diffusion_algorithm_support(self) -> None:
        """Estimator-specific config validation."""

    # ---- attention control ------------------------------------------------
    #
    # The Megatron ABC discovers attention modules by duck-typing over
    # model.modules() for set_inference_mode/set_inference_params/clear_kv_cache.
    # The HF modeling code exposes set_attention_mode(mode, block_size) on its
    # flex-attention class, so we drive that directly instead.

    #: dlm_paradigm values that cause the model to build flex attention with the
    #: asymmetric [noisy | clean] mask. Anything else gives plain attention.
    FLEX_PARADIGMS = ("block_diff", "sbd_block_diff")

    def _validate_attention_paradigm(self) -> None:
        """Fail if the model was built with plain attention.

        VERIFIED ON HARDWARE: MinistralDiffEncoderModel.__init__ selects its
        attention class from ``dlm_paradigm``. The shipped checkpoint declares
        ``autoregressive``, which yields plain Ministral3Attention with
        ``diffusion_lm=False`` -- i.e. ordinary causal attention, with NO error.
        Training would converge on the wrong objective.

        Selection therefore happens at config time:

            policy:
              hf_config_overrides:
                dlm_paradigm: sbd_block_diff

        This check exists because that failure is otherwise silent.
        """
        overrides = self.cfg.get("hf_config_overrides") or {}
        paradigm = overrides.get("dlm_paradigm") or getattr(
            getattr(self, "model_config", None), "dlm_paradigm", None
        )
        if paradigm not in self.FLEX_PARADIGMS:
            raise ValueError(
                f"dlm_paradigm={paradigm!r} builds plain (causal) attention. A "
                f"masked-diffusion objective needs one of {self.FLEX_PARADIGMS}. "
                "Set policy.hf_config_overrides.dlm_paradigm=sbd_block_diff. "
                "Without this the run trains successfully on the wrong objective."
            )
        if not self._diffusion_attention_modules():
            raise RuntimeError(
                f"dlm_paradigm={paradigm!r} was requested but no attention module "
                "exposes set_attention_mode(); flex attention was not constructed."
            )

    def _diffusion_attention_modules(self) -> list[Any]:
        return [m for m in self.model.modules() if hasattr(m, "set_attention_mode")]

    @contextmanager
    def _attention_mode(self, mode: Optional[str], block_size: Optional[int] = None):
        """Temporarily switch every attention module to ``mode``."""
        if mode is None:
            yield
            return
        modules = self._diffusion_attention_modules()
        if not modules:
            raise RuntimeError(
                "No attention module exposes set_attention_mode(). The diffusion "
                "path needs to control the attention regime; the loaded model does "
                "not appear to be a diffusion LM."
            )
        previous = [
            (m, getattr(m, "mode", None), getattr(m, "block_size", None))
            for m in modules
        ]
        try:
            for m in modules:
                m.set_attention_mode(mode, block_size=block_size)
            yield
        finally:
            for m, prev_mode, prev_bs in previous:
                if prev_mode is not None:
                    m.set_attention_mode(prev_mode, block_size=prev_bs)

    def _training_attention_context(self) -> AbstractContextManager[Any]:
        overrides = self.cfg.get("hf_config_overrides") or {}
        return self._attention_mode(
            overrides.get("dlm_paradigm"), block_size=self._diffusion_block_size()
        )

    def _logprob_attention_context(self) -> AbstractContextManager[Any]:
        overrides = self.cfg.get("hf_config_overrides") or {}
        return self._attention_mode(
            overrides.get("dlm_paradigm"), block_size=self._diffusion_block_size()
        )

    # ---- execution ---------------------------------------------------------

    @wrap_with_nvtx_name("diffusion_dtensor_policy_worker/train")
    def train(
        self,
        data: BatchedDataDict[Any],
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
    ) -> dict[str, Any]:
        """Train over an estimator-provided batch or reveal-level schedule."""
        self._validate_diffusion_algorithm_support()
        if gbs is None:
            gbs = self.cfg["train_global_batch_size"]
        if mbs is None:
            mbs = self.cfg["train_micro_batch_size"]

        local_gbs = gbs // self.dp_size
        total_dataset_size = torch.tensor(data.size, device="cuda")
        torch.distributed.all_reduce(
            total_dataset_size,
            op=torch.distributed.ReduceOp.SUM,
            group=self._dp_group(),
        )
        num_global_batches = int(total_dataset_size.item()) // gbs

        if eval_mode:
            ctx: AbstractContextManager[Any] = torch.no_grad()
            self.model.eval()
        else:
            ctx = nullcontext()
            self.model.train()

        empty_cache_steps = self.cfg.get("dtensor_cfg", {}).get(
            "clear_cache_every_n_steps"
        )
        if empty_cache_steps:
            warnings.warn(
                f"Emptying cache every {empty_cache_steps} microbatches; doing so "
                "unnecessarily incurs a large performance overhead.",
                stacklevel=2,
            )

        def on_microbatch_start(microbatch_idx: int) -> None:
            if empty_cache_steps and microbatch_idx % empty_cache_steps == 0:
                torch.cuda.empty_cache()

        data = data.to("cuda")
        losses: list[float] = []
        all_mb_metrics: list[dict[str, Any]] = []
        grad_norm: Optional[torch.Tensor] = None

        with ctx, self._training_attention_context():
            for global_batch_idx in range(num_global_batches):
                global_batch = process_global_batch(
                    data,
                    loss_fn,
                    self._dp_group(),
                    batch_idx=global_batch_idx,
                    batch_size=local_gbs,
                )
                (
                    transformed_data,
                    cfg_for_training,
                    train_mbs,
                    metadata,
                ) = self._build_training_batch(global_batch["batch"], mbs)
                sequence_dim, _ = check_sequence_dim(transformed_data)

                num_levels = int(metadata.get("num_levels", 1))
                samples_per_level = transformed_data.size // max(1, num_levels)
                if samples_per_level % train_mbs != 0:
                    raise ValueError(
                        "The per-rank sample count for each diffusion reveal level "
                        f"({samples_per_level}) must be divisible by the training "
                        f"microbatch size ({train_mbs})."
                    )

                global_valid_seqs = global_batch["global_valid_seqs"]
                global_valid_toks = metadata.get(
                    "fast_global_valid_toks", global_batch["global_valid_toks"]
                )

                processed_iterator, iterator_len = get_microbatch_iterator(
                    transformed_data,
                    cfg_for_training,
                    train_mbs,
                    self.dp_mesh,
                    tokenizer=self.tokenizer,
                    cp_size=self.cp_size,
                )
                processed_iterator = self._wrap_training_microbatch_iterator(
                    processed_iterator, metadata
                )
                loss_post_processor = self._make_loss_post_processor(
                    loss_fn=loss_fn,
                    cfg=cfg_for_training,
                    device_mesh=self.device_mesh,
                    cp_mesh=self.cp_mesh,
                    tp_mesh=self.tp_mesh,
                    cp_size=self.cp_size,
                    dp_size=self.dp_size,
                    enable_seq_packing=self.enable_seq_packing,
                    sampling_params=self.sampling_params,
                )

                def train_context_fn(processed_inputs: Any):
                    return get_train_context(
                        cp_size=self.cp_size,
                        cp_mesh=self.cp_mesh,
                        cp_buffers=processed_inputs.cp_buffers,
                        sequence_dim=sequence_dim,
                        dtype=self.dtype,
                        autocast_enabled=self.autocast_enabled,
                    )

                self.optimizer.zero_grad()
                microbatch_results = automodel_forward_backward(
                    model=self.model,
                    data_iterator=processed_iterator,
                    post_processing_fn=loss_post_processor,
                    forward_only=eval_mode,
                    is_reward_model=False,
                    allow_flash_attn_args=self.allow_flash_attn_args,
                    global_valid_seqs=global_valid_seqs,
                    global_valid_toks=global_valid_toks,
                    sampling_params=self.sampling_params,
                    sequence_dim=sequence_dim,
                    dp_size=self.dp_size,
                    cp_size=self.cp_size,
                    num_global_batches=num_global_batches,
                    train_context_fn=train_context_fn,
                    num_valid_microbatches=iterator_len,
                    on_microbatch_start=on_microbatch_start,
                )

                microbatch_losses = []
                for microbatch_idx, (loss, loss_metrics) in enumerate(
                    microbatch_results
                ):
                    if microbatch_idx >= iterator_len:
                        continue
                    loss_metrics["lr"] = self.optimizer.param_groups[0]["lr"]
                    loss_metrics["global_valid_seqs"] = global_valid_seqs.item()
                    loss_metrics["global_valid_toks"] = global_valid_toks.item()
                    if loss_metrics["num_valid_samples"] > 0:
                        microbatch_losses.append(loss.item())
                        all_mb_metrics.append(loss_metrics)

                if not eval_mode:
                    raw_grad_norm = scale_grads_and_clip_grad_norm(
                        self.max_grad_norm,
                        [self.model],
                        norm_type=2.0,
                        pp_enabled=False,
                        device_mesh=self.device_mesh,
                        moe_mesh=self.moe_mesh,
                        ep_axis_name=(
                            "ep"
                            if self.moe_mesh is not None
                            and "ep" in self.moe_mesh.mesh_dim_names
                            else None
                        ),
                        pp_axis_name=None,
                        foreach=True,
                        num_label_tokens=1,
                        dp_group_size=self.dp_size * self.cp_size,
                    )
                    grad_norm = torch.tensor(
                        raw_grad_norm, device="cpu", dtype=torch.float32
                    )
                    self.optimizer.step()

                losses.append(torch.tensor(microbatch_losses).sum().item())

        self.optimizer.zero_grad()
        if not eval_mode:
            self.scheduler.step()
        torch.cuda.empty_cache()
        return aggregate_training_statistics(
            losses=losses,
            all_mb_metrics=all_mb_metrics,
            grad_norm=grad_norm,
            dp_group=self._dp_group(),
            dtype=self.dtype,
        )

    @wrap_with_nvtx_name("diffusion_dtensor_policy_worker/get_logprobs")
    def get_logprobs(
        self,
        data: BatchedDataDict[Any],
        micro_batch_size: Optional[int] = None,
    ) -> BatchedDataDict[LogprobOutputSpec]:
        """Run estimator-specific diffusion forwards and restore ``[B, S]``."""
        self._validate_diffusion_algorithm_support()
        (
            transformed_data,
            cfg_for_logprobs,
            logprob_mbs,
            metadata,
        ) = self._build_logprob_batch(data, micro_batch_size)
        if transformed_data is None:
            return BatchedDataDict[LogprobOutputSpec](
                logprobs=metadata["empty_logprobs"]
            ).to("cpu")

        sequence_dim, _ = check_sequence_dim(transformed_data)
        transformed_data = transformed_data.to("cuda")
        self.model.eval()
        processed_iterator, iterator_len = get_microbatch_iterator(
            transformed_data,
            cfg_for_logprobs,
            logprob_mbs,
            self.dp_mesh,
            tokenizer=self.tokenizer,
            cp_size=self.cp_size,
        )
        processed_iterator = self._wrap_logprob_microbatch_iterator(
            processed_iterator, metadata
        )
        logprobs_post_processor = self._make_logprobs_post_processor(
            cfg=cfg_for_logprobs,
            device_mesh=self.device_mesh,
            cp_mesh=self.cp_mesh,
            tp_mesh=self.tp_mesh,
            cp_size=self.cp_size,
            enable_seq_packing=self.enable_seq_packing,
            sampling_params=self.sampling_params,
        )

        list_of_logprobs: list[dict[str, torch.Tensor]] = []
        with torch.no_grad(), self._logprob_attention_context():
            for microbatch_idx, processed_microbatch in enumerate(processed_iterator):
                processed_inputs = processed_microbatch.processed_inputs
                with get_train_context(
                    cp_size=self.cp_size,
                    cp_mesh=self.cp_mesh,
                    cp_buffers=processed_inputs.cp_buffers,
                    sequence_dim=sequence_dim,
                    dtype=self.dtype,
                    autocast_enabled=self.autocast_enabled,
                ):
                    token_logprobs, _metrics, _ = forward_with_post_processing_fn(
                        model=self.model,
                        post_processing_fn=logprobs_post_processor,
                        processed_mb=processed_microbatch,
                        is_reward_model=False,
                        allow_flash_attn_args=self.allow_flash_attn_args,
                        sampling_params=self.sampling_params,
                        sequence_dim=sequence_dim,
                    )
                if microbatch_idx < iterator_len:
                    list_of_logprobs.append({"logprobs": token_logprobs})

        logprobs = self._finalize_logprobs_from_outputs(
            list_of_logprobs,
            original_data=data,
            transformed_data=transformed_data,
            metadata=metadata,
        )
        return BatchedDataDict[LogprobOutputSpec](logprobs=logprobs).to("cpu")

    # ---- batch construction ------------------------------------------------

    @abstractmethod
    def _build_training_batch(
        self, data: BatchedDataDict[Any], mbs: int
    ) -> tuple[BatchedDataDict[Any], PolicyConfig, int, dict[str, Any]]:
        """Return ``(batch_or_schedule, cfg, microbatch_size, metadata)``.

        Returning a SCHEDULE rather than a plain batch is what lets one logical
        batch expand into N reveal-level forward passes: the schedule overrides
        only ``.size`` and ``.make_microbatch_iterator``, which are exactly the two
        members the Automodel microbatch iterator uses.
        """

    @abstractmethod
    def _build_logprob_batch(
        self, data: BatchedDataDict[Any], micro_batch_size: Optional[int]
    ) -> tuple[Optional[BatchedDataDict[Any]], PolicyConfig, int, dict[str, Any]]:
        """Same, for the logprob path."""

    # ---- post-processors ---------------------------------------------------

    @abstractmethod
    def _make_logprobs_post_processor(self, **kwargs: Any) -> Any:
        """Return the LogprobsPostProcessor subclass this estimator needs."""

    @abstractmethod
    def _make_loss_post_processor(self, **kwargs: Any) -> Any:
        """Return the LossPostProcessor subclass this estimator needs."""

    @abstractmethod
    def _finalize_logprobs_from_outputs(
        self,
        list_of_logprobs: list[dict[str, torch.Tensor]],
        *,
        original_data: BatchedDataDict[Any],
        transformed_data: BatchedDataDict[Any],
        metadata: dict[str, Any],
    ) -> torch.Tensor:
        """Scatter per-microbatch outputs back into one ``[B, S]`` tensor."""

    # ---- iterator wrapping -------------------------------------------------

    def _set_asymmetric_ar_metadata(self, microbatch: ProcessedMicrobatch) -> None:
        """Configure attention and RoPE positions for one replay microbatch."""
        data_dict = microbatch.data_dict
        if "diffu_grpo_noisy_lengths" not in data_dict:
            return

        noisy_lengths = data_dict["diffu_grpo_noisy_lengths"]
        noisy_valid_lengths = data_dict["diffu_grpo_noisy_valid_lengths"]
        clean_padded_lengths = data_dict["diffu_grpo_clean_padded_lengths"]
        noisy_response_offsets = data_dict["diffu_grpo_noisy_response_offsets"]
        if noisy_lengths.numel() == 0:
            return
        if not torch.all(noisy_lengths == noisy_lengths[0]):
            raise ValueError("Diffusion noisy length must be constant per microbatch")
        if not torch.all(clean_padded_lengths == clean_padded_lengths[0]):
            raise ValueError("Diffusion clean length must be constant per microbatch")
        if not torch.all(noisy_response_offsets == noisy_response_offsets[0]):
            raise ValueError(
                "Diffusion noisy response offset must be constant per microbatch"
            )

        noisy_length = int(noisy_lengths[0].item())
        clean_length = int(clean_padded_lengths[0].item())
        noisy_response_offset = int(noisy_response_offsets[0].item())
        prompt_lengths = data_dict["diffu_grpo_completion_starts"]
        response_lengths = data_dict["diffu_grpo_response_lengths"]
        clean_lengths = data_dict["diffu_grpo_clean_lengths"]

        processed_inputs = microbatch.processed_inputs
        processed_inputs.position_ids = build_asymmetric_position_ids(
            noisy_length=noisy_length,
            clean_length=clean_length,
            noisy_response_offset=noisy_response_offset,
            prompt_lengths=prompt_lengths,
            noisy_valid_lengths=noisy_valid_lengths,
        )
        processed_inputs.model_kwargs["block_size"] = self._diffusion_block_size()

        modules = self._diffusion_attention_modules()
        fallback_modules = [
            module
            for module in modules
            if not hasattr(module, "set_asymmetric_ar_metadata")
        ]
        unknown_modules = [
            module
            for module in fallback_modules
            if module.__class__.__name__ != "MinistralFlexAttention"
        ]
        if unknown_modules:
            names = sorted({module.__class__.__name__ for module in unknown_modules})
            raise RuntimeError(
                "Diffusion completion-only replay requires attention modules with "
                "set_asymmetric_ar_metadata(), or the checked HF NLD compatibility "
                f"adapter; unsupported modules: {names}"
            )

        block_mask = None
        if fallback_modules:
            block_mask = build_asymmetric_semi_ar_block_mask(
                block_size=self._diffusion_block_size(),
                noisy_length=noisy_length,
                clean_length=clean_length,
                noisy_response_offset=noisy_response_offset,
                prompt_lengths=prompt_lengths,
                noisy_valid_lengths=noisy_valid_lengths,
                clean_lengths=clean_lengths,
            )

        for module in modules:
            if hasattr(module, "set_asymmetric_ar_metadata"):
                module.set_asymmetric_ar_metadata(
                    noisy_length=noisy_length,
                    clean_length=clean_length,
                    noisy_response_offset=noisy_response_offset,
                    prompt_lengths=prompt_lengths,
                    response_lengths=response_lengths,
                    noisy_valid_lengths=noisy_valid_lengths,
                    clean_lengths=clean_lengths,
                )
            elif not install_hf_nld_asymmetric_mask(module, block_mask=block_mask):
                raise RuntimeError(
                    f"Unable to install asymmetric attention on {type(module).__name__}"
                )

    def _clear_asymmetric_ar_metadata(self) -> None:
        for module in self._diffusion_attention_modules():
            if hasattr(module, "clear_asymmetric_ar_metadata"):
                module.clear_asymmetric_ar_metadata()
            else:
                clear_hf_nld_asymmetric_mask(module)

    def _wrap_diffusion_microbatch_iterator(
        self, iterator: Iterator[ProcessedMicrobatch]
    ) -> Iterator[ProcessedMicrobatch]:
        try:
            for microbatch in iterator:
                self._set_asymmetric_ar_metadata(microbatch)
                yield microbatch
        finally:
            self._clear_asymmetric_ar_metadata()

    def _wrap_training_microbatch_iterator(
        self, iterator: Iterator[ProcessedMicrobatch], metadata: dict[str, Any]
    ) -> Iterator[ProcessedMicrobatch]:
        del metadata
        return self._wrap_diffusion_microbatch_iterator(iterator)

    def _wrap_logprob_microbatch_iterator(
        self, iterator: Iterator[ProcessedMicrobatch], metadata: dict[str, Any]
    ) -> Iterator[ProcessedMicrobatch]:
        del metadata
        return self._wrap_diffusion_microbatch_iterator(iterator)

    # ---- helpers -----------------------------------------------------------

    def _diffusion_block_size(self) -> int:
        """Block size, taken from the model config so it cannot drift from the model."""
        overrides = self.cfg.get("hf_config_overrides") or {}
        if "block_size" in overrides:
            return int(overrides["block_size"])
        block_size = getattr(getattr(self, "model_config", None), "block_size", None)
        if block_size is None:
            raise ValueError(
                "block_size is not resolvable from hf_config_overrides or the model "
                "config. It must agree with the decode config or generation-KL will "
                "silently blow up (the fork records 1.41 from a 32-vs-16 mismatch)."
            )
        return int(block_size)

    def _dp_group(self) -> Any:
        """DP process group -- the Automodel equivalent of parallel_state's."""
        return self.dp_mesh.get_group()
