# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Any, Literal, NotRequired, TypedDict

from nemo_rl.models.generation.interfaces import GenerationConfig


class VllmSpecificArgs(TypedDict):
    tensor_parallel_size: int
    pipeline_parallel_size: int
    expert_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int
    # Additional arguments for vLLM inserted by nemo rl based on the context of when vllm is used
    skip_tokenizer_init: bool
    async_engine: bool
    load_format: NotRequired[str]
    precision: NotRequired[str]
    kv_cache_dtype: Literal["auto", "fp8", "fp8_e4m3"]
    enforce_eager: NotRequired[bool]
    # By default, NeMo RL only has a Python handle to the vllm.LLM generation engine. The expose_http_server flag here will expose that generation engine as an HTTP server.
    # Exposing vLLM as a server is useful in instances where the multi-turn rollout is performed with utilities outside of NeMo RL, but the user still wants to take advantage of the refit logic in NeMo RL that keeps the policy and generation up to date.
    # Currently it will expose the /tokenize and /v1/chat/completions endpoints. Later on we may expose /v1/completions or /v1/responses.
    expose_http_server: NotRequired[bool]
    # These kwargs are passed to the vllm.LLM HTTP server Chat Completions endpoint config. Typically this will include things like tool parser, chat template, etc
    http_server_serving_chat_kwargs: NotRequired[dict[str, Any]]
    # Miscellaneous top level vLLM HTTP server arguments.
    # A filepath that can be imported to register a vLLM tool parser
    tool_parser_plugin: NotRequired[str]


class VllmConfig(GenerationConfig):
    vllm_cfg: VllmSpecificArgs
    vllm_kwargs: NotRequired[dict[str, Any]]
    # Optional overrides describing how VALIDATION decoding differs from
    # rollout decoding. Must contain a "vllm_cfg" key: unlike SGLang, vLLM has
    # no runtime decode-reconfigure path, since the diffusion knobs are copied
    # into the sampler when the model loads. The dict is deep-merged on top of
    # a copy of this generation config and a SECOND vLLM engine group is
    # launched from it, used only for validation, e.g. AR rollouts with
    # diffusion validation:
    #   vllm_val_dllm_overrides:
    #     temperature: 1.0
    #     vllm_cfg:
    #       gpu_memory_utilization: 0.7
    #     vllm_kwargs:
    #       diffusion_config:
    #         canvas_length: 32
    #         selection_policy: "confidence_threshold"
    #         confidence_threshold: 0.9
    #         temperature: 1.0
    # Note that a diffusion validation group needs `temperature` and
    # `vllm_kwargs.diffusion_config.temperature` to agree numerically: the
    # engine samples at the diffusion_config value while the trainer tempers by
    # generation.temperature, and the worker rejects a mismatch (see
    # vllm_worker.py `_build_sampling_params`).
    #
    # Requires colocated inference and synchronous GRPO. The two groups
    # time-share GPU memory -- vLLM engines are always built with
    # enable_sleep_mode, and the validation group sleeps except around
    # validation -- so vllm_cfg.gpu_memory_utilization must be stated
    # explicitly here rather than inherited from the rollout group.
    #
    # Absent/None => validation uses the same decoding as rollout.
    vllm_val_dllm_overrides: NotRequired[dict[str, Any] | None]

    # Named decode variants to validate under, reported side by side as
    # "val:accuracy/<name>" / "val:avg_length/<name>". Each entry holds that
    # variant's soft decode knobs (same vocabulary as mode 1 above).
    #
    # Every variant sweeps the SAME prompts with the same budget
    # (grpo.max_val_samples // grpo.val_batch_size batches), so their
    # accuracies are directly comparable. The FIRST entry is the primary: it
    # also fills the unsuffixed val:accuracy (which drives checkpoint
    # selection) and owns the printed samples and val_data jsonl.
    #
    # Motivation: on a diffusion LLM the sampler is part of the policy, so a
    # single val curve cannot separate "the weights degraded" from "the
    # sampler stopped working on these weights". Validating the rollout
    # decode alongside a sequential (one-token-per-step) reference answers
    # that -- if the sequential curve holds while the parallel one falls,
    # the damage is in the sampler, not the model:
    #      vllm_val_dllm_variants:
    #        rollout: {...the rollout decode...}
    #        seq_k1:
    #          vllm_kwargs:
    #            diffusion_config:
    #              selection_policy: low_confidence
    #              max_denoising_steps: 16   # == canvas_length, since
    #                # tokens-per-forward is canvas_length/max_denoising_steps
    #
    # Variants may not contain the engine-config key (they are applied via
    # reconfigure_dllm, which cannot change engine-launch settings); use
    # vllm_val_dllm_overrides for that. When absent, validation falls back
    # to the single decode from vllm_val_dllm_overrides.
    vllm_val_dllm_variants: NotRequired[dict[str, Any] | None]
