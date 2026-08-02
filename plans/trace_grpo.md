# Inference-Trajectory-Replay GRPO ("Trace" / TraceGRPO)

Worktree: `RL-Trace` (branch `Trace`, based on `dllm_clean` @ c925cf16)

## 1. Goal

Train a diffusion LLM with GRPO where the per-token training objective replays the
*actual* inference denoising trajectory. During rollout, SGLang FastDiffuser with
`selection_policy=confidence, threshold=0.9` reveals, at each denoising step, the set
of masked positions whose top-1 confidence >= 0.9. We record which step committed
each token, and during training we score each token on exactly the context the
inference engine conditioned on when it committed that token.

This is the same multi-level reveal machinery as block-JustGRPO
(`nemo_rl/algorithms/block_just_grpo_logprobs.py`), with one change: the reveal order
per level is taken from the inference trajectory instead of the deterministic
leftmost-within-block schedule.

## 2. Two properties that motivate this baseline

- Exact multi-token steps. In block-JustGRPO, k>1 is an approximation: the k tokens
  harvested together were not really generated in parallel (see that module's
  docstring, lines 33-40). Here, when a confidence step commits several tokens at
  once, they genuinely were produced in one parallel forward from the same context,
  so harvesting them together in one training forward is exact.
- Forward-pass count = inference NFE. Exact replay needs one training forward per
  distinct reveal event (denoising step). That equals the number of forward passes
  generation itself used. With confidence-0.9 decoding, NFE << sequence length, so
  cost is comparable to block-JustGRPO with a small k -- not the per-token k=1 cost.

## 3. Core encoding: within-block denoising step, blocks in parallel

Each response token's `reveal_level` is its within-block denoising step (NOT a
cross-block ordering). Blocks are processed in parallel; the only loop is over the
denoising steps.

  1. From SGLang we get `commit_step` per generated token: the block-relative
     denoising step that committed it (0..max_steps, where max_steps is the
     force-commit sentinel FastDiffuser already assigns). It is already
     block-relative, so it IS the within-block step.
  2. Per sample, dense-rank the distinct commit_steps to 0,1,2,...  Each token's
     `reveal_level` = the dense rank of its commit_step. Dense ranking drops the
     force-commit gap, so num_levels = max denoising steps, not max_steps+1.

Why no block index / no block loop: previous-block context is supplied
unconditionally by the clean side of the asymmetric semi-AR attention mask
(`compute_asymmetric_semi_ar_mask`: a block-b query attends to ALL clean tokens of
blocks < b). So when level i harvests block b's step-i token, blocks < b are already
fully visible regardless of i -- blocks need no sequencing. This is exactly
block-JustGRPO's structure (all blocks advance together per level), with the reveal
driven by the inference trajectory instead of the leftmost schedule.

Cost: forward passes = max denoising steps over blocks (e.g. ~10), NOT sum over
blocks. An earlier draft globally ranked (block, step) and serialized the blocks
(sum-over-blocks forwards, NFE); that was corrected to this block-parallel form.

### Level view (mirrors make_reveal_level_view in block_just_grpo_logprobs.py)

For level L:
- reveal  = in_response AND (reveal_level <  L)   -> put real target token
- harvest = in_response AND (reveal_level == L) AND score_mask   -> score this token

Everything else (fully-masked base construction, asymmetric [noisy|clean] layout,
scatter back to [N,S], the per-level worker loop, the training reveal schedule) is
reused from block_just_grpo_logprobs.py essentially unchanged. The only real change
is this predicate and how num_levels is computed.

## 4. Variable trajectory length and deadlock safety

Different samples take different numbers of denoising steps -> different level counts.
This is handled the same way block-JustGRPO handles it, with the same hard constraint.

Rule: deadlock occurs iff two DP ranks execute a different number of collective ops.
Each reveal level is one forward (+ grad reduce-scatter in training) over all rows =
a collective. So:

- (a) num_levels must be a single globally-agreed scalar. Every rank loops the
  identical number of times. A sample that finished early still passes through every
  remaining level with a zero harvest/loss mask -- zero gradient, but the collective
  still fires. This is the intended "extra forward pass with no loss addition".
- (b) Within each level, every rank must run the same number of microbatches ->
  even DP shards + fixed microbatch size. Reuse `pad_reveal_batch_to_multiple`
  (`just_grpo_logprobs.py:277`) and even sharding, exactly as block-JustGRPO does.

NEVER skip a level on a rank because its local samples all finished -- that is the
only way to reintroduce the NCCL-timeout crash class. All ranks run all levels.

Computing num_levels: it is data-dependent (unlike block-JustGRPO's pure-config
count_reveal_levels), so compute it with ONE all_reduce(MAX) of the per-rank maximum
per-sample dense-rank count, before the loop, optionally clamped by a
`max_reveal_levels` config cap. The all-reduce is an unconditional collective every
rank calls, so all ranks agree, then run the same number of forwards. Safe.

Decisions locked here (per user):
- No merging of consecutive steps (each distinct step is its own level).
- No compaction of completed samples (they ride along zero-harvest).
- No per-rank level skipping.

## 5. SGLang change: dedicated reveal-step field

Decision: add a dedicated per-token output field `output_token_reveal_steps` rather
than overloading the existing logprob/idx channels. More sites, but keeps semantics
clean and preserves full float32 precision on the logprobs (needed for the parity
check in section 7).

FastDiffuser already computes the trajectory: `commit_step_grid[orig_b, top_idx] = step`
(fastdiffuser.py ~607), with force-commit set to max_steps (~684). Today it is only
kept for `logprob_mode="final_step"` and collapsed to a 1-bit sign sentinel. We add a
`logprob_mode="trajectory"` that keeps the full per-token reveal-step logprobs (no
sentinel overwrite) and emits `commit_step` per token.

Producer -> HTTP plumbing (SGLang fork /home/snorouzi/code/sglang-nemotron-dllm-a652eb48):

1. layers/logits_processor.py:67 -- LogitsProcessorOutput: add
   `next_token_reveal_steps: Optional[List] = None`.
2. srt/dllm/algorithm/fastdiffuser.py -- trajectory mode: build the per-token step
   list from commit_step_grid unconditionally; set logits_output.next_token_reveal_steps;
   do NOT apply the SPG +1.0 sentinel so next_token_logprobs keeps real reveal-step
   logprobs.
3. srt/managers/schedule_batch.py:775 -- Req.__init__: self.output_token_reveal_steps
   = [] in the return_logprob branch, None in the else branch.
4. srt/dllm/mixin/scheduler.py:180 -- alongside the _val/_idx extends,
   req.output_token_reveal_steps.extend(...) when the algorithm provided them.
5. srt/managers/io_struct.py -- add field to BatchTokenIDOut (~1086) and BatchStrOut (~1148).
6. srt/managers/scheduler_output_processor_mixin.py -- BatchTokenIDOut assembly:
   init-empty block (~978), per-req append (~1107), and the constructor call.
7. srt/managers/detokenizer_manager.py:345 -- pass output_token_reveal_steps from
   recv_obj to BatchStrOut (no detokenization needed; raw ints).
8. srt/managers/tokenizer_manager.py -- ReqState field (~178); chunk-accumulation
   extend (~1975); expose meta_info["output_token_reveal_steps"] =
   state.output_token_reveal_steps (~1893). Simpler than logprobs: no
   detokenize_logprob_tokens step, expose the int list directly.

Conditional / verify at build time (likely skip):
- srt/managers/multi_tokenizer_mixin.py:160,236 -- only if the multi-tokenizer
  router is in use.
- srt/disaggregation/decode.py, utils.py -- only for PD-disaggregated serving; the
  colocated RL setup does not use it.

Gate: emit only when return_logprob is on AND logprob_mode="trajectory" (algorithm
set the field). Existing AR / fully_masked / final_step runs are unaffected.

## 6. NeMo-RL changes

Generation side:
9.  nemo_rl/models/generation/sglang/sglang_worker.py --
    _extract_generated_tokens_and_logprobs (~72) reads meta_info["output_token_reveal_steps"];
    generate() (~692) assembles a reveal_steps [B,S] tensor parallel to logprobs
    (zeros for prompt, block-relative step for response tokens).
10. GenerationOutputSpec -- add a reveal_steps key.
11. nemo_rl/algorithms/grpo.py -- carry reveal_steps into the training data dict
    (same flow that carries generation_logprobs).

Training side (new module, mirrors block_just_grpo_logprobs.py):
12. nemo_rl/algorithms/trace_grpo_logprobs.py -- new module.
    - get_trace_grpo_logprob_estimation_cfg (type "trace_grpo").
    - build the fully-masked base via diffu_grpo's build_fully_masked_completion_batch
      (reuse build_block_reveal_base pattern).
    - compute per-sample dense reveal_level from scattered commit_step + block_index.
    - count_levels: all_reduce(MAX) of per-sample distinct-event count, clamped by
      max_reveal_levels.
    - make_reveal_level_view: reveal = reveal_level < L; harvest = reveal_level == L
      AND score_mask.
    - reuse scatter_block_reveal_logprobs and the RevealSchedule pattern.
13. nemo_rl/models/policy/workers/trace_grpo_megatron_policy_worker.py --
    subclass the block-JustGRPO worker (block_just_grpo_megatron_policy_worker.py,
    ~218 lines); override only the level-count call and make_reveal_level_view usage.
14. Register in nemo_rl/distributed/ray_actor_environment_registry.py and
    nemo_rl/models/policy/__init__.py.
15. Config: new logprob_estimation.type "trace_grpo"; mirror
    examples/configs/deepscaler_nemotron_labs_diffusion_3b_sglang_block_just_grpo_k2_megatron.yaml.
    CRITICAL: training block_size and max_steps must match the rollout
    dllm_algorithm_config (tools/nemotron_diffusion/diffugrpo_fastdiffuser.yaml:
    block_size 32, max_steps 32, threshold 0.9), or reveal_level arithmetic will not
    line up with the recorded commit_step. Set logprob_mode: trajectory in that dllm
    config.

## 7. Validation

Free oracle: the training-replayed logprob for token t at level reveal_level(t)
conditions on exactly the context FastDiffuser used to commit t, so it must equal the
reveal-step generation logprob SGLang returns (next_token_logprobs in trajectory mode,
used as generation_logprobs). Add an offline parity probe mirroring
tools/nemotron_diffusion/offline_block_just_grpo_parity_probe.py, asserting agreement
to ~1e-3 in fp32. This is both the correctness gate and the on-policy check.
(block-reveal and leftmost were validated to ~2e-3 fp32 this way.)

generation_logprobs = SGLang reveal-step logprobs (decision confirmed): used for the
GRPO importance ratio and directly compared against the Megatron replay.

## 8. Sequencing

1. SGLang: logprob_mode="trajectory" in FastDiffuser (emit commit_step, drop sentinel)
   + the reveal-step field plumbing (section 5). Verify multi_tokenizer need.
2. NeMo-RL worker + GenerationOutputSpec + grpo rollout plumbing (9-11).
3. trace_grpo_logprobs.py + worker + registry + config (12-15).
4. Offline parity probe -> fp32 toy run -> scale.

## 9. Open items to confirm during implementation

- multi_tokenizer_mixin usage (whether site 8b applies).
- Block-parallel leveling (blocks advance together; level = within-block denoising
  step) relies on the clean-side previous-block visibility in
  compute_asymmetric_semi_ar_mask; the parity probe must confirm it reproduces the
  SGLang reveal-step logprobs. No block_index is used in compute_reveal_levels.
- max_reveal_levels default/cap value and truncation logging if it ever triggers.
