#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Offline TraceGRPO logprob-parity forensics from logged train_data_step*.jsonl.

Compares generation_logprobs (SGLang reveal-step, recorded at each token's
commit) against prev_logprobs (Megatron trajectory replay) over the response,
bucketed by position: first block, middle, final block, and the EOS token
itself. A systematic signed delta on the EOS bucket measures the
tail-conditioning mismatch (see eos_tail_fill in the trace config); the
no-padding baseline measured -0.4 nats on GSM8K (healthy phase).

Usage: analyze_trace_eos_parity.py <exp_log_dir> [step ...]
"""
import json
import statistics
import sys


EOS_ID = 11  # Nemotron <|im_end|>


def analyze(path: str, step: int, max_recs: int = 400) -> None:
    """Buckets are relative to the FIRST EOS in the response (emit_full_blocks
    responses carry a context-only tail after it whose prev is 0 by design --
    those positions are skipped). first_eos is THE metric: the termination
    decision's replay parity."""
    buckets: dict[str, list[float]] = {
        "first_blk": [], "middle": [], "pre_eos_lastblk": [], "first_eos": []
    }
    tail_zero: list[float] = []
    n = 0
    with open(f"{path}/train_data_step{step}.jsonl") as fh:
        for line in fh:
            if n >= max_recs:
                break
            r = json.loads(line)
            mask = r["token_loss_mask"][0]
            gen = r["generation_logprobs"][0]
            prev = r["prev_logprobs"][0]
            ids = r["token_ids"][0]
            pos = [i for i, m in enumerate(mask) if m == 1]
            if len(pos) < 20:
                continue
            n += 1
            eidx = next((j for j, i in enumerate(pos) if ids[i] == EOS_ID), len(pos) - 1)
            blk_start = (eidx // 16) * 16
            tail = pos[eidx + 1:]
            if tail:
                tail_zero.append(sum(1 for i in tail if prev[i] == 0.0) / len(tail))
            for j in range(eidx + 1):
                i = pos[j]
                d = prev[i] - gen[i]
                if j == eidx:
                    b = "first_eos"
                elif j >= blk_start:
                    b = "pre_eos_lastblk"
                elif j < 16:
                    b = "first_blk"
                else:
                    b = "middle"
                buckets[b].append(d)
    print(f"--- step {step} parity (prev - gen), first-EOS semantics, n={n} ---")
    for b, ds in buckets.items():
        if not ds:
            continue
        print(
            f"  {b:16s}: n={len(ds):6d}  signed_mean={statistics.mean(ds):+.4f}"
            f"  mean|d|={statistics.mean(abs(x) for x in ds):.4f}"
        )
    if tail_zero:
        print(f"  tail prev==0 frac: {statistics.mean(tail_zero):.3f} (1.0 = context-only tail masking active)")


if __name__ == "__main__":
    logdir = sys.argv[1]
    steps = [int(x) for x in sys.argv[2:]] or [3]
    for s in steps:
        try:
            analyze(logdir, s)
        except FileNotFoundError:
            print(f"step {s}: no train_data file yet")
