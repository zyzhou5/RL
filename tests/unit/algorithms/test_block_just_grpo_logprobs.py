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

"""CPU unit tests for the BlockJustGRPO block-reveal builder primitives.

Pure-CPU (no GPU / model / Ray). Validates that per reveal level the view
reveals the first ``l*k`` block tokens and the harvest masks partition the
response exactly once, that scatter round-trips noisy positions back to
[N, S], the loss view carries scattered advantages, and the training schedule
emits exactly the per-level views -- across reveal widths k in {1, 2, 3, 4}.
"""
from __future__ import annotations

import torch

import pytest

from nemo_rl.algorithms.block_just_grpo_logprobs import (
    BlockJustGRPORevealSchedule,
    build_block_kept_mask,
    build_block_reveal_base,
    build_block_topk_offsets,
    make_reveal_level_view,
    require_generation_entropy,
    scatter_block_reveal_logprobs,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

MASK = 100
PAD = 0
BLOCK = 4
SCORE = ("diffu_grpo_score_mask",)
LOSS = ("diffu_grpo_score_mask", "diffu_grpo_loss_mask")


def _make_data() -> BatchedDataDict:
    S, N = 16, 2
    input_ids = torch.arange(1, S + 1).unsqueeze(0).repeat(N, 1).long()
    token_mask = torch.zeros(N, S)
    input_lengths = torch.zeros(N, dtype=torch.long)
    token_mask[0, 3:14] = 1.0
    input_lengths[0] = 14
    token_mask[1, 3:8] = 1.0
    input_lengths[1] = 8
    return BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "token_mask": token_mask,
            "sample_mask": torch.ones(N),
        }
    )


def test_reveal_pattern_and_harvest_partition():
    data = _make_data()
    base, N, num_levels, _selected = build_block_reveal_base(
        data, mask_token_id=MASK, pad_token_id=PAD, block_size=BLOCK, include_loss=False
    )
    assert N == 2 and num_levels == 4
    noisy_len = int(base["diffu_grpo_noisy_lengths"][0].item())
    resp = base["diffu_grpo_response_lengths"]
    cover = torch.zeros(N, noisy_len)
    for j in range(num_levels):
        view = make_reveal_level_view(base, j, BLOCK, SCORE)
        target = view["diffu_grpo_target_ids"]
        for s in range(N):
            for rel in range(int(resp[s].item())):
                off = rel % BLOCK
                if off < j:
                    assert int(view["input_ids"][s, rel].item()) == int(target[s, rel].item())
                else:
                    assert int(view["input_ids"][s, rel].item()) == MASK
        cover += view["block_reveal_harvest_mask"][:, :noisy_len]
    for s in range(N):
        r = int(resp[s].item())
        assert torch.all(cover[s, :r] == 1.0) and torch.all(cover[s, r:] == 0.0)


def test_reveal_k_levels_partition_and_scatter():
    """k tokens revealed/harvested per level: correct level count, the harvest
    windows still partition the response exactly once, and scatter round-trips."""
    data = _make_data()
    for k, expect_levels in ((1, 4), (2, 2), (3, 2), (4, 1)):
        base, N, num_levels, _selected = build_block_reveal_base(
            data,
            mask_token_id=MASK,
            pad_token_id=PAD,
            block_size=BLOCK,
            include_loss=False,
            reveal_tokens_per_level=k,
        )
        assert num_levels == expect_levels, (k, num_levels)
        noisy_len = int(base["diffu_grpo_noisy_lengths"][0].item())
        noisy_offset = int(base["diffu_grpo_noisy_response_offsets"][0].item())
        resp = base["diffu_grpo_response_lengths"]
        cover = torch.zeros(N, noisy_len)
        out = torch.zeros(N, data["input_ids"].shape[1])
        for level in range(num_levels):
            view = make_reveal_level_view(base, level, BLOCK, SCORE, k)
            target = view["diffu_grpo_target_ids"]
            for s in range(N):
                for rel in range(int(resp[s].item())):
                    off = rel % BLOCK
                    if off < level * k:
                        assert int(view["input_ids"][s, rel].item()) == int(
                            target[s, rel].item()
                        )
                    else:
                        assert int(view["input_ids"][s, rel].item()) == MASK
            cover += view["block_reveal_harvest_mask"][:, :noisy_len]
            starts = view["diffu_grpo_completion_starts"]
            flat = torch.zeros(N, noisy_len)
            for s in range(N):
                flat[s] = torch.arange(noisy_len) + int(starts[s].item()) + 1.0
            out += scatter_block_reveal_logprobs(
                flat_logprobs=flat,
                harvest_mask=view["block_reveal_harvest_mask"],
                sample_index=view["block_reveal_sample_index"],
                completion_starts=starts,
                noisy_response_offset=noisy_offset,
                original_seq_len=data["input_ids"].shape[1],
                num_samples=N,
            )
        for s in range(N):
            r = int(resp[s].item())
            assert torch.all(cover[s, :r] == 1.0) and torch.all(cover[s, r:] == 0.0)
        for s, (a, b) in {0: (3, 14), 1: (3, 8)}.items():
            assert torch.allclose(out[s, a:b], torch.arange(a, b) + 1.0)
            m = torch.ones(out.shape[1], dtype=torch.bool)
            m[a:b] = False
            assert torch.all(out[s, m] == 0.0)


def test_scatter_roundtrip():
    data = _make_data()
    base, N, num_levels, _selected = build_block_reveal_base(
        data, mask_token_id=MASK, pad_token_id=PAD, block_size=BLOCK, include_loss=False
    )
    noisy_len = int(base["diffu_grpo_noisy_lengths"][0].item())
    noisy_offset = int(base["diffu_grpo_noisy_response_offsets"][0].item())
    out = torch.zeros(N, data["input_ids"].shape[1])
    for j in range(num_levels):
        view = make_reveal_level_view(base, j, BLOCK, SCORE)
        starts = view["diffu_grpo_completion_starts"]
        # encode each noisy position as (mapped original position + 1)
        flat = torch.zeros(N, noisy_len)
        for s in range(N):
            flat[s] = torch.arange(noisy_len) + int(starts[s].item()) + 1.0
        out += scatter_block_reveal_logprobs(
            flat_logprobs=flat,
            harvest_mask=view["block_reveal_harvest_mask"],
            sample_index=view["block_reveal_sample_index"],
            completion_starts=starts,
            noisy_response_offset=noisy_offset,
            original_seq_len=data["input_ids"].shape[1],
            num_samples=N,
        )
    for s, (a, b) in {0: (3, 14), 1: (3, 8)}.items():
        assert torch.allclose(out[s, a:b], torch.arange(a, b) + 1.0)
        m = torch.ones(out.shape[1], dtype=torch.bool)
        m[a:b] = False
        assert torch.all(out[s, m] == 0.0)


def test_loss_view():
    data = _make_data()
    N, S = data["input_ids"].shape
    data["advantages"] = torch.arange(N * S, dtype=torch.float32).reshape(N, S)
    data["prev_logprobs"] = torch.zeros(N, S)
    data["generation_logprobs"] = torch.zeros(N, S)
    base, _, num_levels, _selected = build_block_reveal_base(
        data, mask_token_id=MASK, pad_token_id=PAD, block_size=BLOCK, include_loss=True
    )
    noisy_len = int(base["diffu_grpo_noisy_lengths"][0].item())
    for j in range(num_levels):
        view = make_reveal_level_view(base, j, BLOCK, LOSS)
        assert torch.allclose(view["diffu_grpo_loss_mask"], view["block_reveal_harvest_mask"])
        starts = view["diffu_grpo_completion_starts"]
        for s in range(N):
            start = int(starts[s].item())
            harvested = torch.nonzero(
                view["block_reveal_harvest_mask"][s, :noisy_len] > 0.5, as_tuple=False
            ).flatten().tolist()
            for rel in harvested:
                assert float(view["advantages"][s, rel].item()) == float(
                    data["advantages"][s, start + rel].item()
                )


def test_reveal_schedule():
    """Schedule microbatches == the per-level views concatenated (training path)."""
    data = _make_data()
    N, S = data["input_ids"].shape
    data["advantages"] = torch.arange(N * S, dtype=torch.float32).reshape(N, S)
    data["prev_logprobs"] = torch.zeros(N, S)
    data["generation_logprobs"] = torch.zeros(N, S)
    base, n, num_levels, _selected = build_block_reveal_base(
        data, mask_token_id=MASK, pad_token_id=PAD, block_size=BLOCK, include_loss=True
    )
    schedule = BlockJustGRPORevealSchedule(base).configure(
        num_levels=num_levels, block_size=BLOCK, harvest_keys=LOSS
    )
    mbs = 1  # N == 2; two sample microbatches per reveal level
    assert schedule.size == num_levels * n
    mbs_list = list(schedule.make_microbatch_iterator(mbs))
    assert len(mbs_list) == schedule.size // mbs
    for key in ("input_ids", "diffu_grpo_loss_mask"):
        got = torch.cat([mb[key] for mb in mbs_list], dim=0)
        ref = torch.cat(
            [make_reveal_level_view(base, j, BLOCK, LOSS)[key] for j in range(num_levels)],
            dim=0,
        )
        assert torch.equal(got, ref), key


# --------------------------------------------------------------------------- #
# JustGRPO-Fast: entropy-sparsified block-reveal (fast_entropy_level_ratio)
# --------------------------------------------------------------------------- #
def _make_entropy_data() -> BatchedDataDict:
    """``_make_data`` plus a rollout entropy signal: within-block offset value.

    entropy[response_token r] = r % BLOCK, so the top-``m`` offsets per block are
    deterministically the ``m`` largest within-block offsets (easy to assert).
    """
    data = _make_data()
    N, S = data["input_ids"].shape
    entropy = torch.zeros(N, S)
    tm = data["token_mask"]
    for s in range(N):
        idx = torch.nonzero(tm[s] > 0.5, as_tuple=False).flatten().tolist()
        for rel, pos in enumerate(idx):
            entropy[s, pos] = float(rel % BLOCK)
    data["generation_entropy"] = entropy
    return data


def test_build_block_topk_offsets():
    """Top-``m`` offsets per (sample, block): correct picks, ascending sort, and
    sentinel -1 padding for partial trailing blocks."""
    data = _make_entropy_data()
    base, N, _, _selected = build_block_reveal_base(
        data, mask_token_id=MASK, pad_token_id=PAD, block_size=BLOCK, include_loss=False
    )
    sel = build_block_topk_offsets(
        entropy=data["generation_entropy"],
        response_lengths=base["diffu_grpo_response_lengths"],
        completion_starts=base["diffu_grpo_completion_starts"],
        block_size=BLOCK,
        m=2,
    )
    # sample 0 (resp 11): 3 blocks, last partial (offsets 0,1,2 valid).
    # sample 1 (resp 5): block0 full, block1 has only offset 0, block2 empty.
    expected = torch.tensor(
        [
            [[2, 3], [2, 3], [1, 2]],
            [[2, 3], [0, -1], [-1, -1]],
        ]
    )
    assert sel.shape == (2, 3, 2)
    assert torch.equal(sel, expected), sel


def test_fast_within_block_reveal_exactness():
    """Each kept offset ``o`` (rank ``j``) reveals its own block prefix ``0..o-1``
    and harvests exactly ``o`` -- byte-identical within-block conditioning to the
    full block-reveal view at absolute level ``o``. Sentinel offsets reveal and
    harvest nothing, and the 3-D selection tensor is stripped from the view."""
    data = _make_entropy_data()
    base, N, num_levels, selected = build_block_reveal_base(
        data,
        mask_token_id=MASK,
        pad_token_id=PAD,
        block_size=BLOCK,
        include_loss=False,
        fast_entropy_level_ratio=0.5,
    )
    assert num_levels == 2  # ceil(0.5 * 4)
    resp = base["diffu_grpo_response_lengths"]
    noisy_len = int(base["diffu_grpo_noisy_lengths"][0].item())
    cover = torch.zeros(N, noisy_len)
    for j in range(num_levels):
        view = make_reveal_level_view(base, j, BLOCK, SCORE, selected_offsets=selected)
        assert "block_reveal_selected_offsets" not in view
        target = view["diffu_grpo_target_ids"]
        harvest = view["block_reveal_harvest_mask"]
        cover += harvest[:, :noisy_len]
        for s in range(N):
            for b in range(selected.shape[1]):
                o = int(selected[s, b, j].item())
                for off in range(BLOCK):
                    rel = b * BLOCK + off
                    if rel >= int(resp[s].item()):
                        continue
                    ii = int(view["input_ids"][s, rel].item())
                    if o >= 0 and off < o:
                        assert ii == int(target[s, rel].item())
                    else:
                        assert ii == MASK
                    expect_h = 1.0 if (o >= 0 and off == o) else 0.0
                    assert float(harvest[s, rel].item()) == expect_h
    # No response position is harvested by more than one level.
    assert torch.all(cover <= 1.0)


def test_fast_ratio_one_parity():
    """``fast_entropy_level_ratio = 1.0`` keeps all offsets -> no selection is
    attached and every reveal-level view is byte-for-byte identical to the full
    Block JustGRPO path."""
    data = _make_entropy_data()
    base_full, _, nl_full, sel_full = build_block_reveal_base(
        data, mask_token_id=MASK, pad_token_id=PAD, block_size=BLOCK, include_loss=True
    )
    base_one, _, nl_one, sel_one = build_block_reveal_base(
        data,
        mask_token_id=MASK,
        pad_token_id=PAD,
        block_size=BLOCK,
        include_loss=True,
        fast_entropy_level_ratio=1.0,
    )
    assert nl_one == nl_full
    assert sel_full is None and sel_one is None  # ratio>=1 -> uniform, no selection
    for j in range(nl_full):
        vf = make_reveal_level_view(base_full, j, BLOCK, LOSS)
        vo = make_reveal_level_view(base_one, j, BLOCK, LOSS)
        for key in ("input_ids", "block_reveal_harvest_mask", "diffu_grpo_loss_mask"):
            assert torch.equal(vf[key], vo[key]), key


def test_fast_kept_mask_matches_per_level_harvest_and_is_sparse():
    """The Fast kept mask == the union (== sum, harvests are disjoint) of the per-level
    harvest masks, and covers strictly fewer tokens than the full response set -- so the
    loss normalizer must use this count, not the full ``token_mask`` count (which is the
    bug this guards)."""
    data = _make_entropy_data()
    base, N, num_levels, selected = build_block_reveal_base(
        data,
        mask_token_id=MASK,
        pad_token_id=PAD,
        block_size=BLOCK,
        include_loss=False,
        fast_entropy_level_ratio=0.5,
    )
    assert num_levels == 2
    # Independent reference: sum the actual per-level harvest masks the loss sees.
    per_level_sum = None
    for j in range(num_levels):
        h = make_reveal_level_view(base, j, BLOCK, SCORE, selected_offsets=selected)[
            "block_reveal_harvest_mask"
        ]
        per_level_sum = h if per_level_sum is None else per_level_sum + h
    kept = build_block_kept_mask(base, selected, BLOCK)
    assert torch.equal(kept, per_level_sum)  # per-level harvests are disjoint (0/1)
    kept_count = float(kept.sum())
    full_count = float(base["diffu_grpo_score_mask"].sum())  # full response tokens
    assert kept_count == float(per_level_sum.sum())
    assert 0.0 < kept_count < full_count


def test_fast_requires_reveal_tokens_per_level_one():
    """JustGRPO-Fast is width-1 per selected offset; k>1 + Fast must raise (both the
    training and logprob paths go through build_block_reveal_base)."""
    data = _make_entropy_data()
    with pytest.raises(ValueError, match="reveal_tokens_per_level == 1"):
        build_block_reveal_base(
            data,
            mask_token_id=MASK,
            pad_token_id=PAD,
            block_size=BLOCK,
            include_loss=False,
            fast_entropy_level_ratio=0.5,
            reveal_tokens_per_level=2,
        )
    # k>1 without Fast stays allowed.
    _base, _n, levels, _selected = build_block_reveal_base(
        data,
        mask_token_id=MASK,
        pad_token_id=PAD,
        block_size=BLOCK,
        include_loss=False,
        reveal_tokens_per_level=2,
    )
    assert levels == 2


def test_require_generation_entropy_raises_when_channel_absent():
    """Fast active but generation emitted no entropy -> hard error (not zero-fill)."""
    with_entropy = [
        [{"role": "user"}, {"role": "assistant", "generation_entropy": torch.zeros(3)}]
    ]
    require_generation_entropy(with_entropy)  # present -> no raise

    without_entropy = [[{"role": "user"}, {"role": "assistant"}]]
    with pytest.raises(ValueError, match="return_entropy: true"):
        require_generation_entropy(without_entropy)


def test_fast_training_schedule_has_no_nonsequence_tensors():
    """Regression: the Fast training schedule's data-dict must hold only ``[B, S]`` /
    ``[B]`` tensors. The 3-D selection tensor must NOT ride inside the validated
    container -- otherwise ``get_and_validate_seqlen`` (run on the whole schedule
    before the per-level views are materialized) asserts on its dim-1."""
    data = _make_entropy_data()
    N, S = data["input_ids"].shape
    data["advantages"] = torch.arange(N * S, dtype=torch.float32).reshape(N, S)
    data["prev_logprobs"] = torch.zeros(N, S)
    data["generation_logprobs"] = torch.zeros(N, S)
    base, _n, num_levels, selected = build_block_reveal_base(
        data,
        mask_token_id=MASK,
        pad_token_id=PAD,
        block_size=BLOCK,
        include_loss=True,
        fast_entropy_level_ratio=0.5,
    )
    assert selected is not None and selected.ndim == 3  # [N, num_blocks, m]
    schedule = BlockJustGRPORevealSchedule(base).configure(
        num_levels=num_levels,
        block_size=BLOCK,
        harvest_keys=LOSS,
        selected_offsets=selected,
    )
    for key, value in schedule.data.items():
        if torch.is_tensor(value):
            assert value.ndim <= 2, (key, tuple(value.shape))
    # The emitted per-level microbatches are also clean [B, S] / [B].
    for mb in schedule.make_microbatch_iterator(1):
        for key, value in mb.data.items():
            if torch.is_tensor(value):
                assert value.ndim <= 2, (key, tuple(value.shape))
