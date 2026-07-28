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

"""Block-aware (segmented) CP: NeMo-RL's sharding must match Megatron-Bridge's.

Stage (b) of ``plans/cp_kv_sharding_blockaware_ring.md`` splits ``[noisy |
clean]`` into one zigzag per segment. Four independent places implement or
consume that layout:

  - NeMo-RL ``data.process_microbatch``          (shards input_ids)
  - NeMo-RL ``_cp_sharded_same_position_logprobs`` (shards target ids, re-gathers)
  - Megatron-Bridge attention                     (shards RoPE ids, builds the mask)
  - the attention mask's own index tables

If any two disagree about which global position a local row holds, nothing
raises -- the logprobs are simply taken from the wrong tokens. These tests pin
the two implementations against each other, which is the failure this layout is
most exposed to.
"""

import pytest
import torch

from megatron.bridge.diffusion.common.cp_utils import (
    reorder_segmented_zigzag_shards,
    segmented_zigzag_slice,
)
from nemo_rl.distributed.model_utils import (
    _get_segmented_tokens_on_this_cp_rank,
    _get_tokens_on_this_cp_rank,
)
from nemo_rl.models.megatron.cp_block_aware import assert_block_aware_divisible


SEGMENTS = [(128, 128), (64, 192), (256, 128)]


@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("segments", SEGMENTS)
def test_rl_sharding_matches_bridge(cp_size: int, segments: tuple) -> None:
    """The two implementations of the segmented split must be identical."""
    total = sum(segments)
    seq = torch.arange(total).view(1, total)
    for rank in range(cp_size):
        rl = _get_segmented_tokens_on_this_cp_rank(seq, segments, rank, cp_size, seq_dim=1)
        bridge = segmented_zigzag_slice(seq, segments, rank, cp_size, seq_dim=1)
        assert torch.equal(rl, bridge), (
            f"cp={cp_size} rank={rank}: NeMo-RL and Megatron-Bridge disagree on the "
            f"segmented layout -- RL {rl.tolist()} vs bridge {bridge.tolist()}"
        )


@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("segments", SEGMENTS)
def test_segmented_shards_cover_the_sequence_exactly(cp_size: int, segments: tuple) -> None:
    """Every position lands on exactly one rank, and re-gather is the inverse."""
    total = sum(segments)
    seq = torch.arange(total).view(1, total)
    shards = [
        _get_segmented_tokens_on_this_cp_rank(seq, segments, r, cp_size, seq_dim=1)
        for r in range(cp_size)
    ]
    for shard in shards:
        assert shard.shape[1] == total // cp_size

    owned = torch.cat(shards, dim=1).flatten().sort().values
    assert torch.equal(owned, torch.arange(total)), "shards do not partition the sequence"

    regathered = reorder_segmented_zigzag_shards(shards, segments, cp_size, seq_dim=1)
    assert torch.equal(regathered, seq), "re-gather is not the inverse of the shard"


@pytest.mark.parametrize("cp_size", [2, 4])
def test_segmented_layout_differs_from_global_zigzag(cp_size: int) -> None:
    """The two layouts assign DIFFERENT rows -- the reason all consumers must agree.

    This is not a local permutation that attention could absorb internally: at
    cp=2 the single-zigzag rank 0 holds the noisy first half plus the clean
    second half, while the segmented rank 0 holds noisy chunks {0,3} and clean
    chunks {0,3}. Different positions, not the same positions reordered.
    """
    segments = (128, 128)
    total = sum(segments)
    seq = torch.arange(total).view(1, total)
    differs = False
    for rank in range(cp_size):
        segmented = _get_segmented_tokens_on_this_cp_rank(
            seq, segments, rank, cp_size, seq_dim=1
        ).flatten()
        global_zigzag = _get_tokens_on_this_cp_rank(seq, rank, cp_size, seq_dim=1).flatten()
        if set(segmented.tolist()) != set(global_zigzag.tolist()):
            differs = True
    assert differs, (
        "segmented and global-zigzag layouts own the same rows on every rank -- "
        "if that were true the data path would not need to change at all"
    )


@pytest.mark.parametrize("cp_size", [2, 4])
def test_divisibility_assert_is_per_segment(cp_size: int) -> None:
    """A globally-divisible sequence can still be invalid per segment."""
    # Total divides by 2*cp, but the segments individually do not: exactly the
    # case a single global pad produces.
    bad = (2 * cp_size + 1, 2 * cp_size - 1)
    total = sum(bad)
    assert total % (2 * cp_size) == 0, "test case must be globally divisible"
    with pytest.raises(AssertionError, match="divisible by 2.cp_size"):
        assert_block_aware_divisible(bad, cp_size, total)

    good = (4 * cp_size, 4 * cp_size)
    assert_block_aware_divisible(good, cp_size, sum(good))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# ---------------------------------------------------------------------------
# Segment-aware padding (plan section 2.4)
# ---------------------------------------------------------------------------


def _build_layout(response_len, clean_len, block_size, sequence_length_round):
    """Call the real layout builder and return its (noisy_length, clean_length)."""
    from nemo_rl.algorithms.diffu_grpo_logprobs import _build_completion_only_tensors

    batch, width = 1, clean_len + 8
    result = _build_completion_only_tensors(
        input_ids=torch.zeros(batch, width, dtype=torch.long),
        completion_starts=torch.tensor([clean_len - response_len]),
        response_lengths=torch.tensor([response_len]),
        clean_lengths=torch.tensor([clean_len]),
        mask_token_id=1,
        pad_token_id=0,
        block_size=block_size,
        sequence_length_round=sequence_length_round,
    )
    ints = [x for x in result if isinstance(x, int)]
    assert len(ints) == 2, f"expected (noisy_length, clean_length), got {ints}"
    return ints[0], ints[1]


@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("sequence_length_round", [None, 64, 128])
@pytest.mark.parametrize("response_len,clean_len", [(37, 90), (64, 128), (100, 260)])
def test_padding_is_segment_aware(
    monkeypatch, cp_size, sequence_length_round, response_len, clean_len
) -> None:
    """Each segment must land on its OWN grid, including after the global round.

    The builder used to apply one global round and push all the slack into the
    clean side, which leaves the noisy|clean boundary at an arbitrary offset --
    globally divisible, per-segment not. That is the case this pins.
    """
    import megatron.core.parallel_state as ps

    from nemo_rl.models.megatron import cp_block_aware

    block_size = 16
    monkeypatch.setattr(cp_block_aware, "_BLOCK_AWARE_CP", True)
    monkeypatch.setattr(ps, "get_context_parallel_world_size", lambda: cp_size)

    noisy_length, clean_length = _build_layout(
        response_len, clean_len, block_size, sequence_length_round
    )

    # The exact contract the attention layer and the data path assert on.
    assert_block_aware_divisible(
        (noisy_length, clean_length), cp_size, noisy_length + clean_length
    )
    assert noisy_length % (2 * cp_size * block_size) == 0, (
        f"noisy chunk grid is not block-aligned: {noisy_length} % "
        f"{2 * cp_size * block_size} != 0"
    )
    assert noisy_length >= response_len, "padding must not truncate the response"
    assert clean_length >= clean_len, "padding must not truncate the clean side"


@pytest.mark.parametrize("cp_size", [2, 4])
def test_padding_is_untouched_when_flag_is_off(monkeypatch, cp_size) -> None:
    """With the flag off the layout must be byte-identical to before."""
    import megatron.core.parallel_state as ps

    from nemo_rl.models.megatron import cp_block_aware

    monkeypatch.setattr(cp_block_aware, "_BLOCK_AWARE_CP", False)
    monkeypatch.setattr(ps, "get_context_parallel_world_size", lambda: cp_size)

    noisy_length, clean_length = _build_layout(37, 90, 16, 64)
    # 37 rounded up to the block grid is 48; all slack goes to clean, and the
    # total lands on the global round. No per-segment rounding.
    assert noisy_length == 48, f"unexpected noisy_length {noisy_length}"
    assert (noisy_length + clean_length) % 64 == 0


# ---------------------------------------------------------------------------
# The segmented all-gather, over a real process group
# ---------------------------------------------------------------------------
#
# The shard side above is pure tensor math and needs no collective, which is why
# it was covered first -- and why a real bug got through: the GATHER side calls
# torch.distributed.all_gather on a narrow() view, which is strided, and
# all_gather rejects non-contiguous tensors. That surfaced only in a live
# training run. These tests close that gap.
#
# BATCH >= 2 is load-bearing: with a single row, narrow() along the sequence
# dimension still reports contiguous (PyTorch ignores the stride of size-1
# dims), so a 1-row test would NOT reproduce the failure.

import os
import socket

import torch.distributed as dist
import torch.multiprocessing as mp

BATCH = 3


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _gather_worker(rank, world_size, port, segments, out_q):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    real_all_gather = dist.all_gather
    real_all_reduce = dist.all_reduce
    try:
        from nemo_rl.distributed.model_utils import (
            _get_segmented_tokens_on_this_cp_rank,
            allgather_segmented_cp_sharded_tensor,
        )

        # Enforce the contiguity requirement explicitly rather than relying on
        # the backend to raise. NCCL rejects non-contiguous tensors ("Tensors
        # must be contiguous") but GLOO ACCEPTS them, so a CPU test that only
        # checks the result would pass against code that fails in production --
        # verified by reverting the fix and watching this test still pass.
        def checked_all_gather(tensor_list, tensor, *args, **kwargs):
            if not tensor.is_contiguous():
                raise RuntimeError(
                    "all_gather called with a non-contiguous tensor "
                    f"(shape {tuple(tensor.shape)}, strides {tensor.stride()}); "
                    "NCCL rejects this even though gloo tolerates it"
                )
            return real_all_gather(tensor_list, tensor, *args, **kwargs)

        def checked_all_reduce(tensor, *args, **kwargs):
            if not tensor.is_contiguous():
                raise RuntimeError(
                    "all_reduce called with a non-contiguous tensor "
                    f"(shape {tuple(tensor.shape)}, strides {tensor.stride()}); "
                    "NCCL rejects this even though gloo tolerates it"
                )
            return real_all_reduce(tensor, *args, **kwargs)

        dist.all_gather = checked_all_gather
        dist.all_reduce = checked_all_reduce

        total = sum(segments)
        full = torch.arange(BATCH * total, dtype=torch.float32).view(BATCH, total)
        local = _get_segmented_tokens_on_this_cp_rank(
            full, segments, rank, world_size, seq_dim=1
        )
        regathered = allgather_segmented_cp_sharded_tensor(
            local, segments, dist.group.WORLD, seq_dim=1
        )
        out_q.put((rank, bool(torch.equal(regathered, full)), tuple(regathered.shape)))

        # NOTE: forward only. AllGatherCPTensor.backward builds its gather index
        # with a hardcoded CUDA device, so it cannot run under gloo/CPU at all --
        # that is pre-existing, not a property of the segmented path. The second
        # contiguity bug lived in that backward (all_reduce on the strided slice
        # the cat backward produces) and is therefore NOT covered here; it is
        # covered by the end-to-end cp=2 Block JustGRPO run, which is where it
        # was found. Closing this properly needs a 2-GPU NCCL test.
    except Exception as exc:  # noqa: BLE001
        import traceback

        out_q.put((rank, f"EXC {exc!r}\n{traceback.format_exc()}", None))
    finally:
        dist.all_gather = real_all_gather
        dist.all_reduce = real_all_reduce
        dist.destroy_process_group()


@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("segments", [(128, 128), (64, 192)])
def test_segmented_allgather_reconstructs_full_sequence(cp_size, segments) -> None:
    """Round-trip through the real collective, not a simulated one."""
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target=_gather_worker, args=(r, cp_size, port, segments, queue))
        for r in range(cp_size)
    ]
    for p in procs:
        p.start()
    results = [queue.get(timeout=180) for _ in range(cp_size)]
    for p in procs:
        p.join(timeout=180)

    for rank, ok, shape in results:
        assert not isinstance(ok, str), f"rank {rank} raised:\n{ok}"
        assert ok, f"rank {rank}: re-gather did not reconstruct the full sequence"
        assert shape == (BATCH, sum(segments)), f"rank {rank}: bad shape {shape}"


# ---------------------------------------------------------------------------
# The segmented all-gather BACKWARD, over NCCL (needs 2 GPUs)
# ---------------------------------------------------------------------------
#
# This is the one bug the gloo tests structurally cannot reach:
#
#   * AllGatherCPTensor.backward all-reduces the incoming gradient. The
#     segmented forward ends in torch.cat, so the cat backward hands each
#     segment a STRIDED slice -- which NCCL rejects and gloo accepts.
#   * That backward also builds its gather index on a hardcoded CUDA device, so
#     it cannot execute under gloo/CPU at all, regardless of contiguity.
#
# Hence: real GPUs, real NCCL. Skipped when fewer than 2 are visible, which is
# most interactive allocations -- so run it deliberately with --gpus-per-node=2.


def _nccl_backward_worker(rank, world_size, port, segments, out_q):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        from nemo_rl.distributed.model_utils import (
            _get_segmented_tokens_on_this_cp_rank,
            allgather_segmented_cp_sharded_tensor,
        )

        device = torch.device(f"cuda:{rank}")
        total = sum(segments)
        full = torch.arange(BATCH * total, dtype=torch.float32, device=device).view(
            BATCH, total
        )
        local = (
            _get_segmented_tokens_on_this_cp_rank(full, segments, rank, world_size, seq_dim=1)
            .detach()
            .clone()
            .requires_grad_(True)
        )
        regathered = allgather_segmented_cp_sharded_tensor(
            local, segments, dist.group.WORLD, seq_dim=1
        )
        forward_ok = bool(torch.equal(regathered.detach(), full))

        # The backward is the point of this test.
        regathered.sum().backward()
        grad_ok = local.grad is not None and local.grad.shape == local.shape
        out_q.put((rank, bool(forward_ok and grad_ok), None))
    except Exception as exc:  # noqa: BLE001
        import traceback

        out_q.put((rank, f"EXC {exc!r}\n{traceback.format_exc()}", None))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs 2 GPUs for a real NCCL CP group"
)
@pytest.mark.parametrize("segments", [(128, 128), (64, 192)])
def test_segmented_allgather_backward_nccl(segments) -> None:
    """Forward AND backward through the real NCCL collectives.

    Reverting the .contiguous() in AllGatherCPTensor.backward makes this fail
    with "Tensors must be contiguous"; the gloo tests stay green, which is why
    that bug reached a training run.
    """
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target=_nccl_backward_worker, args=(r, 2, port, segments, queue))
        for r in range(2)
    ]
    for p in procs:
        p.start()
    results = [queue.get(timeout=300) for _ in range(2)]
    for p in procs:
        p.join(timeout=300)

    for rank, ok, _ in results:
        assert not isinstance(ok, str), f"rank {rank} raised:\n{ok}"
        assert ok, f"rank {rank}: segmented all-gather forward/backward failed"
