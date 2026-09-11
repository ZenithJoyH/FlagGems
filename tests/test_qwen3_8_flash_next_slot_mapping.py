# SPDX-License-Identifier: Apache-2.0
"""Eager and graph coverage for Qwen3.8 slot metadata kernels."""

import pytest
import torch

from flag_gems.fused import compute_common_slot_mapping


def test_common_slot_mapping_rejects_non_tensor_layouts():
    table = torch.zeros(4, dtype=torch.int32)
    slots = torch.zeros(8, dtype=torch.int64)
    values = torch.zeros(4, dtype=torch.int32)
    with pytest.raises(ValueError, match="block_table"):
        compute_common_slot_mapping(
            table,
            slots,
            1,
            values[:2],
            values,
            values[:1],
            values[:1],
            max_num_batched_tokens=8,
            block_size=4,
        )


@pytest.mark.gpu
@pytest.mark.parametrize("graph_mode", [False, True])
def test_common_slot_mapping_reads_updated_metadata(graph_mode):
    if not torch.cuda.is_available():
        pytest.skip("CUDA-compatible accelerator runtime required")
    device = torch.device("cuda")
    block_table = torch.tensor(
        [
            [2, 3, 4, 5],
            [8, 9, 10, 11],
            [13, 13, 13, 13],
            [14, 14, 14, 14],
        ],
        dtype=torch.int32,
        device=device,
    )
    slots = torch.empty(16, dtype=torch.int64, device=device)
    query_start = torch.tensor([0, 2, 4, 4, 4], dtype=torch.int32, device=device)
    positions = torch.zeros(16, dtype=torch.int64, device=device)
    positions[:4] = torch.tensor([0, 1, 8, 9], device=device)
    seq_lens = torch.tensor([6, 12, 0, 0], dtype=torch.int32, device=device)
    computed = torch.empty(4, dtype=torch.int32, device=device)

    def run():
        compute_common_slot_mapping(
            block_table,
            slots,
            4,
            query_start,
            positions,
            seq_lens,
            computed,
            max_num_batched_tokens=16,
            block_size=4,
        )

    run()
    if graph_mode:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        slots.cpu(),
        torch.tensor([8, 9, 40, 41] + [-1] * 12, dtype=torch.int64),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        computed.cpu(), torch.tensor([4, 10, 0, 0], dtype=torch.int32)
    )
    assert torch.count_nonzero(block_table[2:]).item() == 0

    block_table[0].copy_(torch.tensor([10, 11, 12, 13], device=device))
    query_start.copy_(torch.tensor([0, 1, 3, 3, 3], device=device))
    positions[:3].copy_(torch.tensor([3, 10, 11], device=device))
    seq_lens.copy_(torch.tensor([7, 15, 0, 0], device=device))
    graph.replay() if graph_mode else run()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        slots.cpu(),
        torch.tensor([43, 42, 43] + [-1] * 13, dtype=torch.int64),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        computed.cpu(), torch.tensor([6, 13, 0, 0], dtype=torch.int32)
    )
