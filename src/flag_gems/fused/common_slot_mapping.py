# Copyright (c) 2025 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph-safe slot and computed-token metadata kernels."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["num_tokens", "max_num_tokens"])
def _compute_slot_mapping_graph_kernel(
    num_tokens,
    max_num_tokens,
    query_start_loc_ptr,
    positions_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    slot_mapping_ptr,
    TOTAL_CP_WORLD_SIZE: tl.constexpr,
    TOTAL_CP_RANK: tl.constexpr,
    CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
    NULL_BLOCK_ID: tl.constexpr,
    PAD_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)

    if req_idx == tl.num_programs(0) - 1:
        actual_num_tokens = tl.load(query_start_loc_ptr + req_idx).to(tl.int64)
        for i in range(actual_num_tokens, max_num_tokens, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(
                slot_mapping_ptr + offsets,
                PAD_ID,
                mask=offsets < max_num_tokens,
            )
        return

    start_idx = tl.load(query_start_loc_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(query_start_loc_ptr + req_idx + 1).to(tl.int64)

    # Rows for graph-padded requests are not refreshed by
    # BlockTable.commit_block_table(). Clear them in the existing per-group
    # metadata producer so replay remains address-stable and no eager fill is
    # needed for every cache group.
    # A graph-padded row has no scheduled query tokens. Do not use seq_len as
    # the predicate: valid scheduler rows can transiently carry seq_len == 0.
    if start_idx == end_idx:
        row_offset = req_idx * block_table_stride
        for i in range(0, block_table_stride, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(
                block_table_ptr + row_offset + offsets,
                NULL_BLOCK_ID,
                mask=offsets < block_table_stride,
            )

    virtual_block_size = block_size * TOTAL_CP_WORLD_SIZE
    row_offset = req_idx * block_table_stride
    for i in range(start_idx, end_idx, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < end_idx
        pos = tl.load(positions_ptr + offsets, mask=mask, other=0)
        block_indices = pos // virtual_block_size
        block_numbers = tl.load(block_table_ptr + row_offset + block_indices).to(
            tl.int64
        )

        virtual_block_offsets = pos - block_indices * virtual_block_size
        is_local = (
            virtual_block_offsets // CP_KV_CACHE_INTERLEAVE_SIZE
        ) % TOTAL_CP_WORLD_SIZE == TOTAL_CP_RANK
        local_block_offsets = (
            virtual_block_offsets // (TOTAL_CP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)
        ) * CP_KV_CACHE_INTERLEAVE_SIZE + (
            virtual_block_offsets % CP_KV_CACHE_INTERLEAVE_SIZE
        )

        slot_ids = block_numbers * block_size + local_block_offsets
        slot_ids = tl.where(is_local, slot_ids, PAD_ID)
        tl.store(slot_mapping_ptr + offsets, slot_ids, mask=mask)


@triton.jit
def _compute_num_computed_tokens_kernel(
    query_start_loc_ptr,
    seq_lens_ptr,
    num_computed_tokens_ptr,
    num_reqs: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_reqs
    query_start = tl.load(query_start_loc_ptr + offsets, mask=mask, other=0)
    query_end = tl.load(query_start_loc_ptr + offsets + 1, mask=mask, other=0)
    seq_len = tl.load(seq_lens_ptr + offsets, mask=mask, other=0)
    tl.store(
        num_computed_tokens_ptr + offsets,
        seq_len - (query_end - query_start),
        mask=mask,
    )


def compute_common_slot_mapping(
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_reqs: int,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    *,
    max_num_batched_tokens: int,
    block_size: int,
    total_cp_world_size: int = 1,
    total_cp_rank: int = 0,
    cp_kv_cache_interleave_size: int = 1,
    null_block_id: int = 0,
    pad_slot_id: int = -1,
    update_num_computed_tokens: bool = True,
) -> None:
    """Generate one cache group's slot mapping and shared token metadata.

    Framework adapters unpack their block-table objects and call this tensor
    API once per cache group. Set ``update_num_computed_tokens`` only for the
    first group because that output is shared across groups.
    """
    if block_table.ndim != 2 or slot_mapping.ndim != 1:
        raise ValueError("block_table must be rank 2 and slot_mapping rank 1")
    if not 0 <= total_cp_rank < total_cp_world_size:
        raise ValueError("total_cp_rank must be within total_cp_world_size")
    if block_size <= 0 or cp_kv_cache_interleave_size <= 0:
        raise ValueError("block sizes must be positive")
    _compute_slot_mapping_graph_kernel[(num_reqs + 1,)](
        positions.shape[0],
        max_num_batched_tokens,
        query_start_loc,
        positions,
        block_table,
        block_table.stride(0),
        block_size,
        slot_mapping,
        TOTAL_CP_WORLD_SIZE=total_cp_world_size,
        TOTAL_CP_RANK=total_cp_rank,
        CP_KV_CACHE_INTERLEAVE_SIZE=cp_kv_cache_interleave_size,
        NULL_BLOCK_ID=null_block_id,
        PAD_ID=pad_slot_id,
        BLOCK_SIZE=1024,
    )
    if update_num_computed_tokens:
        _compute_num_computed_tokens_kernel[(triton.cdiv(num_reqs, 256),)](
            query_start_loc,
            seq_lens,
            num_computed_tokens,
            num_reqs=num_reqs,
            BLOCK_SIZE=256,
        )
