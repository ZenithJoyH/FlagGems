# Copyright 2026 FlagOS Contributors
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

import torch
import triton

from flag_gems.fused.moe_sum import moe_sum as _generic_moe_sum
from flag_gems.fused.moe_sum import moe_sum_kernel


def moe_sum(input: torch.Tensor, output: torch.Tensor) -> None:
    """Use the measured fixed HY4 reduction config on T-Head."""
    num_tokens, topk, hidden_size = input.shape
    if not (
        input.dtype == torch.bfloat16
        and topk == 8
        and hidden_size == 6144
        and input.is_contiguous()
        and output.is_contiguous()
    ):
        _generic_moe_sum(input, output)
        return

    input_strides = input.stride()
    output_strides = output.stride()
    block_size, num_warps = (256, 4) if num_tokens <= 512 else (512, 8)
    grid = (num_tokens, triton.cdiv(hidden_size, block_size))
    moe_sum_kernel.fn[grid](
        input,
        output,
        num_tokens,
        topk,
        hidden_size,
        input_strides[0],
        input_strides[1],
        input_strides[2],
        output_strides[0],
        output_strides[1],
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )


__all__ = ["moe_sum"]
