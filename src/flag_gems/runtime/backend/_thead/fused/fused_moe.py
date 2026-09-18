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

import contextlib
import threading
from typing import Any, Optional

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice

from flag_gems.fused import fused_moe as generic_fused_moe
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from .moe_sum import moe_sum as _thead_moe_sum

_PATCH_LOCK = threading.RLock()
_GENERIC_GET_DEFAULT_CONFIG = generic_fused_moe.get_default_config
_GENERIC_INT8_QUANTIZE = generic_fused_moe._int8_quantize
_W8A8_FOUR_WARP_MIN_TOKENS = 256


def _is_target_ppu() -> bool:
    return generic_fused_moe._get_device_name() == "PPU-ZW810E"


def _thead_get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
    dtype: str | None,
    block_shape: list[int] | None = None,
    gemm_stage: str = "gemm1",
    enable_gemm_fast_path: bool = False,
) -> dict[str, Any]:
    config = _GENERIC_GET_DEFAULT_CONFIG(
        M,
        E,
        N,
        K,
        topk,
        dtype,
        block_shape,
        gemm_stage,
        enable_gemm_fast_path,
    )
    if (
        dtype == "int8_w8a8"
        and _is_target_ppu()
        and M >= _W8A8_FOUR_WARP_MIN_TOKENS
    ):
        config = config.copy()
        config["num_warps"] = 4
    return config


@libentry()
@triton.jit
def _dynamic_per_token_int8_quant_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size
    values = tl.load(
        input_ptr + token_idx * hidden_size + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    absmax = tl.max(tl.abs(values), axis=0)
    absmax = tl.maximum(absmax, 1e-10)
    scale = absmax / 127.0
    quantized = libdevice.rint(libdevice.div_rn(values, scale))
    quantized = tl.minimum(tl.maximum(quantized, -128.0), 127.0)
    tl.store(
        output_ptr + token_idx * hidden_size + offsets,
        quantized.to(tl.int8),
        mask=mask,
    )
    tl.store(scale_ptr + token_idx, scale)


def _dynamic_per_token_int8_quant(
    A: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    original_shape = A.shape
    A_flat = A.reshape(-1, A.shape[-1])
    output = torch.empty_like(A_flat, dtype=torch.int8)
    scale = torch.empty(
        (A_flat.shape[0], 1), device=A.device, dtype=torch.float32
    )
    block_size = triton.next_power_of_2(A_flat.shape[1])
    with torch_device_fn.device(A.device):
        _dynamic_per_token_int8_quant_kernel[(A_flat.shape[0],)](
            A_flat,
            output,
            scale,
            hidden_size=A_flat.shape[1],
            BLOCK_SIZE=block_size,
        )
    return output.reshape(original_shape), scale.reshape(original_shape[:-1] + (1,))


def _thead_int8_quantize(
    A: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    per_act_token: bool,
    block_shape: Optional[list[int]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        per_act_token
        and block_shape is None
        and _is_target_ppu()
        and A.is_contiguous()
        and 0 < A.shape[-1] <= 8192
    ):
        return _dynamic_per_token_int8_quant(A)
    return _GENERIC_INT8_QUANTIZE(A, A_scale, per_act_token, block_shape)


@contextlib.contextmanager
def _thead_moe_patch():
    with _PATCH_LOCK:
        original_get_default_config = generic_fused_moe.get_default_config
        original_int8_quantize = generic_fused_moe._int8_quantize
        original_moe_sum = generic_fused_moe.moe_sum
        generic_fused_moe.get_default_config = _thead_get_default_config
        generic_fused_moe._int8_quantize = _thead_int8_quantize
        generic_fused_moe.moe_sum = _thead_moe_sum
        try:
            yield
        finally:
            generic_fused_moe.get_default_config = original_get_default_config
            generic_fused_moe._int8_quantize = original_int8_quantize
            generic_fused_moe.moe_sum = original_moe_sum


def fused_experts_impl(*args, **kwargs):
    with _thead_moe_patch():
        return generic_fused_moe.fused_experts_impl(*args, **kwargs)


def inplace_fused_experts(*args, **kwargs):
    with _thead_moe_patch():
        return generic_fused_moe.inplace_fused_experts(*args, **kwargs)


def outplace_fused_experts(*args, **kwargs):
    with _thead_moe_patch():
        return generic_fused_moe.outplace_fused_experts(*args, **kwargs)


__all__ = ["fused_experts_impl", "inplace_fused_experts", "outplace_fused_experts"]
