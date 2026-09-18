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

import pytest
import torch

import flag_gems
from flag_gems import testing as utils
from flag_gems.fused import fused_moe as generic_fused_moe
from flag_gems.runtime.backend._thead.fused import fused_moe as thead_fused_moe
from flag_gems.runtime.backend._thead.fused.moe_sum import moe_sum as thead_moe_sum


@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="T-Head only")
def test_thead_fused_moe_is_selected_by_backend_registrar():
    assert flag_gems.fused_experts_impl.__module__ == thead_fused_moe.__name__
    assert flag_gems.moe_sum.__module__.endswith("_thead.fused.moe_sum")


@pytest.mark.parametrize(
    "is_target,num_tokens,expected_warps",
    [
        (True, 255, 8),
        (True, 256, 4),
        (True, 2048, 4),
        (False, 256, 8),
    ],
)
def test_thead_ppu_w8a8_uses_four_warps_from_256_tokens(
    monkeypatch, is_target, num_tokens, expected_warps
):
    monkeypatch.setattr(thead_fused_moe, "_is_target_ppu", lambda: is_target)
    config = thead_fused_moe._thead_get_default_config(
        num_tokens,
        256,
        256,
        6144,
        8,
        "int8_w8a8",
    )
    assert config["num_warps"] == expected_warps


def _quantize_reference(x):
    x_flat = x.reshape(-1, x.shape[-1])
    scale = x_flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10).float() / 127
    quantized = (x_flat.float() / scale).round().clamp(-128, 127)
    return quantized.reshape(x.shape), scale.reshape(x.shape[:-1] + (1,))


@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="T-Head only")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("hidden_size", [128, 256, 6144])
def test_thead_dynamic_per_token_int8_quant_is_exact(dtype, hidden_size):
    if not thead_fused_moe._is_target_ppu():
        pytest.skip("PPU-ZW810E-specific fused quantization regression")
    generator = torch.Generator(device="cpu").manual_seed(20260916 + hidden_size)
    x = torch.randn(7, hidden_size, generator=generator, dtype=dtype)
    x[0].zero_()
    x[1, :8] = torch.tensor(
        [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5], dtype=dtype
    )
    x = x.to(flag_gems.device)
    expected_q, expected_scale = _quantize_reference(x)
    actual_q, actual_scale = thead_fused_moe._thead_int8_quantize(
        x,
        A_scale=None,
        per_act_token=True,
    )
    assert torch.equal(actual_q, expected_q.to(torch.int8))
    assert torch.equal(actual_scale, expected_scale)


@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="T-Head only")
@pytest.mark.parametrize("m", [1, 256, 513, 2048])
def test_thead_hy4_moe_sum_specialization(m):
    shape = (m, 8, 6144)
    inp = torch.randn(shape, dtype=torch.bfloat16, device=flag_gems.device)
    out = torch.empty((m, shape[-1]), dtype=inp.dtype, device=inp.device)
    reference = torch.sum(utils.to_reference(inp), dim=1)

    thead_moe_sum(inp, out)

    utils.gems_assert_close(out, reference, inp.dtype)


def test_generic_fused_moe_helpers_remain_platform_independent():
    config = generic_fused_moe.get_default_config(
        2048,
        256,
        256,
        6144,
        8,
        "int8_w8a8",
    )
    assert config["num_warps"] == 8
