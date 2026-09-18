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


def _thead_splitk_inputs():
    torch.manual_seed(20260915)
    q = (torch.randn(2, 4, 576, device="cuda") * 0.05).to(torch.bfloat16)
    kv = (torch.randn(2048, 1, 576, device="cuda") * 0.05).to(
        torch.bfloat16
    )
    indices = torch.arange(2048, device="cuda", dtype=torch.int32)
    indices = indices.view(1, 1, 2048).expand(2, 1, 2048).contiguous()
    indices[1, 0, -8:] = -1
    lengths = torch.tensor([0, 2048], device="cuda", dtype=torch.int32)
    sinks = torch.tensor(
        [float("inf"), float("-inf"), -0.5, 0.5],
        device="cuda",
        dtype=torch.float32,
    )
    return q, kv, indices, lengths, sinks


def _thead_hq4_prefill_inputs(sq):
    torch.manual_seed(20260916)
    q = (torch.randn(sq, 4, 576, device="cuda") * 0.05).to(torch.bfloat16)
    kv = (torch.randn(2048, 1, 576, device="cuda") * 0.05).to(
        torch.bfloat16
    )
    indices = torch.arange(2048, device="cuda", dtype=torch.int32)
    indices = indices.view(1, 1, 2048).expand(sq, 1, 2048).contiguous()
    lengths = torch.linspace(1, 2048, sq, device="cuda").to(torch.int32)
    sinks = torch.randn(4, device="cuda", dtype=torch.float32)
    return q, kv, indices, lengths, sinks


def _inputs(heads: int):
    torch.manual_seed(20260907)
    q = (torch.randn(2, heads, 576, device="cuda") * 0.05).to(torch.bfloat16)
    kv = (torch.randn(256, 1, 576, device="cuda") * 0.05).to(torch.bfloat16)
    indices = torch.arange(128, device="cuda", dtype=torch.int32)
    indices = indices.view(1, 1, 128).expand(2, 1, 128).contiguous()
    lengths = torch.tensor([96, 127], device="cuda", dtype=torch.int32)
    sinks = torch.randn(heads, device="cuda", dtype=torch.float32)
    return q, kv, indices, lengths, sinks


def _reference(q, kv, indices, lengths, sinks, scale):
    outputs, maxima, lses = [], [], []
    for token in range(q.shape[0]):
        ids = indices[token, 0, : lengths[token]].long()
        keys = kv[ids, 0].float()
        scores = torch.einsum("hd,kd->hk", q[token].float(), keys) * scale
        maximum = scores.max(dim=-1).values
        lse = torch.logsumexp(scores, dim=-1)
        weights = torch.exp(scores - maximum[:, None])
        denominator = weights.sum(dim=-1) + torch.exp(sinks - maximum)
        outputs.append(
            (torch.einsum("hk,kd->hd", weights, keys[:, :512]) / denominator[:, None])
            .to(torch.bfloat16)
        )
        maxima.append(maximum)
        lses.append(lse)
    return torch.stack(outputs), torch.stack(maxima), torch.stack(lses)


@pytest.mark.parametrize("heads", [4, 64])
def test_flash_mla_sparse_supports_small_head_counts(heads):
    q, kv, indices, lengths, sinks = _inputs(heads)
    scale = 576**-0.5
    expected = _reference(q, kv, indices, lengths, sinks, scale)
    actual = flag_gems.flash_mla_sparse_fwd(
        q, kv, indices, scale, attn_sink=sinks, topk_length=lengths
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(actual[0], expected[0], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual[1], expected[1], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual[2], expected[2], atol=2e-2, rtol=2e-2)


def test_flash_mla_sparse_small_heads_graph_replay_changed_input():
    q, kv, indices, lengths, sinks = _inputs(4)
    scale = 576**-0.5
    flag_gems.flash_mla_sparse_fwd(
        q, kv, indices, scale, attn_sink=sinks, topk_length=lengths
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = flag_gems.flash_mla_sparse_fwd(
            q, kv, indices, scale, attn_sink=sinks, topk_length=lengths
        )
    q.add_(0.01)
    expected = _reference(q, kv, indices, lengths, sinks, scale)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual[0], expected[0], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual[1], expected[1], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual[2], expected[2], atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="T-Head only")
def test_flash_mla_sparse_thead_splitk_boundaries():
    from flag_gems.fused.flashmla_sparse import (
        flash_mla_sparse_fwd as generic_flash_mla_sparse_fwd,
    )

    q, kv, indices, lengths, sinks = _thead_splitk_inputs()
    scale = 576**-0.5
    expected = generic_flash_mla_sparse_fwd(
        q, kv, indices, scale, attn_sink=sinks, topk_length=lengths
    )
    actual = flag_gems.flash_mla_sparse_fwd(
        q, kv, indices, scale, attn_sink=sinks, topk_length=lengths
    )
    torch.cuda.synchronize()

    assert flag_gems.flash_mla_sparse_fwd.__module__.endswith(
        "_thead.fused.flashmla_sparse"
    )
    for value in actual:
        assert not torch.isnan(value).any()
    torch.testing.assert_close(actual[0], expected[0], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual[1], expected[1], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual[2], expected[2], atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="T-Head only")
def test_flash_mla_sparse_thead_splitk_graph_replay():
    q, kv, indices, lengths, sinks = _thead_splitk_inputs()
    lengths.fill_(2048)
    scale = 576**-0.5
    flag_gems.flash_mla_sparse_fwd(
        q, kv, indices, scale, attn_sink=sinks, topk_length=lengths
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = flag_gems.flash_mla_sparse_fwd(
            q, kv, indices, scale, attn_sink=sinks, topk_length=lengths
        )
    graph.replay()
    graph.replay()
    q.add_(0.01)
    graph.replay()
    torch.cuda.synchronize()

    expected = _reference(q, kv, indices, lengths, sinks, scale)
    torch.testing.assert_close(actual[0], expected[0], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual[1], expected[1], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual[2], expected[2], atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("sq", [128, 2048])
@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="T-Head only")
def test_flash_mla_sparse_thead_hq4_prefill_range(sq):
    from flag_gems.fused.flashmla_sparse import (
        flash_mla_sparse_fwd as generic_flash_mla_sparse_fwd,
    )
    from flag_gems.runtime.backend._thead.fused.flashmla_sparse import (
        _can_use_thead_hq4_prefill,
    )

    q, kv, indices, lengths, sinks = _thead_hq4_prefill_inputs(sq)
    scale = 576**-0.5
    assert _can_use_thead_hq4_prefill(
        q, kv, indices, 512, sinks, lengths
    )
    expected = generic_flash_mla_sparse_fwd(
        q, kv, indices, scale, attn_sink=sinks, topk_length=lengths
    )
    actual = flag_gems.flash_mla_sparse_fwd(
        q, kv, indices, scale, attn_sink=sinks, topk_length=lengths
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual[0], expected[0], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual[1], expected[1], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual[2], expected[2], atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="T-Head only")
def test_flash_mla_sparse_thead_hq4_prefill_graph_replay():
    q, kv, indices, lengths, sinks = _thead_hq4_prefill_inputs(128)
    scale = 576**-0.5
    flag_gems.flash_mla_sparse_fwd(
        q, kv, indices, scale, attn_sink=sinks, topk_length=lengths
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = flag_gems.flash_mla_sparse_fwd(
            q, kv, indices, scale, attn_sink=sinks, topk_length=lengths
        )
    graph.replay()
    graph.replay()
    q.add_(0.01)
    graph.replay()
    torch.cuda.synchronize()

    from flag_gems.fused.flashmla_sparse import (
        flash_mla_sparse_fwd as generic_flash_mla_sparse_fwd,
    )

    expected = generic_flash_mla_sparse_fwd(
        q, kv, indices, scale, attn_sink=sinks, topk_length=lengths
    )
    torch.testing.assert_close(actual[0], expected[0], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual[1], expected[1], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual[2], expected[2], atol=2e-2, rtol=2e-2)
