# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU dispatch and analytical derivatives for the shared SegMM implementation."""

import pytest

torch = pytest.importorskip("torch")

from nvalchemiops.torch._warp_op_helpers import (  # noqa: E402
    scoped_warp_stream,
)
from nvalchemiops.torch.segment_mm import segment_mm  # noqa: E402


def _reference(a, b, lengths):
    """Compute independent dense products with FP32 accumulation."""
    products = []
    start = 0
    for group, length in enumerate(lengths):
        stop = min(start + max(length, 0), a.shape[0])
        products.append((a[start:stop].float() @ b[group].float()).to(a.dtype))
        start = stop
    products.append(a.new_zeros((a.shape[0] - start, b.shape[2])))
    return torch.cat(products)


def _inputs(dtype, strided=False, lengths=(3, 0, 2), capacity=8):
    """Use a local seed and optional noncontiguous CPU storage."""
    generator = torch.Generator().manual_seed(73)
    a = torch.randn(capacity, 4, generator=generator, dtype=dtype)
    b = torch.randn(len(lengths), 4, 5, generator=generator, dtype=dtype)
    if strided:
        a = a.T.contiguous().T
        b = b.transpose(1, 2).contiguous().transpose(1, 2)
    return a, b, torch.tensor(lengths, dtype=torch.int32)


def _close(actual, expected):
    """Compare native products without demanding identical reduction order."""
    tolerance = 3e-2 if actual.dtype == torch.bfloat16 else 3e-5
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


class TestSegmentMMCPU:
    """Exercise CPU custom-op dispatch without a CUDA availability requirement."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("compiled", [False, True])
    @pytest.mark.parametrize("strided", [False, True])
    def test_forward_backward(self, dtype, compiled, strided):
        """CPU operators and AOT autograd preserve native dtype and runtime lengths."""
        a, b, lengths = _inputs(dtype, strided)
        a.requires_grad_()
        b.requires_grad_()
        if compiled:
            torch.compiler.reset()
        operation = (
            torch.compile(segment_mm, backend="aot_eager", fullgraph=True)
            if compiled
            else segment_mm
        )
        g = torch.ones(8, 5, dtype=dtype)
        for values in ((3, 0, 2), (0, 4, 1), (0, 0, 0)):
            lengths.copy_(torch.tensor(values, dtype=torch.int32))
            with torch.no_grad():
                a[sum(values) :] = torch.nan
                g[sum(values) :] = torch.nan
            out = operation(a, b, lengths)
            da, db = torch.autograd.grad(out, (a, b), g)
            # Mask padding before dense multiplication: even a NaN cotangent
            # on inactive rows must not affect any active gradient.
            expected_out = _reference(a, b, values)
            expected_da, expected_db = torch.autograd.grad(expected_out, (a, b), g)
            for actual, expected in zip(
                (out, da, db), (expected_out, expected_da, expected_db), strict=True
            ):
                _close(actual, expected)
            assert torch.count_nonzero(out[sum(values) :]) == 0
            assert torch.count_nonzero(da[sum(values) :]) == 0

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("length_dtype", [torch.int32, torch.int64])
    def test_higher_derivatives(self, dtype, length_dtype):
        """CPU JVP-over-VJP and force-loss gradients use the analytical rules."""
        a, b, lengths = _inputs(dtype)
        lengths = lengths.to(length_dtype)

        def evaluate(operation):
            def energy(a, b):
                y = operation(a, b, lengths).float()
                return (torch.sin(y) + 0.1 * y.square()).sum()

            def force_loss(a, b):
                forces = -torch.func.grad(energy)(a, b)
                return (forces.float() - 0.15).square().sum()

            gradient = torch.func.grad(energy, (0, 1))
            first, tangent = torch.func.jvp(
                gradient, (a, b), (torch.ones_like(a), torch.ones_like(b))
            )
            return (*first, *tangent, *torch.func.grad(force_loss, (0, 1))(a, b))

        actual = evaluate(segment_mm)
        expected = evaluate(lambda a, b, _: _reference(a, b, (3, 0, 2)))
        for value, ref in zip(actual, expected, strict=True):
            _close(value, ref)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("lengths,capacity", [((), 0), ((), 4), ((0, 0), 0)])
    def test_empty(self, dtype, lengths, capacity):
        """CPU empty shapes overwrite outputs and return zero operand gradients."""
        a, b, metadata = _inputs(dtype, lengths=lengths, capacity=capacity)
        a.requires_grad_()
        b.requires_grad_()
        out = segment_mm(a, b, metadata)
        da, db = torch.autograd.grad(out, (a, b), torch.ones_like(out))
        assert out.shape == (capacity, 5)
        for value in (out, da, db):
            assert torch.count_nonzero(value) == 0

    def test_cpu_scopes_are_noops(self, monkeypatch):
        """CPU execution never queries CUDA streams or Warp capture state."""
        import warp as wp

        from nvalchemiops.torch._warp_op_helpers import scoped_warp_capture

        def unexpected(*args, **kwargs):
            raise AssertionError("CPU scope queried a CUDA stream")

        monkeypatch.setattr(torch.cuda, "current_stream", unexpected)
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", unexpected)
        monkeypatch.setattr(wp, "get_stream", unexpected)
        with scoped_warp_stream("cpu"), scoped_warp_capture("cpu"):
            assert torch.ones(1).device.type == "cpu"
        a, b, lengths = _inputs(torch.float32)
        _close(segment_mm(a, b, lengths), _reference(a, b, (3, 0, 2)))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_cpu_autocast_uses_explicit_dtype(self, dtype):
        """CPU autocast leaves the explicit SegMM operand dtype in control."""
        a, b, lengths = _inputs(dtype)
        expected = _reference(a, b, (3, 0, 2))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = segment_mm(a, b, lengths)
        _close(actual, expected)
