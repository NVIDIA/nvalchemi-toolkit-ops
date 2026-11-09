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

"""External CUDA capture of scan scratch, split gradients, and force losses."""

import pytest

torch = pytest.importorskip("torch")

from nvalchemiops.torch.segment_mm import segment_mm  # noqa: E402

from .test_segment_mm import (  # noqa: E402
    _close,
    _differentiable_reference,
    _energy,
    _reference,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def _length_cases(groups):
    """Alternate active and empty splits, including the last scanned segment."""
    if groups == 1:
        return ((1007,), (0,), (33,), (1019,))
    return (
        (2049,) + (0,) * 63 + (31,) + (0,) * 63 + (1025,),
        (0,) * groups,
        (0,) * 64 + (769,) + (0,) * 63 + (33,),
        (512,) + (0,) * 126 + (1023, 17),
    )


def _capture(operation, capture_error_mode="global"):
    """Warm every derivative kernel on a side stream before external capture."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            operation()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream, capture_error_mode=capture_error_mode):
        outputs = operation()
    return graph, outputs


class TestSegmentMMCapture:
    """Replay captured scratch-dependent paths against independent dense products."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("capacity,groups", [(1024, 1), (40000, 129)])
    @pytest.mark.parametrize(
        "capture_error_mode", ["global", "thread_local", "relaxed"]
    )
    def test_split_gradients_and_prefix_scan(
        self, dtype, capacity, groups, capture_error_mode
    ):
        """Replay must refresh prefix scans and every partial weight-gradient buffer."""
        generator = torch.Generator(device="cuda").manual_seed(712)
        a = torch.randn(capacity, 7, device="cuda", dtype=dtype, generator=generator)
        b = torch.randn(groups, 7, 5, device="cuda", dtype=dtype, generator=generator)
        g = torch.randn(capacity, 5, device="cuda", dtype=dtype, generator=generator)
        a.requires_grad_()
        b.requires_grad_()
        cases = _length_cases(groups)
        lengths = torch.tensor(cases[0], device="cuda", dtype=torch.int32)

        def evaluate():
            out = segment_mm(a, b, lengths)
            return out, *torch.autograd.grad(out, (a, b), g)

        # Both shapes split dB; 129 groups also exercises the cooperative prefix.
        graph, actual = _capture(evaluate, capture_error_mode)
        for values in cases:
            used = sum(values)
            with torch.no_grad():
                a.normal_(0, 0.2, generator=generator)
                b.normal_(0, 0.2, generator=generator)
                g.normal_(0, 0.2, generator=generator)
                a[used:] = torch.nan
                g[used:] = torch.nan
                lengths.copy_(torch.tensor(values, device="cuda", dtype=torch.int32))
            graph.replay()
            expected = _reference(a, b, values, g)
            for value, reference in zip(actual, expected, strict=True):
                _close(value, reference)
                assert torch.isfinite(value).all()
            assert torch.count_nonzero(actual[0][used:]) == 0
            assert torch.count_nonzero(actual[1][used:]) == 0
            empty = torch.tensor(values, device="cuda") == 0
            assert torch.count_nonzero(actual[2][empty]) == 0
            if used:
                assert expected[2].abs().max() > 0.1

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_force_loss_double_backward(self, dtype):
        """Captured force-loss gradients refresh after empty and short-row replays."""
        generator = torch.Generator(device="cuda").manual_seed(913)
        positions = torch.randn(
            1024, 7, device="cuda", dtype=dtype, generator=generator, requires_grad=True
        )
        weights = torch.randn(
            1, 7, 5, device="cuda", dtype=dtype, generator=generator, requires_grad=True
        )
        target = torch.zeros(1024, 7, device="cuda")
        cases = _length_cases(1)
        lengths = torch.tensor(cases[0], device="cuda", dtype=torch.int32)

        def evaluate(operation, positions, weights):
            energy = _energy(operation, positions, weights, lengths)
            forces = -torch.autograd.grad(energy, positions, create_graph=True)[0]
            loss = (forces.float() - target).square().sum()
            return forces, loss, *torch.autograd.grad(loss, (positions, weights))

        graph, actual = _capture(lambda: evaluate(segment_mm, positions, weights))
        tolerance = 3e-2 if dtype == torch.bfloat16 else 1e-4
        for values in cases:
            used = sum(values)
            with torch.no_grad():
                positions.normal_(0, 0.5, generator=generator)
                weights.normal_(0, 0.3, generator=generator)
                target.normal_(0, 0.1, generator=generator)
                target[used:] = 0
                lengths.copy_(torch.tensor(values, device="cuda", dtype=torch.int32))
            graph.replay()
            expected = evaluate(
                lambda a, b, _: _differentiable_reference(a, b, values),
                positions.detach().clone().requires_grad_(),
                weights.detach().clone().requires_grad_(),
            )
            for value, reference in zip(actual, expected, strict=True):
                assert value.dtype == reference.dtype
                torch.testing.assert_close(
                    value, reference, rtol=tolerance, atol=tolerance
                )
                assert torch.isfinite(value).all()
            assert torch.count_nonzero(actual[0][used:]) == 0
            assert torch.count_nonzero(actual[2][used:]) == 0
            if used:
                assert expected[3].abs().max() > 0.1
            else:
                assert torch.count_nonzero(actual[3]) == 0
