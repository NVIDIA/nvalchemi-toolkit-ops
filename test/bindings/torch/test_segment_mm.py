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

"""PyTorch tensor, autograd, compilation, and CUDA stream contracts for segment_mm."""

import importlib

import pytest

torch = pytest.importorskip("torch")

from nvalchemiops.torch.segment_mm import segment_mm  # noqa: E402

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.fixture(autouse=True)
def _seed():
    """Keep numerical examples deterministic."""
    torch.manual_seed(27)


def _inputs(dtype=torch.float32, lengths=(3, 0, 4), capacity=10, strided=False):
    """Create framework-native inputs, optionally with noncontiguous storage."""
    if strided:
        a = torch.randn(7, capacity * 2, device="cuda", dtype=dtype).T[::2]
        b = torch.randn(len(lengths), 9, 14, device="cuda", dtype=dtype).transpose(
            1, 2
        )[:, ::2]
        seglen = torch.tensor(
            [v for length in lengths for v in (length, 91)],
            device="cuda",
            dtype=torch.int32,
        )[::2]
    else:
        a = torch.randn(capacity, 7, device="cuda", dtype=dtype)
        b = torch.randn(len(lengths), 7, 9, device="cuda", dtype=dtype)
        seglen = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    return a, b, seglen


def _reference(a, b, lengths, cotangent=None):
    """Compute independent dense products in FP64 and round to the native dtype."""
    out = torch.zeros((a.shape[0], b.shape[2]), device=a.device, dtype=a.dtype)
    da, db = torch.zeros_like(a), torch.zeros_like(b)
    start = 0
    for group, length in enumerate(lengths):
        stop = start + length
        aa, bb = a[start:stop].double(), b[group].double()
        out[start:stop] = (aa @ bb).to(a.dtype)
        if cotangent is not None:
            gg = cotangent[start:stop].double()
            da[start:stop] = (gg @ bb.T).to(a.dtype)
            db[group] = (aa.T @ gg).to(b.dtype)
        start = stop
    return out, da, db


def _close(actual, expected):
    """Compare native BF16 or FP32 outputs with accumulation-appropriate tolerance."""
    assert actual.dtype == expected.dtype
    tolerance = 2e-2 if actual.dtype == torch.bfloat16 else 2e-5
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


def _differentiable_reference(a, b, lengths):
    """Build a differentiable reference from independent dense segment products."""
    products = []
    start = 0
    for group, length in enumerate(lengths):
        products.append(a[start : start + length] @ b[group])
        start += length
    products.append(a.new_zeros((a.shape[0] - start, b.shape[2])))
    return torch.cat(products, dim=0)


def _energy(operation, positions, weights, lengths):
    """Use nonlinear features and energy to exercise all second-order paths."""
    features = torch.sin(positions) * 0.2
    output = operation(features, weights, lengths).float()
    return (torch.sin(output) + output.square() * 0.1).sum()


class TestSegmentMMAutocast:
    """Check CUDA BF16 computation with gradients to the original parameter dtype."""

    @pytest.mark.parametrize(
        "a_dtype,b_dtype",
        [
            (torch.float32, torch.float32),
            (torch.bfloat16, torch.float32),
            (torch.float32, torch.bfloat16),
            (torch.bfloat16, torch.bfloat16),
        ],
    )
    @pytest.mark.parametrize("compiled", [False, True])
    def test_forward_backward(self, a_dtype, b_dtype, compiled):
        """AMP casts both operands while backward restores their original dtypes."""
        a, b, lengths = _inputs(strided=True)
        a = a.to(a_dtype).requires_grad_()
        b = b.to(b_dtype).requires_grad_()
        cotangent = torch.randn(10, 9, device="cuda", dtype=torch.bfloat16)
        if compiled:
            torch.compiler.reset()
        operation = (
            torch.compile(segment_mm, fullgraph=True) if compiled else segment_mm
        )
        for values in ((3, 0, 4), (0, 5, 1), (0, 0, 0)):
            lengths.copy_(torch.tensor(values, device="cuda", dtype=torch.int32))
            with torch.no_grad():
                a.normal_()
                a[sum(values) :] = torch.nan
                cotangent.normal_()
                cotangent[sum(values) :] = torch.nan
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = operation(a, b, lengths)
            da, db = torch.autograd.grad(out, (a, b), cotangent)
            expected = _reference(
                a.to(torch.bfloat16), b.to(torch.bfloat16), values, cotangent
            )
            assert out.dtype == torch.bfloat16
            assert da.dtype == a_dtype
            assert db.dtype == b_dtype
            for actual, ref in zip((out, da, db), expected, strict=True):
                torch.testing.assert_close(
                    actual, ref.to(actual.dtype), atol=2e-2, rtol=2e-2
                )
            assert torch.count_nonzero(out[sum(values) :]) == 0
            assert torch.count_nonzero(da[sum(values) :]) == 0

    @pytest.mark.parametrize("compiled", [False, True])
    @pytest.mark.parametrize("derivative", ["jvp", "jvp_over_vjp"])
    def test_function_transforms(self, compiled, derivative):
        """Autocast preserves tangent and higher derivative paths to FP32 operands."""
        a, b, lengths = _inputs()
        tangents = (torch.randn_like(a), torch.randn_like(b))

        def evaluate(operation, a, b, ta, tb):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if derivative == "jvp":

                    def fn(a, b):
                        return operation(a, b, lengths)

                else:
                    fn = torch.func.grad(
                        lambda a, b: _energy(operation, a, b, lengths), (0, 1)
                    )
                return torch.func.jvp(fn, (a, b), (ta, tb))

        def reference(a, b, _):
            return _differentiable_reference(
                a.to(torch.bfloat16), b.to(torch.bfloat16), (3, 0, 4)
            )

        def actual(a, b, ta, tb):
            return evaluate(segment_mm, a, b, ta, tb)

        if compiled:
            torch.compiler.reset()
            actual = torch.compile(actual, fullgraph=True)
        result = actual(a, b, *tangents)
        expected = evaluate(reference, a, b, *tangents)
        for pair, ref_pair in zip(result, expected, strict=True):
            if derivative == "jvp":
                pair, ref_pair = (pair,), (ref_pair,)
            for value, ref in zip(pair, ref_pair, strict=True):
                assert value.dtype == ref.dtype
                torch.testing.assert_close(value, ref, atol=3e-2, rtol=3e-2)

    def test_disabled_context(self):
        """Nested disabled autocast preserves FP32 and rejects unmatched inputs."""
        a, b, lengths = _inputs()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.autocast("cuda", enabled=False):
                out = segment_mm(a, b, lengths)
                assert out.dtype == torch.float32
                _close(out, _reference(a, b, (3, 0, 4))[0])
                with pytest.raises(ValueError, match="same.*dtype"):
                    segment_mm(a.to(torch.bfloat16), b, lengths)
        with pytest.raises(ValueError, match="same.*dtype"):
            segment_mm(a.to(torch.bfloat16), b, lengths)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_fp16_autocast_rejected(self, dtype):
        """Unsupported FP16 autocast fails explicitly before launching a kernel."""
        a, b, lengths = _inputs(dtype)
        with torch.autocast("cuda", dtype=torch.float16):
            with pytest.raises(ValueError, match="autocast.*bfloat16"):
                segment_mm(a, b, lengths)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float64, torch.int32])
    @pytest.mark.parametrize("operand", [0, 1])
    def test_unsupported_input_dtype(self, dtype, operand):
        """Autocast does not silently accept unsupported input dtypes."""
        a, b, lengths = _inputs()
        inputs = [a, b, lengths]
        inputs[operand] = inputs[operand].to(dtype)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with pytest.raises(ValueError, match="dtype"):
                segment_mm(*inputs)


class TestSegmentMM:
    """Check public forward and first-order reverse-mode behavior."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("strided", [False, True])
    @pytest.mark.parametrize("requires", [(True, True), (True, False), (False, True)])
    def test_forward_backward(self, dtype, strided, requires):
        """Both and individual operand gradients preserve input dtype and strided reads."""
        a, b, lengths = _inputs(dtype, strided=strided)
        a[7:] = torch.nan
        a.requires_grad_(requires[0])
        b.requires_grad_(requires[1])
        cotangent = torch.randn(9, 10, device="cuda", dtype=dtype).T
        cotangent[7:] = torch.nan
        saved = tuple(x.clone() for x in (a, b, lengths))
        out = segment_mm(a, b, lengths)
        expected, da, db = _reference(a, b, (3, 0, 4), cotangent)
        _close(out, expected)
        assert out.is_contiguous()
        assert out.data_ptr() != a.data_ptr()
        operands = tuple(x for x in (a, b) if x.requires_grad)
        gradients = torch.autograd.grad(out, operands, cotangent)
        wanted = tuple(
            g for g, required in zip((da, db), requires, strict=True) if required
        )
        for actual, reference in zip(gradients, wanted, strict=True):
            _close(actual, reference)
        for actual, previous in zip((a, b, lengths), saved, strict=True):
            torch.testing.assert_close(actual, previous, equal_nan=True)

    @pytest.mark.parametrize(
        "lengths,capacity", [((0, 0, 0), 10), ((0, 0), 0), ((), 10), ((), 0)]
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_empty(self, lengths, capacity, dtype):
        """Empty segments and buffers produce defined zero outputs and gradients."""
        a, b, seglen = _inputs(dtype, lengths, capacity)
        a.requires_grad_()
        b.requires_grad_()
        out = segment_mm(a, b, seglen)
        _close(out, torch.zeros_like(out))
        da, db = torch.autograd.grad(out, (a, b), torch.ones_like(out))
        _close(da, torch.zeros_like(a))
        _close(db, torch.zeros_like(b))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("requires", [(True, True), (True, False), (False, True)])
    def test_compile_runtime_lengths(self, dtype, requires):
        """One full graph handles changed GPU lengths and new input storage with autograd."""
        torch.compiler.reset()
        compiled_graphs = []

        def backend(graph, example_inputs):
            compiled_graphs.append(graph)
            return graph.forward

        compiled = torch.compile(segment_mm, backend=backend, fullgraph=True)
        for values in ((3, 0, 4), (0, 5, 1), (0, 0, 0)):
            a, b, seglen = _inputs(dtype, values)
            a.requires_grad_(requires[0])
            b.requires_grad_(requires[1])
            cotangent = torch.randn(10, 9, device="cuda", dtype=dtype)
            expected, da, db = _reference(a, b, values, cotangent)
            out = compiled(a, b, seglen)
            _close(out, expected)
            operands = tuple(x for x in (a, b) if x.requires_grad)
            gradients = torch.autograd.grad(out, operands, cotangent)
            wanted = tuple(
                g for g, required in zip((da, db), requires, strict=True) if required
            )
            for actual, reference in zip(gradients, wanted, strict=True):
                _close(actual, reference)
        assert len(compiled_graphs) == 1

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("requires", [(True, True), (True, False), (False, True)])
    def test_inductor_fullgraph_backward(self, dtype, requires):
        """Default compiled execution lowers the opaque forward and backward operators."""
        a, b, lengths = _inputs(dtype)
        a.requires_grad_(requires[0])
        b.requires_grad_(requires[1])
        torch.compiler.reset()
        compiled = torch.compile(segment_mm, fullgraph=True)
        out = compiled(a, b, lengths)
        cotangent = torch.randn_like(out)
        expected, da, db = _reference(a, b, (3, 0, 4), cotangent)
        _close(out, expected)
        operands = tuple(x for x in (a, b) if x.requires_grad)
        actual = torch.autograd.grad(out, operands, cotangent)
        wanted = tuple(
            g for g, required in zip((da, db), requires, strict=True) if required
        )
        for gradient, reference in zip(actual, wanted, strict=True):
            _close(gradient, reference)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("active", [(True, False), (False, True), (True, True)])
    def test_jvp(self, dtype, active):
        """Functional JVP propagates either operand and their sum with native dtypes."""
        a, b, lengths = _inputs(dtype, strided=True)
        tangents = tuple(
            torch.randn_like(x) if enabled else torch.zeros_like(x)
            for x, enabled in zip((a, b), active, strict=True)
        )
        actual = torch.func.jvp(
            lambda a, b: segment_mm(a, b, lengths), (a, b), tangents
        )
        expected = torch.func.jvp(
            lambda a, b: _differentiable_reference(a, b, (3, 0, 4)), (a, b), tangents
        )
        for value, ref in zip(actual, expected, strict=True):
            _close(value, ref)

    @pytest.mark.parametrize("operand", [0, 1])
    def test_forward_ad(self, operand):
        """Low-level forward AD preserves a single operand tangent."""
        a, b, lengths = _inputs()
        tangent = torch.randn_like((a, b)[operand])
        with torch.autograd.forward_ad.dual_level():
            inputs = [a, b]
            inputs[operand] = torch.autograd.forward_ad.make_dual(
                inputs[operand], tangent
            )
            actual = torch.autograd.forward_ad.unpack_dual(segment_mm(*inputs, lengths))
            expected = torch.autograd.forward_ad.unpack_dual(
                _differentiable_reference(*inputs, (3, 0, 4))
            )
            _close(actual.primal, expected.primal)
            assert actual.tangent is not None
            _close(actual.tangent, expected.tangent)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_force_loss_training(self, dtype):
        """A force loss differentiates through forces to positions and model weights."""
        positions, weights, lengths = _inputs(dtype)
        positions.requires_grad_()
        weights.requires_grad_()

        def evaluate(operation):
            energy = _energy(operation, positions, weights, lengths)
            forces = -torch.autograd.grad(energy, positions, create_graph=True)[0]
            loss = (forces.float() - 0.15).square().sum()
            grads = torch.autograd.grad(loss, (positions, weights))
            return forces, *grads

        actual = evaluate(segment_mm)
        expected = evaluate(lambda a, b, _: _differentiable_reference(a, b, (3, 0, 4)))
        assert expected[2].abs().max() > 0.1
        for value, ref in zip(actual, expected, strict=True):
            _close(value, ref)
        assert torch.count_nonzero(actual[0][7:]) == 0
        assert torch.count_nonzero(actual[2][1]) == 0

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_jvp_over_vjp(self, dtype):
        """Forward-over-reverse includes both operand and nonlinear cotangent variations."""
        a, b, lengths = _inputs(dtype)
        tangents = (torch.randn_like(a), torch.randn_like(b))

        def gradient(operation):
            return torch.func.grad(
                lambda a, b: _energy(operation, a, b, lengths), (0, 1)
            )

        actual = torch.func.jvp(gradient(segment_mm), (a, b), tangents)
        expected = torch.func.jvp(
            gradient(lambda a, b, _: _differentiable_reference(a, b, (3, 0, 4))),
            (a, b),
            tangents,
        )
        for actual_pair, expected_pair in zip(actual, expected, strict=True):
            for value, ref in zip(actual_pair, expected_pair, strict=True):
                _close(value, ref)

    @pytest.mark.parametrize("selected", [(0,), (1,), (0, 1)])
    def test_backward_cotangent_derivative(self, selected):
        """Second backward handles unused gradient outputs and a varying cotangent."""
        a, b, lengths = _inputs()
        a.requires_grad_()
        b.requires_grad_()
        g = torch.randn(10, 9, device="cuda", requires_grad=True)
        u, v = torch.randn_like(a), torch.randn_like(b)

        def evaluate(operation):
            da, db = torch.autograd.grad(
                operation(a, b, lengths), (a, b), g, create_graph=True
            )
            terms = ((da * u).sum(), (db * v).sum())
            return torch.autograd.grad(
                sum(terms[index] for index in selected), (a, b, g), allow_unused=True
            )

        actual = evaluate(segment_mm)
        expected = evaluate(lambda a, b, _: _differentiable_reference(a, b, (3, 0, 4)))
        for value, ref in zip(actual, expected, strict=True):
            if ref is None:
                assert value is None
            else:
                _close(value, ref)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("derivative", ["double_backward", "jvp"])
    def test_one_sided_weight_derivative_skips_unused_row_product(
        self, dtype, derivative, monkeypatch
    ):
        """A weight-only derivative avoids launching a discarded input product."""
        binding = importlib.import_module("nvalchemiops.torch.segment_mm")
        core = importlib.import_module("nvalchemiops.segment_mm")
        a, b, lengths = _inputs(dtype)
        g = torch.randn(10, 9, device="cuda", dtype=dtype)
        u = torch.randn_like(a)
        row_products = []
        launch = core.wp.launch

        def observe(kernel, *args, **kwargs):
            if any(
                kernel is product for product in core._segment_mm_overloads.values()
            ):
                row_products.append(kernel)
            return launch(kernel, *args, **kwargs)

        if derivative == "double_backward":
            a.requires_grad_()
            b.requires_grad_()
            g.requires_grad_()
            da = torch.autograd.grad(
                segment_mm(a, b, lengths), a, g, create_graph=True
            )[0]
            with monkeypatch.context() as patch:
                patch.setattr(core.wp, "launch", observe)
                actual = torch.autograd.grad(da, (g, b), u)
            ref_da = torch.autograd.grad(
                _differentiable_reference(a, b, (3, 0, 4)), a, g, create_graph=True
            )[0]
            expected = torch.autograd.grad(ref_da, (g, b), u)
            for value, ref in zip(actual, expected, strict=True):
                _close(value, ref)
        else:
            # Only A carries a tangent: d(dA)=0, d(dB)=U.T @ G. The
            # primal backward needs one row product; its JVP needs none.
            with torch.autograd.forward_ad.dual_level():
                dual_a = torch.autograd.forward_ad.make_dual(a, u)
                with monkeypatch.context() as patch:
                    patch.setattr(core.wp, "launch", observe)
                    dual_da, dual_db = binding._apply_backward(g, dual_a, b, lengths)
                actual_da = torch.autograd.forward_ad.unpack_dual(dual_da)
                actual_db = torch.autograd.forward_ad.unpack_dual(dual_db)
                _, ref_da, ref_db = _reference(a, b, (3, 0, 4), g)
                _, _, ref_tangent_db = _reference(u, b, (3, 0, 4), g)
                _close(actual_da.primal, ref_da)
                _close(actual_db.primal, ref_db)
                _close(actual_da.tangent, torch.zeros_like(a))
                _close(actual_db.tangent, ref_tangent_db)
        assert len(row_products) == 1, "launched a discarded input-gradient product"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_weight_product_higher_transforms(self, dtype):
        """The weight-only branch retains both reverse and forward derivatives."""
        binding = importlib.import_module("nvalchemiops.torch.segment_mm")
        a, b, lengths = _inputs(dtype, strided=True)
        g = torch.randn(10, 9, device="cuda", dtype=dtype)
        ta, tg = torch.randn_like(a), torch.randn_like(g)
        cotangent = torch.randn_like(b)

        def reference(a, g):
            return torch.stack((a[:3].T @ g[:3], a[3:3].T @ g[3:3], a[3:7].T @ g[3:7]))

        def evaluate(operation):
            def loss(a, g):
                product = operation(a, g)
                return (product * cotangent).float().sum()

            return torch.func.jvp(torch.func.grad(loss, (0, 1)), (a, g), (ta, tg))

        actual = evaluate(lambda a, g: binding._apply_weight_product(a, g, lengths))
        expected = evaluate(reference)
        for pair, ref_pair in zip(actual, expected, strict=True):
            for value, ref in zip(pair, ref_pair, strict=True):
                _close(value, ref)
        _, actual_tangent = torch.func.jvp(
            lambda a, g: binding._apply_weight_product(a, g, lengths),
            (a, g),
            (ta, tg),
        )
        _, expected_tangent = torch.func.jvp(reference, (a, g), (ta, tg))
        _close(actual_tangent, expected_tangent)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_compiled_force_loss(self, dtype):
        """A full compiled force-loss gradient follows changing runtime segment lengths."""
        a, b, lengths = _inputs(dtype)

        def make_gradient(operation):
            def force_loss(a, b, lengths):
                forces = -torch.func.grad(lambda a: _energy(operation, a, b, lengths))(
                    a
                )
                return (forces.float() - 0.15).square().sum()

            return torch.func.grad(force_loss, (0, 1))

        torch.compiler.reset()
        compiled = torch.compile(make_gradient(segment_mm), fullgraph=True)
        for values in ((3, 0, 4), (0, 5, 1), (0, 0, 0)):
            lengths.copy_(torch.tensor(values, dtype=torch.int32, device="cuda"))
            expected = make_gradient(
                lambda a, b, _: _differentiable_reference(a, b, values)
            )(a, b, lengths)
            actual = compiled(a, b, lengths)
            for value, ref in zip(actual, expected, strict=True):
                _close(value, ref)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_compiled_jvp(self, dtype):
        """Compiled forward AD retains both tangents with runtime segment lengths."""
        a, b, lengths = _inputs(dtype)
        tangent_a, tangent_b = torch.randn_like(a), torch.randn_like(b)

        def jvp(a, b, lengths, tangent_a, tangent_b):
            return torch.func.jvp(
                lambda a, b: segment_mm(a, b, lengths),
                (a, b),
                (tangent_a, tangent_b),
            )

        torch.compiler.reset()
        compiled = torch.compile(jvp, fullgraph=True)
        for values in ((3, 0, 4), (0, 5, 1), (0, 0, 0)):
            lengths.copy_(torch.tensor(values, dtype=torch.int32, device="cuda"))
            expected = torch.func.jvp(
                lambda a, b: _differentiable_reference(a, b, values),
                (a, b),
                (tangent_a, tangent_b),
            )
            actual = compiled(a, b, lengths, tangent_a, tangent_b)
            for value, ref in zip(actual, expected, strict=True):
                _close(value, ref)

    def test_opcheck(self):
        """Internal execution ops satisfy alias, fake, and AOT metadata contracts."""
        a, b, lengths = _inputs(strided=True)
        # Differentiation is owned and tested through the public Function API.
        torch.library.opcheck(
            torch.ops.nvalchemiops.segment_mm.default,
            (a, b, lengths),
            test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
        )

    def test_nondefault_stream(self, torch_stream_runner):
        """Forward and backward consume delayed producers on the current Torch stream."""
        a, b, lengths = _inputs()
        g = torch.randn(10, 9, device="cuda")
        target = tuple(torch.zeros_like(value) for value in (a, b, lengths, g))

        def operation(inputs):
            aa, bb, ll, gg = inputs
            aa = aa.detach().requires_grad_()
            bb = bb.detach().requires_grad_()
            out = segment_mm(aa, bb, ll)
            return (out, *torch.autograd.grad(out, (aa, bb), gg))

        _, actual, expected = torch_stream_runner((a, b, lengths, g), target, operation)
        for value, reference in zip(actual, expected, strict=True):
            _close(value, reference)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("strided", [False, True])
    def test_cuda_graph_replay(self, dtype, strided):
        """Captured forward/backward scratch remains valid with changed values and lengths."""
        a, b, lengths = _inputs(dtype, strided=strided)
        a.requires_grad_()
        b.requires_grad_()
        g = torch.randn(9, 10, device="cuda", dtype=dtype).T
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                out = segment_mm(a, b, lengths)
                torch.autograd.grad(out, (a, b), g)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = segment_mm(a, b, lengths)
            duplicate = segment_mm(a, b, lengths)
            da, db = torch.autograd.grad(out, (a, b), g)
        for values in ((3, 0, 4), (1, 5, 0), (0, 0, 0)):
            with torch.no_grad():
                a.normal_()
                b.normal_()
                g.normal_()
                lengths.copy_(torch.tensor(values, device="cuda", dtype=torch.int32))
            graph.replay()
            expected = _reference(a, b, values, g)
            _close(duplicate, expected[0])
            for value, reference in zip((out, da, db), expected, strict=True):
                _close(value, reference)

    @pytest.mark.parametrize(
        "case,match",
        [
            ("a_rank", "a.*2"),
            ("b_rank", "b.*3"),
            ("seglen_rank", "seglen.*1"),
            ("dtype", "dtype"),
            ("length_dtype", "int32"),
            ("channels", "channel"),
            ("groups", "length"),
            ("zero_k", "positive"),
            ("zero_n", "positive"),
            ("device", "same.*device"),
        ],
    )
    def test_validation(self, case, match):
        """Invalid metadata fails before the raw GPU launch without inspecting values."""
        a, b, lengths = _inputs()
        if case == "a_rank":
            a = a.unsqueeze(0)
        elif case == "b_rank":
            b = b[0]
        elif case == "seglen_rank":
            lengths = lengths.unsqueeze(0)
        elif case == "dtype":
            b = b.double()
        elif case == "length_dtype":
            lengths = lengths.long()
        elif case == "channels":
            b = b[:, :-1]
        elif case == "groups":
            lengths = lengths[:-1]
        elif case == "zero_k":
            a, b = a[:, :0], b[:, :0]
        elif case == "zero_n":
            b = b[:, :, :0]
        elif case == "device":
            lengths = lengths.cpu()
        with pytest.raises(ValueError, match=match):
            segment_mm(a, b, lengths)

    def test_cpu_rejected(self):
        """The CUDA-only contract reports unsupported CPU execution clearly."""
        a, b, lengths = _inputs()
        with pytest.raises(RuntimeError, match="CUDA"):
            segment_mm(a.cpu(), b.cpu(), lengths.cpu())
