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

"""Opt-in segment-length validation and integer dtype normalization."""

import functools
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from nvalchemiops.torch.segment_mm import segment_mm  # noqa: E402
from test.bindings.torch.test_segment_mm_cpu import (  # noqa: E402
    _close,
    _inputs,
    _reference,
)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=(
                pytest.mark.gpu,
                pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA required"
                ),
            ),
        ),
    ],
)
class TestSegmentMMLengths:
    """Both validation modes preserve shapes, native products, and derivatives."""

    @pytest.mark.parametrize("length_dtype", [torch.int32, torch.int64])
    @pytest.mark.parametrize("validate", [False, True])
    def test_fake_tensor_shapes(self, device, length_dtype, validate, monkeypatch):
        """Fake tensors propagate metadata without value reads or capture queries."""
        from torch._subclasses.fake_tensor import FakeTensorMode, is_fake

        def unexpected(*args, **kwargs):
            raise AssertionError(
                "Fake tensor validation read values or queried capture"
            )

        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", unexpected)
        monkeypatch.setattr(torch.Tensor, "item", unexpected)
        with FakeTensorMode():
            a = torch.empty((8, 2), device=device)
            b = torch.empty((3, 2, 4), device=device)
            lengths = torch.empty((3,), dtype=length_dtype, device=device)
            out = segment_mm(a, b, lengths, validate=validate)
            assert is_fake(out)
            assert out.shape == (8, 4)
            assert out.dtype == a.dtype
            assert out.device == a.device

    @pytest.mark.parametrize("validate", [False, True])
    def test_metadata_always_checked(self, device, validate):
        """Disabling value checks does not disable the tensor shape contract."""
        a = torch.ones((8, 2), device=device)
        b = torch.ones((3, 4, 2), device=device)
        lengths = torch.tensor([3, 0, 2], device=device, dtype=torch.int64)
        with pytest.raises(ValueError, match="channel dimensions.*match"):
            segment_mm(a, b, lengths, validate=validate)

    def test_default_has_no_scalar_reads_or_assertions(self, device, monkeypatch):
        """Default eager execution neither reads lengths on the host nor asserts."""
        a, b, lengths = _inputs(torch.float32)
        a, b, lengths = (value.to(device) for value in (a, b, lengths))
        lengths = lengths.to(torch.int64)

        def unexpected(*args, **kwargs):
            raise AssertionError("Unchecked SegMM performed value validation")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "item", unexpected)
            patch.setattr(torch, "_assert_async", unexpected)
            out = segment_mm(a, b, lengths)
        _close(out, _reference(a, b, (3, 0, 2)))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("compiled", [False, True])
    @pytest.mark.parametrize("validate", [False, True])
    def test_int64_forward_backward(self, device, dtype, compiled, validate):
        """Native products and gradients follow changed strided int64 lengths."""
        a, b, _ = _inputs(dtype, strided=True)
        a = a.to(device).detach().requires_grad_()
        b = b.to(device).detach().requires_grad_()
        lengths = torch.tensor([3, 91, 0, 91, 2, 91], device=device)[::2]
        g = torch.ones((8, 5), device=device, dtype=dtype)
        operation = functools.partial(segment_mm, validate=validate)
        if compiled:
            torch.compiler.reset()
            operation = torch.compile(operation, backend="aot_eager", fullgraph=True)
        for values in ((3, 0, 2), (0, 4, 1), (0, 0, 0)):
            lengths.copy_(torch.tensor(values, device=device))
            out = operation(a, b, lengths)
            expected = _reference(a, b, values)
            actual_grads = torch.autograd.grad(out, (a, b), g)
            expected_grads = torch.autograd.grad(expected, (a, b), g)
            _close(out, expected)
            for actual, reference in zip(actual_grads, expected_grads, strict=True):
                _close(actual, reference)
            assert lengths.dtype == torch.int64
            torch.testing.assert_close(lengths, torch.tensor(values, device=device))

    @pytest.mark.parametrize(
        "values,capacity", [((), 0), ((), 8), ((0, 0, 0), 0), ((0, 0, 0), 8)]
    )
    @pytest.mark.parametrize("validate", [False, True])
    def test_int64_empty_lengths(self, device, values, capacity, validate):
        """Checked and unchecked empty shapes return defined zero outputs."""
        a = torch.ones((capacity, 2), device=device)
        b = torch.ones((len(values), 2, 3), device=device)
        lengths = torch.tensor(values, dtype=torch.int64, device=device)
        out = segment_mm(a, b, lengths, validate=validate)
        assert out.shape == (capacity, 3)
        assert torch.count_nonzero(out) == 0

    @pytest.mark.parametrize("compiled", [False, True])
    @pytest.mark.parametrize("derivative", ["jvp_over_vjp", "force_loss"])
    def test_checked_higher_derivatives(self, device, compiled, derivative):
        """Checked execution composes with JVP-over-VJP and force-loss gradients."""
        a, b, lengths = _inputs(torch.float32)
        a, b = a.to(device), b.to(device)
        lengths = lengths.to(device=device, dtype=torch.int64)

        def make_derivative(operation):
            def energy(a, b):
                y = operation(a, b, lengths)
                return (torch.sin(y) + 0.1 * y.square()).sum()

            if derivative == "jvp_over_vjp":

                def evaluate(a, b):
                    first, tangent = torch.func.jvp(
                        torch.func.grad(energy, (0, 1)),
                        (a, b),
                        (torch.ones_like(a), torch.ones_like(b)),
                    )
                    return *first, *tangent

                return evaluate

            def force_loss(a, b):
                forces = -torch.func.grad(energy)(a, b)
                return (forces - 0.15).square().sum()

            return torch.func.grad(force_loss, (0, 1))

        operation = make_derivative(functools.partial(segment_mm, validate=True))
        if compiled:
            torch.compiler.reset()
            operation = torch.compile(operation, backend="aot_eager", fullgraph=True)
        actual = operation(a, b)
        expected = make_derivative(lambda a, b, _: _reference(a, b, (3, 0, 2)))(a, b)
        for value, reference in zip(actual, expected, strict=True):
            _close(value, reference)


class TestSegmentMMValidationOption:
    """Checked CPU execution reports recoverable errors, including in full graphs."""

    def test_checked_cpu_keyword(self):
        """The checked API accepts valid lengths and rejects a later bad value."""
        a, b, lengths = _inputs(torch.float32)
        out = segment_mm(a, b, lengths, validate=True)
        _close(out, _reference(a, b, (3, 0, 2)))
        lengths[0] = -1
        with pytest.raises(ValueError, match="nonnegative"):
            segment_mm(a, b, lengths, validate=True)

    @pytest.mark.parametrize("compiled", [False, True])
    @pytest.mark.parametrize("length_dtype", [torch.int32, torch.int64])
    @pytest.mark.parametrize(
        "values,capacity,match",
        [
            ((-1, 2, 3), 8, "nonnegative"),
            ((4, 4, 4), 8, "sum"),
            ((9, 0, 0), 8, "sum"),
            ((2147483647, 2147483647, 0), 8, "sum"),
            ((0, 1, 0), 0, "sum"),
        ],
    )
    def test_cpu_invalid_lengths(self, compiled, length_dtype, values, capacity, match):
        """CPU validation runs on changed runtime values inside compiled opaque ops."""
        a = torch.ones((capacity, 2))
        b = torch.ones((len(values), 2, 2))
        lengths = torch.zeros(len(values), dtype=length_dtype)
        operation = functools.partial(segment_mm, validate=True)
        if compiled:
            torch.compiler.reset()
            operation = torch.compile(operation, backend="aot_eager", fullgraph=True)
        operation(a, b, lengths)
        lengths.copy_(torch.tensor(values, dtype=length_dtype))
        with pytest.raises(ValueError, match=match):
            operation(a, b, lengths)

    @pytest.mark.parametrize("compiled", [False, True])
    @pytest.mark.parametrize(
        "values",
        [(2**32 + 3, 0, 2), (2**63 - 1, 2**63 - 1, 2), (2**31, 0, 0)],
    )
    def test_cpu_int64_overflow(self, compiled, values):
        """Validation happens before narrowing or an overflowing int64 sum."""
        a, b, _ = _inputs(torch.float32)
        lengths = torch.zeros(3, dtype=torch.int64)
        operation = functools.partial(segment_mm, validate=True)
        if compiled:
            torch.compiler.reset()
            operation = torch.compile(operation, backend="aot_eager", fullgraph=True)
        operation(a, b, lengths)
        lengths.copy_(torch.tensor(values))
        with pytest.raises(ValueError, match="representable.*int32"):
            operation(a, b, lengths)


def _invalid_cuda_child(mode, values, capacity, dtype_name):
    """Exercise one invalid runtime input in a disposable CUDA context."""
    a = torch.ones((capacity, 2), device="cuda")
    b = torch.ones((len(values), 2, 3), device="cuda")
    lengths = torch.zeros(len(values), dtype=getattr(torch, dtype_name), device="cuda")
    changed = torch.tensor(values, dtype=lengths.dtype, device="cuda")
    operation = functools.partial(segment_mm, validate=True)
    if mode == "compile":
        operation = torch.compile(operation, fullgraph=True)
    if mode == "graph":
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                output = operation(a, b, lengths)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = operation(a, b, lengths)
        graph.replay()
    else:
        output = operation(a, b, lengths)
    torch.cuda.synchronize()
    lengths.copy_(changed)
    torch.cuda.synchronize()
    try:
        if mode == "graph":
            graph.replay()
        else:
            output = operation(a, b, lengths)
        torch.cuda.synchronize()
    except RuntimeError as error:
        assert "device-side assert" in str(error).lower(), str(error)
        print("checked CUDA validation rejected runtime lengths", flush=True)
        # CUDA assertions poison the context; avoid cleanup operations on it.
        os._exit(0)
    raise AssertionError(f"Invalid lengths were accepted in {mode}: {output.shape}")


def _mixed_validation_capture_child():
    """Keep an unchecked graph alive across the first checked kernel launches."""
    import nvalchemiops.batch_utils as batch_utils

    kernels = {
        name: getattr(batch_utils, name)
        for name in (
            "_segment_offsets_small",
            "_segment_offsets_small_fused",
            "_segment_offsets_small_checked",
            "_segment_offsets_small_fused_checked",
            "_segment_counts_checked",
        )
    }
    for name in (
        "_segment_offsets_cooperative",
        "_segment_offsets_cooperative_checked",
    ):
        kernels.update(
            (f"{name}[{bucket}]", kernel)
            for bucket, kernel in getattr(batch_utils, name).items()
        )

    def overloads():
        return {
            name: tuple(sorted(kernel.overloads)) for name, kernel in kernels.items()
        }

    a = torch.ones((8, 4), device="cuda", requires_grad=True)
    b = torch.arange(1, 4, device="cuda", dtype=torch.float32)
    b = b[:, None, None].expand(-1, 4, 5).contiguous().requires_grad_()
    lengths = torch.tensor([3, 0, 2], device="cuda", dtype=torch.int32)
    g = torch.ones((8, 5), device="cuda")
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
        da, db = torch.autograd.grad(out, (a, b), g)
    torch.cuda.synchronize()
    initial_overloads = overloads()
    module = batch_utils._segment_offsets_small.module
    initial_hash = module.get_module_hash().hex()
    print(
        json.dumps(
            {"captured_module_hash": initial_hash, "executions": len(module.execs)}
        ),
        flush=True,
    )

    for length_dtype in (torch.int64, torch.int32):
        for groups, capacity in ((3, 8), (64, 2048), (129, 2048)):
            other_a = torch.ones((capacity, 4), device="cuda", requires_grad=True)
            other_b = torch.ones((groups, 4, 5), device="cuda", requires_grad=True)
            other_lengths = torch.zeros(groups, device="cuda", dtype=length_dtype)
            other_lengths[0] = min(capacity, 1536)
            other = segment_mm(other_a, other_b, other_lengths, validate=True)
            torch.autograd.grad(other, (other_a, other_b), torch.ones_like(other))
            torch.cuda.synchronize()

    lengths.copy_(torch.tensor([1, 0, 4], device="cuda", dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()
    a_ref = a.detach().requires_grad_()
    b_ref = b.detach().requires_grad_()
    expected = _reference(a_ref, b_ref, (1, 0, 4))
    expected_da, expected_db = torch.autograd.grad(expected, (a_ref, b_ref), g)
    for actual, reference in zip(
        (out, da, db), (expected, expected_da, expected_db), strict=True
    ):
        _close(actual, reference)
    final_overloads = overloads()
    changes = {
        name: (initial_overloads[name], signatures)
        for name, signatures in final_overloads.items()
        if initial_overloads[name] != signatures
    }
    final_hash = module.get_module_hash().hex()
    print(
        json.dumps({"final_module_hash": final_hash, "executions": len(module.execs)}),
        flush=True,
    )
    assert not changes, f"Prefix kernels created overloads after capture: {changes}"
    assert final_hash == initial_hash, "Prefix module changed after capture"
    print("mixed validation graph replay preserved", flush=True)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestSegmentMMCUDAValidation:
    """Device assertions are tested in subprocesses to preserve pytest's context."""

    @pytest.mark.parametrize("mode", ["eager", "compile", "graph"])
    @pytest.mark.parametrize(
        "values,capacity,dtype_name",
        [
            pytest.param((-1, 2, 3), 8, "int32", id="negative"),
            pytest.param((4, 4, 4), 8, "int32", id="oversum"),
            pytest.param((2**32 + 3, 0, 2), 8, "int64", id="int64-wrap"),
            pytest.param(
                (2**63 - 1, 2**63 - 1, 2), 8, "int64", id="int64-sum-overflow"
            ),
            pytest.param((0, 1, 0), 0, "int64", id="empty-capacity"),
            pytest.param((0,) * 128 + (-1,), 8, "int64", id="scan-negative"),
        ],
    )
    def test_runtime_invalid_lengths(self, mode, values, capacity, dtype_name):
        """Checked GPU calls reject changed lengths in eager, compiled, and replayed paths."""
        root = Path(__file__).resolve().parents[3]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join((str(root), env.get("PYTHONPATH", "")))
        env.pop("CUDA_LAUNCH_BLOCKING", None)
        script = (
            "import json, sys; "
            "from test.bindings.torch.test_segment_mm_validation import _invalid_cuda_child; "
            "_invalid_cuda_child(*json.loads(sys.argv[1]))"
        )
        result = subprocess.run(  # noqa: S603 -- fixed interpreter and test helper
            [
                sys.executable,
                "-c",
                script,
                json.dumps((mode, values, capacity, dtype_name)),
            ],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            timeout=180,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "checked CUDA validation rejected runtime lengths" in result.stdout

    def test_graph_survives_first_checked_kernel_launches(self):
        """New validation schedules cannot replace modules used by captured graphs."""
        root = Path(__file__).resolve().parents[3]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join((str(root), env.get("PYTHONPATH", "")))
        script = (
            "from test.bindings.torch.test_segment_mm_validation "
            "import _mixed_validation_capture_child; _mixed_validation_capture_child()"
        )
        result = subprocess.run(  # noqa: S603 -- fixed interpreter and test helper
            [sys.executable, "-c", script],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            timeout=180,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "mixed validation graph replay preserved" in result.stdout

    def test_checked_graph_replays_valid_lengths(self):
        """Checked forward and backward graphs retain device validation on replay."""
        a, b, lengths = _inputs(torch.float32)
        a = a.cuda().requires_grad_()
        b = b.cuda().requires_grad_()
        lengths = lengths.cuda().to(torch.int64)
        g = torch.ones((8, 5), device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                out = segment_mm(a, b, lengths, validate=True)
                torch.autograd.grad(out, (a, b), g)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = segment_mm(a, b, lengths, validate=True)
            da, db = torch.autograd.grad(out, (a, b), g)
        for values in ((3, 0, 2), (0, 4, 1), (0, 0, 0)):
            lengths.copy_(torch.tensor(values, device="cuda"))
            graph.replay()
            a_ref = a.detach().requires_grad_()
            b_ref = b.detach().requires_grad_()
            expected = _reference(a_ref, b_ref, values)
            expected_da, expected_db = torch.autograd.grad(expected, (a_ref, b_ref), g)
            for actual, reference in zip(
                (out, da, db), (expected, expected_da, expected_db), strict=True
            ):
                _close(actual, reference)

    def test_checked_backward_keeps_validation(self, monkeypatch):
        """Checked reverse derivatives assert on-device without scalar reads."""
        a, b, lengths = _inputs(torch.float32)
        a = a.cuda().requires_grad_()
        b = b.cuda().requires_grad_()
        lengths = lengths.cuda().to(torch.int64)
        original = torch._assert_async
        assertions = []

        def record(condition, *args, **kwargs):
            assertions.append(condition.device)
            return original(condition, *args, **kwargs)

        def unexpected(*args, **kwargs):
            raise AssertionError("Checked CUDA execution read a device scalar")

        with monkeypatch.context() as patch:
            patch.setattr(torch, "_assert_async", record)
            patch.setattr(torch.Tensor, "item", unexpected)
            output = segment_mm(a, b, lengths, validate=True)
            assert assertions
            assertions.clear()
            da, db = torch.autograd.grad(output, (a, b), torch.ones_like(output))
            assert assertions
            assert all(device.type == "cuda" for device in assertions)
        expected = _reference(a, b, (3, 0, 2))
        expected_da, expected_db = torch.autograd.grad(
            expected, (a, b), torch.ones_like(expected)
        )
        _close(output, expected)
        _close(da, expected_da)
        _close(db, expected_db)
