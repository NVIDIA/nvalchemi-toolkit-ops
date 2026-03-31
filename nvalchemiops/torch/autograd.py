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

"""
Autograd Utilities for Warp-PyTorch Integration
================================================

This module provides utilities for integrating Warp's automatic differentiation
with PyTorch custom operators. It abstracts common patterns for:

1. Checking if any tensor requires gradients
2. Conditionally creating Warp tapes
3. Storing tape and warp arrays on output tensors
4. Retrieving them in backward passes
5. Decorator-based custom op registration with automatic backward generation

import warp as wp
import torch
from contextlib import contextmanager, nullcontext
from typing import Any, Optional, Sequence, Union
"""

import inspect
import itertools
import weakref
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import wraps
from typing import Any, get_args, get_origin, get_type_hints

import torch
import warp as wp
from torch._subclasses.fake_tensor import is_fake

from nvalchemiops.torch.types import get_wp_dtype, get_wp_vec_dtype

# =============================================================================
# Dtype Resolution Helper
# =============================================================================


def _resolve_warp_dtype(dtype, tensor: torch.Tensor):
    """Resolve a potentially generic Warp dtype to a concrete dtype.

    This handles:
    - typing.Any: infer from tensor dtype
    - wp.array(dtype=Any, ...): extract inner dtype and infer
    - Concrete dtypes (wp.float64, wp.vec3d, etc.): pass through

    Parameters
    ----------
    dtype : Any
        The dtype specification, which may be typing.Any, a wp.array type,
        or a concrete Warp dtype.
    tensor : torch.Tensor
        The tensor to infer dtype from if needed.

    Returns
    -------
    wp.dtype
        Concrete Warp dtype.
    """
    # Handle typing.Any directly
    if dtype is Any:
        # Check tensor shape to determine if it's scalar or vector
        if tensor.dim() >= 2 and tensor.shape[-1] == 3:
            return get_wp_vec_dtype(tensor.dtype)
        return get_wp_dtype(tensor.dtype)

    # Handle wp.array types that have Any as inner dtype
    # These look like: array(ndim=2, dtype=typing.Any)
    if hasattr(dtype, "dtype"):
        inner_dtype = dtype.dtype
        if inner_dtype is Any:
            # Check tensor shape to determine if it's scalar or vector
            if tensor.dim() >= 2 and tensor.shape[-1] == 3:
                return get_wp_vec_dtype(tensor.dtype)
            return get_wp_dtype(tensor.dtype)

    # Return the dtype as-is if it's concrete
    return dtype


def _resolve_output_dtype(dtype_spec, *args):
    """Resolve an OutputSpec dtype, which may be a callable or concrete type.

    Warp dtypes (e.g. ``wp.float64``) are *type* objects and therefore
    ``callable``.  We distinguish user-provided resolver functions (lambdas,
    regular functions) from Warp types by checking ``isinstance(dtype_spec, type)``.

    Parameters
    ----------
    dtype_spec : Any
        Concrete Warp dtype **or** a callable ``(*forward_args) -> wp_dtype``.
    *args
        Forward-pass positional arguments, forwarded to the callable.

    Returns
    -------
    wp.dtype
        Resolved concrete Warp dtype.
    """
    if callable(dtype_spec) and not isinstance(dtype_spec, type):
        return dtype_spec(*args)
    return dtype_spec


# Warp dtype -> PyTorch dtype mapping (scalar dtype underlying vec/mat types)
_WP_TO_TORCH: dict[type, torch.dtype] = {
    wp.float16: torch.float16,
    wp.float32: torch.float32,
    wp.float64: torch.float64,
    wp.vec3h: torch.float16,
    wp.vec3f: torch.float32,
    wp.vec3d: torch.float64,
    wp.mat33h: torch.float16,
    wp.mat33f: torch.float32,
    wp.mat33d: torch.float64,
}


def _wp_dtype_to_torch(wp_dtype) -> torch.dtype:
    """Map a Warp scalar/vec/mat dtype to its PyTorch scalar dtype."""
    return _WP_TO_TORCH.get(wp_dtype, torch.float64)


# =============================================================================
# Output Specification for warp_custom_op decorator
# =============================================================================


@dataclass
class OutputSpec:
    """Specification for a custom op output.

    Parameters
    ----------
    name : str
        Name of the output (used for backward pass).
    dtype : wp dtype
        Warp dtype (e.g., wp.float64, wp.vec3d).
    shape : Callable or tuple
        Either a tuple of ints, or a callable that takes the input tensors
        and returns the shape. For callable, signature should match the
        custom op's input signature.
    torch_dtype : torch.dtype, optional
        PyTorch dtype override. If omitted (``None``), the dtype is inferred
        from the resolved Warp dtype via ``_wp_dtype_to_torch``.

    Examples
    --------
    >>> OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],))
    >>> OutputSpec("forces", wp.vec3d, lambda pos, *_: (pos.shape[0], 3))
    >>> OutputSpec("virial", wp.mat33d, (3, 3))  # Static shape
    """

    name: str
    dtype: Any  # Warp dtype
    shape: Callable | tuple
    torch_dtype: torch.dtype | None = None


@dataclass
class _RegisteredBackwardState:
    """Runtime Warp state stored behind a saved tensor token."""

    tape: wp.Tape
    arrays: dict[str, wp.array]


@dataclass(frozen=True)
class _TensorGradInputSpec:
    """Metadata for tensor-valued inputs in the generated backward wrapper."""

    name: str
    optional: bool


def _is_tensor_annotation(annotation: Any) -> bool:
    """Return True when a type annotation contains ``torch.Tensor``."""
    if annotation is inspect.Signature.empty:
        return False
    if annotation is torch.Tensor:
        return True
    origin = get_origin(annotation)
    if origin is None:
        return False
    return any(_is_tensor_annotation(arg) for arg in get_args(annotation))


def _is_optional_tensor_annotation(annotation: Any) -> bool:
    """Return True when a type annotation is ``torch.Tensor | None``-like."""
    if annotation is inspect.Signature.empty:
        return False
    origin = get_origin(annotation)
    if origin is None:
        return False
    args = get_args(annotation)
    return torch.Tensor in args and type(None) in args


def _schema_arg_from_parameter(
    parameter: inspect.Parameter,
    resolved_annotation: Any,
) -> str:
    """Translate a Python annotation/default pair into a custom-op schema arg."""
    annotation = resolved_annotation
    if annotation is torch.Tensor:
        schema_type = "Tensor"
    elif annotation is float:
        schema_type = "float"
    elif annotation is int:
        schema_type = "SymInt"
    elif annotation is bool:
        schema_type = "bool"
    else:
        origin = get_origin(annotation)
        args = get_args(annotation)
        if origin is None or annotation is inspect.Signature.empty:
            raise TypeError(
                f"Unsupported annotation {annotation!r} for warp_custom_op input "
                f"'{parameter.name}'."
            )
        if torch.Tensor in args and type(None) in args:
            schema_type = "Tensor?"
        else:
            raise TypeError(
                f"Unsupported annotation {annotation!r} for warp_custom_op input "
                f"'{parameter.name}'."
            )

    if parameter.default is inspect.Signature.empty:
        return f"{schema_type} {parameter.name}"
    if parameter.default is None:
        return f"{schema_type} {parameter.name}=None"
    return f"{schema_type} {parameter.name}={parameter.default!r}"


def _normalize_outputs(result: Any) -> tuple[torch.Tensor, ...]:
    """Normalize a custom-op result into a tuple of tensors."""
    if isinstance(result, tuple):
        return result
    return (result,)


@contextmanager
def warp_stream_from_torch(*values: Any):
    """Bind Warp launches to PyTorch's current CUDA stream when tensors are CUDA."""
    stream_tensor = next(
        (
            value
            for value in values
            if isinstance(value, torch.Tensor) and value.is_cuda
        ),
        None,
    )
    if stream_tensor is None:
        with nullcontext():
            yield None
        return

    torch_stream = torch.cuda.current_stream(stream_tensor.device)
    with wp.ScopedStream(wp.stream_from_torch(torch_stream)):
        yield torch_stream


def _first_tensor_device(*values: Any) -> torch.device:
    """Return the first tensor device found, or CPU as a defensive fallback."""
    for value in values:
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cpu")


def _zero_grads_for_inputs(
    tensor_inputs: tuple[torch.Tensor | None, ...],
    fallback_device: torch.device,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    """Return zero-valued gradient placeholders matching tensor_inputs.

    Parameters
    ----------
    tensor_inputs : tuple[torch.Tensor | None, ...]
        The saved tensor inputs from the forward pass.
    fallback_device : torch.device
        Device to use for scalar zero when an input is None.

    Returns
    -------
    torch.Tensor | tuple[torch.Tensor, ...]
        Single tensor if one input, otherwise a tuple.
    """
    grads = tuple(
        torch.zeros_like(t)
        if isinstance(t, torch.Tensor)
        else torch.zeros((), device=fallback_device)
        for t in tensor_inputs
    )
    if len(grads) == 1:
        return grads[0]
    return grads


def _set_output_gradients(
    arrays: dict[str, wp.array],
    output_names: list[str],
    grad_outputs: tuple[torch.Tensor | None, ...],
) -> None:
    """Copy upstream PyTorch gradients into Warp output arrays."""
    for output_name, grad_output in zip(output_names, grad_outputs):
        if grad_output is None or output_name not in arrays:
            continue
        output_array = arrays[output_name]
        expected_torch_dtype = _wp_dtype_to_torch(output_array.dtype)
        grad_tensor = grad_output.contiguous()
        grad_tensor = grad_tensor.to(expected_torch_dtype)
        wp_grad = wp.from_torch(grad_tensor, dtype=output_array.dtype)
        wp.copy(output_array.grad, wp_grad)


def _extract_tensor_input_gradients(
    arrays: dict[str, wp.array],
    tensor_input_specs: list[_TensorGradInputSpec],
    tensor_inputs: tuple[Any, ...],
) -> tuple[torch.Tensor, ...]:
    """Materialize Warp input gradients as PyTorch tensors."""
    placeholder_device = _first_tensor_device(*tensor_inputs)
    gradients = []
    for spec, tensor_input in zip(tensor_input_specs, tensor_inputs):
        if not isinstance(tensor_input, torch.Tensor):
            gradients.append(torch.zeros((), device=placeholder_device))
        elif spec.name in arrays and arrays[spec.name].grad is not None:
            gradients.append(wp.to_torch(arrays[spec.name].grad))
        else:
            gradients.append(torch.zeros_like(tensor_input))
    return tuple(gradients)


def warp_custom_op(
    name: str,
    outputs: list[OutputSpec],
    grad_arrays: list[str] | None = None,
    mutates_args: tuple = (),
):
    """Decorator to create a Warp-backed PyTorch op with compile-safe autograd.

    This decorator eliminates boilerplate by automatically generating:
    - A ``torch.library.custom_op`` forward registered with fake/meta support
    - A hidden token input for runtime state handoff while the public wrapper
      still exposes only the user-visible signature
    - A traceable ``register_autograd`` wrapper that replays Warp tapes through
      an opaque backward custom op
    - Stream binding so Warp launches execute on PyTorch's current CUDA stream

    Parameters
    ----------
    name : str
        Full custom op name (e.g., "alchemiops::_my_kernel").
    outputs : list[OutputSpec]
        Specifications for each output tensor.
    grad_arrays : list[str], optional
        Names of warp arrays to track for gradients. Should include output names
        first, then differentiable input names. If None, auto-generated from
        outputs + all inputs that are likely differentiable (excludes common
        non-differentiable names like neighbor_list, batch_idx, etc.).
    mutates_args : tuple, default=()
        Arguments that are mutated by the op (passed to custom_op).

    Returns
    -------
    Callable
        Decorator function.

    Examples
    --------
    >>> @warp_custom_op(
    ...     name="alchemiops::_ewald_real_space_energy",
    ...     outputs=[
    ...         OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],)),
    ...     ],
    ...     grad_arrays=["energies", "positions", "charges", "cell", "alpha"],
    ... )
    ... def _ewald_real_space_energy(
    ...     positions: torch.Tensor,
    ...     charges: torch.Tensor,
    ...     cell: torch.Tensor,
    ...     alpha: torch.Tensor,
    ...     neighbor_list: torch.Tensor,
    ...     neighbor_shifts: torch.Tensor,
    ... ) -> torch.Tensor:
    ...     # Implementation here - no boilerplate needed!
    ...     ...
    ...     return energies

    Notes
    -----
    The decorated function should still call ``attach_for_backward()`` at the
    end of grad-enabled forward execution so the registered forward op can
    collect the runtime Warp tape and arrays from the real output tensor.
    """
    # Non-differentiable input names (won't receive gradients)
    NON_GRAD_INPUTS = {
        "neighbor_list",
        "neighbor_shifts",
        "neighbor_matrix",
        "neighbor_matrix_shifts",
        "batch_idx",
        "mask_value",
        "idx_i",
        "idx_j",
        "unit_shifts",
        "compute_virial",
    }

    def decorator(func: Callable) -> Callable:
        _state_counter = itertools.count(1)
        _state_registry: dict[int, _RegisteredBackwardState] = {}

        def _pop_state(token_id: int) -> _RegisteredBackwardState:
            state = _state_registry.pop(token_id, None)
            if state is None:
                raise RuntimeError(
                    f"Missing registered Warp backward state for token {token_id}. "
                    "The forward custom op likely did not attach a tape, or the state "
                    "was released before backward executed."
                )
            return state

        def _discard_state(token_id: int) -> None:
            """Best-effort cleanup for runtime Warp state when graphs are abandoned."""
            _state_registry.pop(token_id, None)

        # Extract input names from function signature
        sig = inspect.signature(func)
        resolved_hints = get_type_hints(func)
        input_names = list(sig.parameters.keys())
        tensor_input_names = [
            name
            for name in input_names
            if _is_tensor_annotation(
                resolved_hints.get(name, sig.parameters[name].annotation)
            )
        ]

        # Auto-generate grad_arrays if not provided
        nonlocal grad_arrays
        if grad_arrays is None:
            output_names = [o.name for o in outputs]
            differentiable_inputs = [n for n in input_names if n not in NON_GRAD_INPUTS]
            grad_arrays = output_names + differentiable_inputs

        output_names = [o.name for o in outputs]
        differentiable_input_names = [
            n for n in input_names if n not in NON_GRAD_INPUTS
        ]
        hidden_state_name = "_warp_state"
        hidden_input_position = next(
            (
                index
                for index, name in enumerate(input_names)
                if sig.parameters[name].default is not inspect.Signature.empty
            ),
            len(input_names),
        )
        raw_input_names = (
            input_names[:hidden_input_position]
            + [hidden_state_name]
            + input_names[hidden_input_position:]
        )
        raw_input_positions = {
            raw_name: index for index, raw_name in enumerate(raw_input_names)
        }
        tensor_grad_input_specs = [
            _TensorGradInputSpec(
                name=name,
                optional=_is_optional_tensor_annotation(
                    resolved_hints.get(name, sig.parameters[name].annotation)
                ),
            )
            for name in differentiable_input_names
            if name in tensor_input_names
        ]
        tensor_grad_input_name_set = {spec.name for spec in tensor_grad_input_specs}
        raw_forward_input_args = []
        for raw_name in raw_input_names:
            if raw_name == hidden_state_name:
                raw_forward_input_args.append(f"Tensor {hidden_state_name}")
            else:
                raw_forward_input_args.append(
                    _schema_arg_from_parameter(
                        sig.parameters[raw_name],
                        resolved_hints.get(
                            raw_name, sig.parameters[raw_name].annotation
                        ),
                    )
                )
        raw_forward_inputs = ", ".join(raw_forward_input_args)
        return_count = len(outputs)
        if return_count == 1:
            forward_schema = f"({raw_forward_inputs}) -> Tensor"
        else:
            forward_returns = ", ".join("Tensor" for _ in range(return_count))
            forward_schema = f"({raw_forward_inputs}) -> ({forward_returns})"

        def _bind_call(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            return tuple(bound.arguments[name] for name in input_names)

        def _needs_registered_backward(args: tuple[Any, ...]) -> bool:
            return any(
                isinstance(arg, torch.Tensor) and arg.requires_grad
                for name, arg in zip(input_names, args)
                if name in tensor_grad_input_name_set
            )

        def _register_runtime_state(
            args: tuple[Any, ...],
            result: tuple[torch.Tensor, ...],
            token_tensor: torch.Tensor,
        ) -> None:
            if not _needs_registered_backward(args):
                return

            first_output = result[0]
            if not hasattr(first_output, "_warp_tape"):
                raise RuntimeError(
                    f"{func.__name__} did not attach Warp backward state to its first output. "
                    "Gradient-enabled warp_custom_op calls must use attach_for_backward()."
                )

            arrays = {
                array_name: getattr(first_output, f"_wp_{array_name}")
                for array_name in grad_arrays
                if hasattr(first_output, f"_wp_{array_name}")
            }
            state = _RegisteredBackwardState(
                tape=first_output._warp_tape,
                arrays=arrays,
            )
            token_id = int(token_tensor.item())
            _state_registry[token_id] = state
            weakref.finalize(token_tensor, _discard_state, token_id)
            for array_name in list(arrays):
                delattr(first_output, f"_wp_{array_name}")
            delattr(first_output, "_warp_tape")

        @torch.library.custom_op(name, mutates_args=mutates_args, schema=forward_schema)
        @wraps(func)
        def custom_op_impl(*all_args):
            token_tensor = all_args[hidden_input_position]
            args = (
                all_args[:hidden_input_position] + all_args[hidden_input_position + 1 :]
            )
            with warp_stream_from_torch(*args):
                result = _normalize_outputs(func(*args))
            _register_runtime_state(args, result, token_tensor)
            if len(result) == 1:
                return result[0]
            return result

        @custom_op_impl.register_fake
        def fake_impl(*all_args):
            args = (
                all_args[:hidden_input_position] + all_args[hidden_input_position + 1 :]
            )
            device = None
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    device = arg.device
                    break
            if device is None:
                device = torch.device("cpu")

            fake_outputs = []
            for spec in outputs:
                if callable(spec.shape):
                    shape = spec.shape(*args)
                else:
                    shape = spec.shape
                if spec.torch_dtype is not None:
                    tdtype = spec.torch_dtype
                else:
                    resolved_wp = _resolve_output_dtype(spec.dtype, *args)
                    tdtype = _wp_dtype_to_torch(resolved_wp)
                fake_outputs.append(torch.zeros(shape, device=device, dtype=tdtype))

            if len(fake_outputs) == 1:
                return fake_outputs[0]
            return tuple(fake_outputs)

        backward_custom_op = None
        if tensor_grad_input_specs:
            backward_name = f"{name}_backward"
            backward_inputs = ", ".join(
                [
                    "Tensor token",
                    *[f"Tensor? grad_{output_name}" for output_name in output_names],
                    *[
                        f"{'Tensor?' if spec.optional else 'Tensor'} {spec.name}"
                        for spec in tensor_grad_input_specs
                    ],
                ]
            )
            backward_returns = ", ".join("Tensor" for _ in tensor_grad_input_specs)
            if len(tensor_grad_input_specs) == 1:
                backward_schema = f"({backward_inputs}) -> Tensor"
            else:
                backward_schema = f"({backward_inputs}) -> ({backward_returns})"

            @torch.library.custom_op(
                backward_name,
                mutates_args=(),
                schema=backward_schema,
            )
            def backward_custom_op_impl(*all_args):
                token = all_args[0]
                num_outputs = len(output_names)
                grad_outputs = tuple(all_args[1 : 1 + num_outputs])
                tensor_inputs = tuple(all_args[1 + num_outputs :])
                if is_fake(token):
                    device = _first_tensor_device(*grad_outputs, *tensor_inputs)
                    return _zero_grads_for_inputs(tensor_inputs, device)
                state = _pop_state(int(token.item()))
                with warp_stream_from_torch(*grad_outputs, *tensor_inputs):
                    _set_output_gradients(state.arrays, output_names, grad_outputs)
                    state.tape.backward()
                gradients = _extract_tensor_input_gradients(
                    state.arrays,
                    tensor_grad_input_specs,
                    tensor_inputs,
                )
                if len(gradients) == 1:
                    return gradients[0]
                return gradients

            @backward_custom_op_impl.register_fake
            def backward_fake_impl(*all_args):
                tensor_inputs = tuple(all_args[1 + len(output_names) :])
                device = _first_tensor_device(
                    *all_args[1 : 1 + len(output_names)],
                    *tensor_inputs,
                )
                return _zero_grads_for_inputs(tensor_inputs, device)

            backward_custom_op = backward_custom_op_impl

            def setup_context_impl(ctx, inputs, output):
                del output
                token_tensor = inputs[hidden_input_position]
                saved_tensors = [token_tensor]
                runtime_input_meta = []
                for spec in tensor_grad_input_specs:
                    inp = inputs[raw_input_positions[spec.name]]
                    was_present = isinstance(inp, torch.Tensor)
                    required_grad = was_present and inp.requires_grad
                    runtime_input_meta.append((spec.name, was_present, required_grad))
                    if was_present:
                        saved_tensors.append(inp)
                ctx._tensor_grad_runtime_meta = runtime_input_meta
                ctx.save_for_backward(*saved_tensors)

            def backward_impl(ctx, *grad_outputs):
                saved_tensors = ctx.saved_tensors
                token = saved_tensors[0]
                saved_runtime_tensors = list(saved_tensors[1:])
                tensor_inputs = []
                for _, was_present, _ in ctx._tensor_grad_runtime_meta:
                    if was_present:
                        tensor_inputs.append(saved_runtime_tensors.pop(0))
                    else:
                        tensor_inputs.append(None)
                raw_gradients = backward_custom_op(token, *grad_outputs, *tensor_inputs)
                if len(tensor_grad_input_specs) == 1:
                    raw_gradients = (raw_gradients,)
                gradients_by_name = {
                    spec.name: grad
                    for spec, grad in zip(tensor_grad_input_specs, raw_gradients)
                }
                runtime_input_meta = {
                    name: (was_present, required_grad)
                    for name, was_present, required_grad in ctx._tensor_grad_runtime_meta
                }
                gradients = []
                for name in input_names:
                    was_present, required_grad = runtime_input_meta.get(
                        name, (False, False)
                    )
                    if was_present and required_grad:
                        gradients.append(gradients_by_name[name])
                    else:
                        gradients.append(None)
                gradients.insert(hidden_input_position, None)
                return tuple(gradients)

            torch.library.register_autograd(
                custom_op_impl,
                backward_impl,
                setup_context=setup_context_impl,
            )

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_args = _bind_call(*args, **kwargs)
            token_tensor = torch.tensor(next(_state_counter), dtype=torch.int64)
            raw_args = (
                bound_args[:hidden_input_position]
                + (token_tensor,)
                + bound_args[hidden_input_position:]
            )
            return custom_op_impl(*raw_args)

        return wrapper

    return decorator


def warp_from_torch(
    tensor: torch.Tensor,
    warp_dtype: type,
    requires_grad: bool | None = None,
) -> wp.array:
    """
    Convert a PyTorch tensor to a Warp array with proper gradient tracking.

    Parameters
    ----------
    tensor : torch.Tensor
        Input PyTorch tensor
    warp_dtype : wp.dtype
        Warp data type for the array
    requires_grad : bool | None, optional
        Override gradient tracking. If None, inherits from tensor.requires_grad

    Returns
    -------
    wp.array
        Warp array with gradient tracking if needed
    """
    # Determine if we need gradient tracking
    needs_grad = requires_grad if requires_grad is not None else tensor.requires_grad

    # For backward compatibility, we need full warp arrays, not ctypes
    # ctypes are lightweight wrappers that don't work with tape.backward()
    use_ctype = not needs_grad
    return wp.from_torch(
        tensor.detach(),
        dtype=warp_dtype,
        requires_grad=needs_grad,
        return_ctype=use_ctype,
    )


def needs_grad(*tensors: torch.Tensor) -> bool:
    """
    Check if any of the provided tensors requires gradients.

    This is useful for conditionally enabling Warp gradient tracking
    and tape recording only when needed for backpropagation.

    Parameters
    ----------
    *tensors : torch.Tensor
        Variable number of PyTorch tensors to check

    Returns
    -------
    bool
        True if any tensor requires gradients, False otherwise

    Examples
    --------
    >>> positions = torch.randn(100, 3, requires_grad=True)
    >>> charges = torch.randn(100, requires_grad=False)
    >>> needs_grad(positions, charges)
    True
    >>> needs_grad(charges)
    False
    """
    return any(t.requires_grad for t in tensors if isinstance(t, torch.Tensor))


@contextmanager
def WarpAutogradContextManager(enable: bool):
    """
    Conditionally create a Warp tape as a context manager.

    Returns a Warp Tape if enable=True for gradient recording,
    otherwise returns a nullcontext (no-op) for zero overhead.

    Parameters
    ----------
    enable : bool
        Whether to create a tape for gradient recording

    Yields
    ------
    wp.Tape or nullcontext
        Active tape for recording if enabled, otherwise nullcontext

    Examples
    --------
    >>> needs_grad_flag = needs_grad(positions, charges)
    >>> with WarpAutogradContextManager(needs_grad_flag) as tape:
    ...     wp.launch(kernel, ...)
    >>> if needs_grad_flag:
    ...     # tape is a wp.Tape instance
    ...     tape.backward()
    """
    if enable:
        tape = wp.Tape()
        with tape:
            yield tape
    else:
        with nullcontext():
            yield None


def attach_for_backward(
    output: torch.Tensor, tape: wp.Tape | None = None, **warp_arrays: wp.array
) -> None:
    """
    Attach Warp tape and arrays to a PyTorch tensor for later retrieval in backward.

    This stores the tape and warp arrays as attributes on the output tensor,
    allowing them to be retrieved in the backward pass of a custom operator.

    Parameters
    ----------
    output : torch.Tensor
        PyTorch tensor to attach attributes to (usually the output of forward)
    tape : wp.Tape, optional
        Warp tape containing recorded operations for backward pass
    **warp_arrays : wp.array
        Named warp arrays to store (e.g., positions=wp_positions, charges=wp_charges)

    Examples
    --------
    >>> attach_for_backward(
    ...     output,
    ...     tape=tape,
    ...     positions=wp_positions,
    ...     charges=wp_charges,
    ...     energies=wp_energies,
    ... )
    >>> # Later in backward:
    >>> tape = output._warp_tape
    >>> wp_positions = output._wp_positions
    """
    if tape is not None:
        output._warp_tape = tape
    for name, array in warp_arrays.items():
        setattr(output, f"_wp_{name}", array)


def retrieve_for_backward(
    output: torch.Tensor, *array_names: str
) -> tuple[wp.Tape, dict[str, wp.array]]:
    """
    Retrieve Warp tape and arrays from a PyTorch tensor in backward pass.

    Parameters
    ----------
    output : torch.Tensor
        PyTorch tensor that has attached Warp objects (from attach_for_backward)
    *array_names : str
        Names of warp arrays to retrieve (without '_wp_' prefix)

    Returns
    -------
    tape : wp.Tape
        The stored Warp tape
    arrays : dict[str, wp.array]
        Dictionary mapping names to warp arrays

    Examples
    --------
    >>> tape, arrays = retrieve_for_backward(
    ...     ctx.output,
    ...     'positions', 'charges', 'energies'
    ... )
    >>> wp_positions = arrays['positions']
    >>> tape.backward()
    """
    tape = output._warp_tape
    # Some optional outputs (e.g., virial in forward-only mode) may not be attached.
    # Keep retrieval tolerant so backward can skip non-attached arrays.
    arrays = {
        name: getattr(output, f"_wp_{name}")
        for name in array_names
        if hasattr(output, f"_wp_{name}")
    }
    return tape, arrays


def extract_gradients(
    ctx: Any,
    warp_arrays: dict[str, wp.array],
    input_names: list[str] | tuple[str],
) -> tuple[torch.Tensor | None, ...]:
    """
    Extract gradients from warp arrays and return in correct order for PyTorch.

    This helper extracts gradients from warp arrays and returns them in the
    same order as the forward pass inputs, with None for inputs that don't
    require gradients.

    Parameters
    ----------
    ctx : Any
        PyTorch autograd context with saved tensors (must have attributes
        matching input_names)
    warp_arrays : dict[str, wp.array]
        Dictionary mapping input names to warp arrays with computed gradients
    input_names : Sequence[str]
        Names of inputs in the order they appear in forward function signature

    Returns
    -------
    tuple[Optional[torch.Tensor], ...]
        Gradients in order, with None for inputs without requires_grad

    Examples
    --------
    >>> # In backward function:
    >>> tape, arrays = retrieve_for_backward(ctx.output, 'positions', 'charges')
    >>> tape.backward()
    >>> return extract_gradients(
    ...     ctx,
    ...     arrays,
    ...     ['positions', 'charges', 'cell', 'alpha']
    ... )
    >>> # Returns: (grad_pos, grad_charges, None, None)
    """
    gradients = []
    for name in input_names:
        input_tensor = getattr(ctx, name)
        if hasattr(input_tensor, "requires_grad") and input_tensor.requires_grad:
            if name in warp_arrays:
                gradients.append(wp.to_torch(warp_arrays[name].grad))
            else:
                # Warp array not provided, return zeros
                gradients.append(torch.zeros_like(input_tensor))
        else:
            gradients.append(None)
    return tuple(gradients)


def standard_backward(
    ctx: Any,
    grad_outputs: torch.Tensor | tuple[torch.Tensor | None, ...],
    output_names: str | list[str] | tuple[str],
    array_names: list[str] | tuple[str],
    input_names: list[str] | tuple[str],
    output_dtypes: Any | list[Any] | tuple[Any] | None = None,
) -> tuple[torch.Tensor | None, ...]:
    """
    Standard backward implementation for Warp-PyTorch custom operators.

    This function handles both single-output and multiple-output operators.
    It encapsulates the common backward pattern:
    1. Retrieve tape and warp arrays from context
    2. Set gradient(s) on output(s)
    3. Run tape backward
    4. Extract and return gradients

    Parameters
    ----------
    ctx : Any
        PyTorch autograd context with saved tensors
    grad_outputs : torch.Tensor or tuple[Optional[torch.Tensor], ...]
        Gradient(s) from upstream operations.
        - Single output: pass the gradient tensor directly
        - Multiple outputs: pass tuple of gradient tensors (None if unused in loss)
    output_names : str or Sequence[str]
        Name(s) of the output array(s) stored in ctx.
        - Single output: 'output' or 'energies'
        - Multiple outputs: ['energies', 'forces']
    array_names : Sequence[str]
        Names of ALL warp arrays that were attached (outputs + inputs).
        MUST include all output array names first!
        Examples:
        - Single output: ['output', 'positions', 'charges']
        - Multiple outputs: ['energies', 'forces', 'positions']
    input_names : Sequence[str]
        Names of all inputs in forward function signature order
    output_dtypes : Any or Sequence[Any], optional
        Warp dtype(s) for each output. Required for multiple outputs or non-float32 outputs.
        - Single output: wp.float32 (default) or wp.vec3f
        - Multiple outputs: [wp.float32, wp.vec3f]

    Returns
    -------
    tuple[Optional[torch.Tensor], ...]
        Gradients for all inputs (None for those without requires_grad)

    Examples
    --------
    Single output operator:

    >>> # In forward:
    >>> attach_for_backward(output, tape=tape, output=wp_output,
    ...                     positions=wp_positions, charges=wp_charges)
    >>>
    >>> # In backward:
    >>> def backward(ctx, grad_output):
    ...     return standard_backward(
    ...         ctx,
    ...         grad_outputs=grad_output,  # Single tensor (note: parameter name)
    ...         output_names='output',  # Single string
    ...         array_names=['output', 'positions', 'charges'],
    ...         input_names=['positions', 'charges', 'cell', 'alpha'],
    ...     )

    Multiple output operator:

    >>> # In forward:
    >>> attach_for_backward(energies, tape=tape, energies=wp_energies,
    ...                     forces=wp_forces, positions=wp_positions)
    >>> return energies, forces
    >>>
    >>> # In backward:
    >>> def backward(ctx, grad_energies, grad_forces):
    ...     return standard_backward(
    ...         ctx,
    ...         grad_outputs=(grad_energies, grad_forces),  # Tuple
    ...         output_names=['energies', 'forces'],  # List
    ...         output_dtypes=[wp.float32, wp.vec3f],  # Required!
    ...         array_names=['energies', 'forces', 'positions'],
    ...         input_names=['positions'],
    ...     )
    """
    # Normalize inputs to lists/tuples for uniform handling
    is_single_output = isinstance(output_names, str)
    if is_single_output:
        # Single output case
        output_names = [output_names]
        grad_outputs = [grad_outputs]
        if output_dtypes is None:
            output_dtypes = [wp.float32]  # Default for single output
        else:
            output_dtypes = [output_dtypes]
    else:
        # Multiple outputs case
        if output_dtypes is None:
            raise ValueError(
                "output_dtypes must be specified for multiple outputs. "
                "Example: output_dtypes=[wp.float32, wp.vec3f]"
            )
        if not isinstance(grad_outputs, (tuple, list)):
            raise ValueError(
                "grad_outputs must be a tuple/list for multiple outputs. "
                f"Got: {type(grad_outputs)}"
            )
        # Validate lengths match
        if len(grad_outputs) != len(output_names):
            raise ValueError(
                f"Mismatch: got {len(grad_outputs)} grad_outputs but {len(output_names)} output_names"
            )
        if len(output_dtypes) != len(output_names):
            raise ValueError(
                f"Mismatch: got {len(output_dtypes)} output_dtypes but {len(output_names)} output_names"
            )

    # Get the first output tensor from context (tape is attached there)
    first_output = getattr(ctx, output_names[0])

    # Retrieve tape and warp arrays
    tape, arrays = retrieve_for_backward(first_output, *array_names)

    stream_values = list(grad_outputs)
    stream_values.extend(getattr(ctx, name, None) for name in input_names)
    with warp_stream_from_torch(*stream_values):
        for output_name, grad_output, dtype in zip(
            output_names, grad_outputs, output_dtypes
        ):
            if grad_output is not None and output_name in arrays:
                output_array = arrays[output_name]
                actual_dtype = output_array.dtype
                wp_grad = wp.from_torch(grad_output.contiguous(), dtype=actual_dtype)
                wp.copy(output_array.grad, wp_grad)

        tape.backward()

    # Extract and return gradients
    return extract_gradients(ctx, arrays, input_names)
