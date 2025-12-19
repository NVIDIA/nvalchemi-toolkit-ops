# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import wraps
from typing import Any

import torch
import warp as wp

from nvalchemiops.types import get_wp_dtype, get_wp_vec_dtype

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
        PyTorch dtype. Defaults to torch.float64.

    Examples
    --------
    >>> OutputSpec("energies", wp.float64, lambda pos, *_: (pos.shape[0],))
    >>> OutputSpec("forces", wp.vec3d, lambda pos, *_: (pos.shape[0], 3))
    >>> OutputSpec("virial", wp.mat33d, (3, 3))  # Static shape
    """

    name: str
    dtype: Any  # Warp dtype
    shape: Callable | tuple
    torch_dtype: torch.dtype = torch.float64


def warp_custom_op(
    name: str,
    outputs: list[OutputSpec],
    grad_arrays: list[str] | None = None,
    mutates_args: tuple = (),
):
    """Decorator to create a PyTorch custom op with automatic autograd registration.

    This decorator eliminates boilerplate by automatically generating:
    - The custom op registration
    - The `register_fake` implementation
    - The `setup_context` function
    - The `backward` function using `standard_backward`
    - The `register_autograd` call

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
    The decorated function should still call `attach_for_backward` at the end
    to link warp arrays for gradient computation. The decorator handles everything
    else (fake registration, setup_context, backward, register_autograd).
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
    }

    def decorator(func: Callable) -> Callable:
        # Extract input names from function signature
        sig = inspect.signature(func)
        input_names = list(sig.parameters.keys())

        # Auto-generate grad_arrays if not provided
        nonlocal grad_arrays
        if grad_arrays is None:
            output_names = [o.name for o in outputs]
            differentiable_inputs = [n for n in input_names if n not in NON_GRAD_INPUTS]
            grad_arrays = output_names + differentiable_inputs

        # Create the custom op
        @torch.library.custom_op(name, mutates_args=mutates_args)
        @wraps(func)
        def custom_op_impl(*args, **kwargs):
            return func(*args, **kwargs)

        # Register fake implementation
        @custom_op_impl.register_fake
        def fake_impl(*args, **kwargs):
            # Determine device from first tensor argument
            device = None
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    device = arg.device
                    break
            if device is None:
                device = torch.device("cpu")

            # Create fake outputs
            fake_outputs = []
            for spec in outputs:
                if callable(spec.shape):
                    shape = spec.shape(*args)
                else:
                    shape = spec.shape
                fake_outputs.append(
                    torch.zeros(shape, device=device, dtype=spec.torch_dtype)
                )

            if len(fake_outputs) == 1:
                return fake_outputs[0]
            return tuple(fake_outputs)

        # Create backward function
        output_names = [o.name for o in outputs]
        output_dtypes = [o.dtype for o in outputs]

        def backward_impl(ctx, *grad_outputs_tuple):
            # Handle single vs multiple outputs
            if len(outputs) == 1:
                grad_outputs = grad_outputs_tuple[0]
                output_names_arg = output_names[0]
                output_dtypes_arg = output_dtypes[0]
            else:
                grad_outputs = grad_outputs_tuple
                output_names_arg = output_names
                output_dtypes_arg = output_dtypes

            return standard_backward(
                ctx,
                grad_outputs=grad_outputs,
                output_names=output_names_arg,
                output_dtypes=output_dtypes_arg,
                array_names=grad_arrays,
                input_names=input_names,
            )

        # Create setup_context function
        # Note: 'outputs' in the outer scope refers to the OutputSpec list from the decorator.
        # We capture num_outputs and output_spec_names to avoid confusion with the
        # 'outputs' parameter that PyTorch passes to setup_context.
        num_outputs = len(outputs)
        output_spec_names = [o.name for o in outputs]

        def setup_context_impl(ctx, inputs, output=None, outputs=None):
            if output is None:
                output = outputs
            # Save all inputs
            for name, inp in zip(input_names, inputs):
                setattr(ctx, name, inp)

            # Save outputs
            # For single output, PyTorch passes the tensor directly (not in tuple)
            # For multiple outputs, PyTorch passes a tuple
            if num_outputs == 1:
                setattr(ctx, output_spec_names[0], output)
            else:
                for spec_name, out in zip(output_spec_names, output):
                    setattr(ctx, spec_name, out)

        # Register autograd
        custom_op_impl.register_autograd(
            backward_impl, setup_context=setup_context_impl
        )

        return custom_op_impl

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
    arrays = {name: getattr(output, f"_wp_{name}") for name in array_names}
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

    # Set gradients on all outputs that participate in backward
    # Skip outputs that weren't attached for gradients (not in arrays dict)
    for output_name, grad_output, dtype in zip(
        output_names, grad_outputs, output_dtypes
    ):
        if grad_output is not None and output_name in arrays:
            output_array = arrays[output_name]

            # Resolve dtype - handle typing.Any or wp.array types with Any
            actual_dtype = _resolve_warp_dtype(dtype, grad_output)

            wp_grad = wp.from_torch(grad_output.contiguous(), dtype=actual_dtype)
            wp.copy(output_array.grad, wp_grad)

    # Run backward pass
    tape.backward()

    # Extract and return gradients
    return extract_gradients(ctx, arrays, input_names)
