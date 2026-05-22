# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Boilerplate-reducing helpers for warp-backed ``torch.library.custom_op``s.

Two levels of helper:

  * ``register_warp_op_chain(...)`` — the do-everything factory. One call
    per op chain. Builds the forward custom_op + register_fake, the
    backward custom_op + register_fake (optionally), the double-backward
    custom_op + register_fake (optionally), and wires forward→backward
    and backward→double_backward via register_autograd. Each launcher
    is supplied as a Python function; output shapes are inferred from
    ``diff_input_positions`` (output i has the shape of forward input
    ``diff_input_positions[i]``). Override with explicit ``*_fake`` and
    schema kwargs when defaults aren't right.

  * ``attach_simple_backward(...)`` — wires ONLY register_autograd onto
    an already-registered custom_op. Useful when the custom_op +
    register_fake declarations are written manually (e.g. for ops with
    unusual schemas like the PME convolve, whose backward op puts a
    complex cotangent in non-position-0 to work around a torch.compile
    inductor bug).

  * ``_match_shape`` / ``_match_shape_batch`` — reshape launcher-style
    grads to match input tensor shapes (collapse 0-d via ``.sum()``,
    reshape otherwise).

Both helpers assume the codebase-wide convention "backward op takes
``(*cotangents, *forward_inputs)``" — unless ``backward_arg_order`` is
overridden for ops that break that convention.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import torch

# ===========================================================================
# Grad-shape coercion helpers
# ===========================================================================


def _match_shape(grad: torch.Tensor, ref: torch.Tensor) -> torch.Tensor | None:
    """Reshape a launcher-style grad to match an input's shape.

    Launchers return alpha/volume/total_charge grads as ``(1,)`` tensors
    regardless of whether the input is 0-d or 1-d. This helper:
      * returns ``None`` if the input doesn't need a grad,
      * collapses to a scalar via ``.sum()`` for 0-d inputs,
      * reshapes + dtype-casts otherwise.
    """
    if not ref.requires_grad:
        return None
    if ref.dim() == 0:
        return grad.sum().to(ref.dtype)
    return grad.reshape(ref.shape).to(ref.dtype)


def _match_shape_batch(grad: torch.Tensor, ref: torch.Tensor) -> torch.Tensor | None:
    """Batch variant of ``_match_shape`` — never collapses to 0-d.

    Use for batched ops where alpha/volume/total_charges are per-system
    ``(B,)`` arrays; ``_match_shape``'s 0-d ``.sum()`` would be wrong here.
    """
    if not ref.requires_grad:
        return None
    return grad.reshape(ref.shape).to(ref.dtype)


# ===========================================================================
# Internal: build register_autograd wiring (used by both helpers below)
# ===========================================================================


def _build_setup_ctx_and_backward_chain(
    backward_op_callable: Callable,
    *,
    diff_input_positions: tuple[int, ...],
    n_forward_inputs: int,
    batch_match: bool = False,
    propagate_outputs: tuple[int, ...] | None = None,
    backward_args: Callable[[tuple, tuple], tuple] | None = None,
) -> tuple[Callable, Callable]:
    """Build (setup_ctx, backward_chain) for register_autograd.

    See ``attach_simple_backward`` for the parameter semantics.

    ``backward_args``: if provided, a callable
    ``(grad_outputs_c: tuple, full_inputs: tuple) -> tuple`` that returns
    the positional arg list to pass to ``backward_op_callable``. Default
    is ``lambda g, f: g + f`` (cotangents-first).
    """
    match_fn = _match_shape_batch if batch_match else _match_shape
    if backward_args is None:

        def backward_args(grad_outputs_c, full_inputs):
            return tuple(grad_outputs_c) + tuple(full_inputs)

    def setup_ctx(ctx, inputs, output):
        tensor_positions = tuple(
            i for i, x in enumerate(inputs) if isinstance(x, torch.Tensor)
        )
        ctx.save_for_backward(*(inputs[i] for i in tensor_positions))
        ctx._warp_tensor_positions = tensor_positions
        ctx._warp_non_tensor = tuple(
            (i, x) for i, x in enumerate(inputs) if not isinstance(x, torch.Tensor)
        )

    def backward_chain(ctx, *grad_outputs):
        if propagate_outputs is not None:
            grad_outputs = tuple(grad_outputs[i] for i in propagate_outputs)

        if all(g is None for g in grad_outputs):
            return tuple(None for _ in range(n_forward_inputs))

        # Reconstruct full inputs in original signature order.
        saved = ctx.saved_tensors
        full_inputs: list = [None] * n_forward_inputs
        for tensor, idx in zip(saved, ctx._warp_tensor_positions):
            full_inputs[idx] = tensor
        for idx, val in ctx._warp_non_tensor:
            full_inputs[idx] = val

        grad_outputs_c = tuple(
            g.contiguous() if g is not None else None for g in grad_outputs
        )
        raw_grads = backward_op_callable(
            *backward_args(grad_outputs_c, tuple(full_inputs))
        )
        if not isinstance(raw_grads, tuple):
            raw_grads = (raw_grads,)

        out: list = [None] * n_forward_inputs
        for raw_grad, fwd_idx in zip(raw_grads, diff_input_positions):
            out[fwd_idx] = match_fn(raw_grad, full_inputs[fwd_idx])
        return tuple(out)

    return setup_ctx, backward_chain


def attach_simple_backward(
    forward_op_name: str,
    backward_op_callable: Callable,
    *,
    diff_input_positions: tuple[int, ...],
    n_forward_inputs: int,
    batch_match: bool = False,
    propagate_outputs: tuple[int, ...] | None = None,
    backward_args: Callable[[tuple, tuple], tuple] | None = None,
) -> None:
    """Wire ``forward_op_name``'s autograd via ``backward_op_callable``.

    By default, assumes the codebase convention: backward op signature is
    ``(*cotangents, *forward_inputs)``. Override ``backward_args`` for ops
    that use a different call ordering (e.g. the PME convolve, whose
    backward op puts a complex cotangent at position 1 to work around a
    torch.compile inductor bug).

    Parameters
    ----------
    forward_op_name : str
        Qualified name (``"<ns>::<op>"``) of the registered forward op.
    backward_op_callable : Callable
        The registered backward custom op, accessed as
        ``torch.ops.<ns>.<backward_name>``.
    diff_input_positions : tuple[int, ...]
        Positions in the forward signature whose grads the backward op
        returns, in the SAME order as the backward's tuple output.
    n_forward_inputs : int
        Total forward inputs (incl. non-tensors / non-diff).
    batch_match : bool, default False
        Use ``_match_shape_batch`` (no 0-d collapse) for batched ops.
    propagate_outputs : tuple[int, ...] | None
        Forward output indices whose cotangents to forward to the backward
        op. Default ``None`` passes them all.
    backward_args : Callable | None
        ``(grad_outputs_c, full_inputs) -> tuple`` returning the positional
        args to pass to ``backward_op_callable``. Default is
        ``(*grad_outputs, *full_inputs)``.
    """
    setup_ctx, backward_chain = _build_setup_ctx_and_backward_chain(
        backward_op_callable,
        diff_input_positions=diff_input_positions,
        n_forward_inputs=n_forward_inputs,
        batch_match=batch_match,
        propagate_outputs=propagate_outputs,
        backward_args=backward_args,
    )
    torch.library.register_autograd(
        forward_op_name,
        backward_chain,
        setup_context=setup_ctx,
    )


# ===========================================================================
# Schema inference + default-fake helpers
# ===========================================================================


def _annotation_to_schema_type(ann: Any, fn_qualname: str, param_name: str) -> str:
    """Map a Python annotation to a ``torch.library`` schema type string."""
    import typing

    scalar_map = {
        torch.Tensor: "Tensor",
        bool: "bool",
        int: "int",
        float: "float",
    }
    if ann in scalar_map:
        return scalar_map[ann]

    # Subscripted generics: tuple[int, int, int] → int[3]; list[int] → int[].
    origin = typing.get_origin(ann)
    args = typing.get_args(ann)
    if origin is tuple and args and all(a is int for a in args):
        return f"int[{len(args)}]"
    if origin is list and args == (int,):
        return "int[]"
    if origin is tuple and args and all(a is float for a in args):
        return f"float[{len(args)}]"
    if origin is list and args == (float,):
        return "float[]"

    raise TypeError(
        f"{fn_qualname}: parameter {param_name!r} has unsupported "
        f"annotation {ann!r}. Pass an explicit ``schema`` kwarg, or add "
        "support for this annotation in ``_annotation_to_schema_type``."
    )


def _schema_from_callable(fn: Callable, return_arity: int) -> str:
    """Build a torch.library schema string from a launcher's signature.

    Maps Python annotations to torch.library schema types. Supported:
      ``torch.Tensor`` → ``Tensor``
      ``bool`` / ``int`` / ``float`` → same
      ``tuple[int, int, int]`` → ``int[3]`` (any fixed-size int tuple)
      ``list[int]`` → ``int[]``
      same for ``float``-valued variants

    Uses ``typing.get_type_hints`` to resolve string annotations
    (``from __future__ import annotations`` modules). ``return_arity == 1``
    produces ``-> Tensor``; otherwise ``-> (Tensor, Tensor, ...)`` with
    ``return_arity`` entries.
    """
    import typing

    try:
        hints = typing.get_type_hints(fn)
    except Exception as e:
        raise TypeError(
            f"{fn.__qualname__}: failed to resolve type hints — {e!r}. "
            "Pass an explicit ``schema`` kwarg."
        ) from e
    sig = inspect.signature(fn)
    parts = []
    for name in sig.parameters:
        if name not in hints:
            raise TypeError(f"{fn.__qualname__}: parameter {name!r} has no annotation.")
        schema_type = _annotation_to_schema_type(hints[name], fn.__qualname__, name)
        parts.append(f"{schema_type} {name}")
    args = ", ".join(parts)
    if return_arity == 1:
        ret = "Tensor"
    else:
        ret = "(" + ", ".join("Tensor" for _ in range(return_arity)) + ")"
    return f"({args}) -> {ret}"


def _default_forward_fake(launcher: Callable) -> Callable:
    """Default forward fake: ``empty_like`` of the first tensor input."""

    def forward_fake(*args):
        for a in args:
            if isinstance(a, torch.Tensor):
                return torch.empty_like(a)
        raise RuntimeError(
            f"{launcher.__qualname__}: cannot derive default fake "
            "(no tensor inputs found)."
        )

    return forward_fake


def _default_backward_fake(
    diff_input_positions: tuple[int, ...],
    cotangent_arity: int,
) -> Callable:
    """Default backward fake: for each ``diff_input_positions[i]``, return
    ``empty_like`` of the forward input at that position.

    Backward op signature is ``(*cotangents, *forward_inputs)``; the fake
    receives args in the same order, so the forward inputs start at
    position ``cotangent_arity``.
    """

    def backward_fake(*args):
        forward_inputs = args[cotangent_arity:]
        out = []
        for pos in diff_input_positions:
            inp = forward_inputs[pos]
            if not isinstance(inp, torch.Tensor):
                raise RuntimeError(
                    f"diff_input_positions[{out.__len__()}]={pos} points to a "
                    f"non-tensor forward input ({type(inp).__name__}); pass an "
                    "explicit ``backward_fake`` for this op."
                )
            out.append(torch.empty_like(inp))
        return tuple(out)

    return backward_fake


def _default_double_backward_fake(
    second_order_diff_positions: tuple[int, ...],
    n_cotangents_of_backward: int,
    n_inputs_of_backward: int,
) -> Callable:
    """Default double-backward fake: ``empty_like`` of each backward input
    at the corresponding diff position.

    Double-bwd signature is ``(*bwd_cotangents, *bwd_inputs)``. The bwd
    inputs themselves are ``(*forward_cotangents, *forward_inputs)``.
    """

    def double_backward_fake(*args):
        backward_inputs = args[n_cotangents_of_backward:]
        out = []
        for pos in second_order_diff_positions:
            inp = backward_inputs[pos]
            if not isinstance(inp, torch.Tensor):
                raise RuntimeError(
                    f"second_order_diff_positions[{out.__len__()}]={pos} "
                    f"points to a non-tensor backward input "
                    f"({type(inp).__name__}); pass explicit "
                    "``double_backward_fake``."
                )
            out.append(torch.empty_like(inp))
        return tuple(out)

    return double_backward_fake


# ===========================================================================
# The do-everything factory
# ===========================================================================


def register_warp_op_chain(
    *,
    name: str,
    forward: Callable,
    backward: Callable | None = None,
    double_backward: Callable | None = None,
    diff_input_positions: tuple[int, ...] | None = None,
    n_forward_inputs: int | None = None,
    second_order_diff_positions: tuple[int, ...] | None = None,
    n_backward_inputs: int | None = None,
    forward_schema: str | None = None,
    backward_schema: str | None = None,
    double_backward_schema: str | None = None,
    forward_fake: Callable | None = None,
    backward_fake: Callable | None = None,
    double_backward_fake: Callable | None = None,
    forward_return_arity: int = 1,
    backward_return_arity: int | None = None,
    double_backward_return_arity: int | None = None,
    batch_match: bool = False,
    propagate_outputs: tuple[int, ...] | None = None,
    backward_args: Callable[[tuple, tuple], tuple] | None = None,
    second_order_backward_args: Callable[[tuple, tuple], tuple] | None = None,
    mutates_args: tuple = (),
) -> dict[str, Any]:
    """Register a complete warp-backed op chain in a single call.

    Builds and wires:
      * ``<name>``                       — forward custom_op + register_fake
      * ``<name>_backward``              — first-order backward custom_op + fake
      * ``<name>_double_backward``       — second-order custom_op + fake (if
                                            ``double_backward`` is provided)

    and registers autograd:
      * forward.autograd  → calls backward (via register_autograd)
      * backward.autograd → calls double_backward (if double_backward given)

    The minimal call for the common case is:

        register_warp_op_chain(
            name="nvalchemiops::pme_energy_corrections",
            forward=_energy_corrections_forward_launch,
            backward=_energy_corrections_backward_launch,
            double_backward=_energy_corrections_double_backward_launch,
            diff_input_positions=(0, 1, 2, 3, 4),
            n_forward_inputs=5,
            second_order_diff_positions=(0, 1, 2, 3, 4, 5),
            n_backward_inputs=6,
        )

    Defaults applied unless explicitly overridden:
      * Schemas inferred from launcher type annotations.
      * Forward fake = ``empty_like(first_tensor_input)``.
      * Backward fake = ``empty_like(forward_input[pos])`` for each
        ``pos`` in ``diff_input_positions``.
      * Double-backward fake = ``empty_like(backward_input[pos])`` for
        each ``pos`` in ``second_order_diff_positions``.

    Returns a dict with the registered ``torch.ops.*`` callables, so the
    caller can route their public functions through them.
    """
    if "::" not in name:
        raise ValueError(f"name must be qualified (e.g. 'ns::op'), got {name!r}")
    namespace, base_name = name.split("::", 1)

    # ---- Forward op + register_fake ------------------------------------
    if forward_schema is None:
        forward_schema = _schema_from_callable(forward, forward_return_arity)

    if forward_fake is None:
        forward_fake = _default_forward_fake(forward)
    torch.library.register_fake(name, forward_fake)

    out: dict[str, Any] = {
        "forward": getattr(getattr(torch.ops, namespace), base_name),
    }

    # ---- Backward op + register_fake + autograd ------------------------
    if backward is None:
        return out
    if diff_input_positions is None or n_forward_inputs is None:
        raise ValueError(
            "When `backward` is provided, `diff_input_positions` and "
            "`n_forward_inputs` are required."
        )
    bwd_name = f"{name}_backward"
    if backward_return_arity is None:
        backward_return_arity = len(diff_input_positions)
    if backward_schema is None:
        backward_schema = _schema_from_callable(backward, backward_return_arity)
    torch.library.custom_op(
        bwd_name,
        backward,
        mutates_args=(),
        schema=backward_schema,
    )
    if backward_fake is None:
        # Cotangent arity = number of forward outputs = forward_return_arity.
        backward_fake = _default_backward_fake(
            diff_input_positions,
            cotangent_arity=forward_return_arity,
        )
    torch.library.register_fake(bwd_name, backward_fake)
    bwd_callable = getattr(getattr(torch.ops, namespace), f"{base_name}_backward")
    out["backward"] = bwd_callable

    attach_simple_backward(
        name,
        bwd_callable,
        diff_input_positions=diff_input_positions,
        n_forward_inputs=n_forward_inputs,
        batch_match=batch_match,
        propagate_outputs=propagate_outputs,
        backward_args=backward_args,
    )

    # ---- Double-backward op + register_fake + autograd ------------------
    if double_backward is None:
        return out
    if second_order_diff_positions is None or n_backward_inputs is None:
        raise ValueError(
            "When `double_backward` is provided, `second_order_diff_positions` "
            "and `n_backward_inputs` are required."
        )
    dbwd_name = f"{name}_double_backward"
    if double_backward_return_arity is None:
        double_backward_return_arity = len(second_order_diff_positions)
    if double_backward_schema is None:
        double_backward_schema = _schema_from_callable(
            double_backward,
            double_backward_return_arity,
        )
    torch.library.custom_op(
        dbwd_name,
        double_backward,
        mutates_args=(),
        schema=double_backward_schema,
    )
    if double_backward_fake is None:
        # n_cotangents_of_backward = backward_return_arity (1 grad per bwd output)
        double_backward_fake = _default_double_backward_fake(
            second_order_diff_positions,
            n_cotangents_of_backward=backward_return_arity,
            n_inputs_of_backward=n_backward_inputs,
        )
    torch.library.register_fake(dbwd_name, double_backward_fake)
    dbwd_callable = getattr(
        getattr(torch.ops, namespace),
        f"{base_name}_double_backward",
    )
    out["double_backward"] = dbwd_callable

    attach_simple_backward(
        bwd_name,
        dbwd_callable,
        diff_input_positions=second_order_diff_positions,
        n_forward_inputs=n_backward_inputs,
        batch_match=batch_match,
        backward_args=second_order_backward_args,
    )

    return out


__all__ = [
    "_match_shape",
    "_match_shape_batch",
    "attach_simple_backward",
    "register_warp_op_chain",
]
