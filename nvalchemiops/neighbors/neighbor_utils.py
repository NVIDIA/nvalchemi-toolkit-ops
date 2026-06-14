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

"""Core warp utilities for neighbor list construction.

This module contains warp kernels and launchers for neighbor list operations.
See `nvalchemiops.torch.neighbors` for PyTorch bindings.
"""

import contextlib
import math
import warnings
from functools import lru_cache
from typing import Any

import warp as wp

DTYPE_INFO_ALL: dict[type, tuple[type, type]] = {
    wp.float16: (wp.vec3h, wp.mat33h),
    wp.float32: (wp.vec3f, wp.mat33f),
    wp.float64: (wp.vec3d, wp.mat33d),
}

_DTYPE_NAME: dict[type, str] = {
    wp.float16: "f16",
    wp.float32: "f32",
    wp.float64: "f64",
}


def wp_device_str(device) -> str:
    """Return Warp's canonical device alias for cache keys and allocations."""
    return str(wp.get_device(device))


def require_supported_dtype(
    wp_dtype: type, allowed: tuple[type, ...] | None = None
) -> None:
    """Validate that ``wp_dtype`` is a supported Warp scalar dtype."""
    allowed_dtypes = tuple(DTYPE_INFO_ALL) if allowed is None else allowed
    if wp_dtype not in allowed_dtypes:
        names = ", ".join(str(dtype) for dtype in allowed_dtypes)
        raise ValueError(f"Unsupported dtype {wp_dtype!r}; expected one of: {names}")


def dtype_info(
    wp_dtype: type, allowed: tuple[type, ...] | None = None
) -> tuple[type, type]:
    """Return ``(vec_dtype, mat_dtype)`` for a supported scalar dtype."""
    require_supported_dtype(wp_dtype, allowed)
    return DTYPE_INFO_ALL[wp_dtype]


def kernel_specialization_name(
    base: str,
    *,
    wp_dtype: type | None = None,
    features: tuple[str, ...] = (),
) -> str:
    """Return a stable name for a factory-created Warp specialization."""
    tokens = tuple(str(feature) for feature in features if feature)
    name = str(base)
    if tokens:
        name = f"{name}__{'_'.join(tokens)}"
    if wp_dtype is not None:
        require_supported_dtype(wp_dtype)
        name = f"{name}__{_DTYPE_NAME[wp_dtype]}"
    return name


def set_fn_name(fn: Any, name: str) -> Any:
    """Set Python- and Warp-visible names on a generated function object."""
    fn.__name__ = name
    fn.__qualname__ = name
    if hasattr(fn, "key"):
        old_key = fn.key
        fn.key = name
        # If this is a Warp kernel/function registered in a module, update its registration key
        if hasattr(fn, "module") and fn.module is not None:
            if hasattr(fn.module, "kernels") and old_key in fn.module.kernels:
                del fn.module.kernels[old_key]
                fn.module.kernels[name] = fn
            elif hasattr(fn.module, "functions") and old_key in fn.module.functions:
                del fn.module.functions[old_key]
                fn.module.functions[name] = fn
            # If this is a unique module, rename the module itself to match the new kernel name
            if hasattr(fn.module, "name") and fn.module.name:
                old_module_name = fn.module.name
                if old_key in old_module_name:
                    new_module_name = old_module_name.replace(old_key, name)
                    from warp._src.context import user_modules

                    if old_module_name in user_modules:
                        del user_modules[old_module_name]
                        user_modules[new_module_name] = fn.module
                    fn.module.name = new_module_name
            # Clear module hashers cache to force hash re-evaluation and rebuild on all platforms
            if hasattr(fn.module, "hashers"):
                fn.module.hashers.clear()
        # Force recomputing kernel hash using the new key if it has one
        # (best-effort: a Warp-internal change here must not break naming).
        if hasattr(fn, "hash"):
            with contextlib.suppress(Exception):
                opts = (
                    fn.module.options
                    if (hasattr(fn, "module") and fn.module is not None)
                    else {}
                )
                hasher = wp._src.context.ModuleHasher([], opts)
                fn.hash = hasher.hash_kernel(fn)
    wrapped = getattr(fn, "func", None)
    if wrapped is not None:
        wrapped.__name__ = name
        wrapped.__qualname__ = name
    return fn


def set_fn_doc(fn: Any, doc: str) -> Any:
    """Set Python- and Warp-visible docs on a generated function object."""
    fn.__doc__ = doc
    if hasattr(fn, "doc"):
        fn.doc = doc
    wrapped = getattr(fn, "func", None)
    if wrapped is not None:
        wrapped.__doc__ = doc
    return fn


def _append_specialization_doc(
    base_doc: str | None,
    *,
    dtype: type | str | None = None,
    entries: tuple[tuple[str, object], ...] = (),
) -> str:
    """Append runtime specialization metadata to a source docstring."""
    doc = (base_doc or "").rstrip()
    lines = ["", "Specialization", "--------------"]
    if dtype is not None:
        dtype_value = _DTYPE_NAME.get(dtype, dtype)
        lines.append(f"dtype : {dtype_value}")
    for name, value in entries:
        lines.append(f"{name} : {value}")
    return f"{doc}\n" + "\n".join(lines)


def resolve_buffer_alias(new_name, new_value, old_name, old_value):
    """Resolve a deprecated scratch-buffer kwarg alias.

    Returns the active value, emitting a :class:`DeprecationWarning` if the
    caller used the old unsuffixed name.  Raises ``ValueError`` if both names
    are populated.
    """
    if old_value is None:
        return new_value
    warnings.warn(
        f"The {old_name!r} kwarg is deprecated; use {new_name!r} instead.",
        DeprecationWarning,
        stacklevel=3,
    )
    if new_value is not None:
        raise ValueError(f"Pass either {new_name!r} or {old_name!r}, not both.")
    return old_value


def empty_sentinel(ndim: int, dtype: type, device) -> wp.array:
    """Return a cached zero-size sentinel array for ``ndim``/``dtype``/``device``."""
    return _empty_sentinel_cached(int(ndim), dtype, wp_device_str(device))


@lru_cache(maxsize=None)
def _empty_sentinel_cached(ndim: int, dtype: type, device: str) -> wp.array:
    """Allocate a cached zero-size sentinel array for a canonical device alias."""
    return wp.empty((0,) * ndim, dtype=dtype, device=device)


_empty_sentinel = empty_sentinel


class NeighborOverflowError(Exception):
    """Exception raised when a neighbor output exceeds its capacity.

    This error indicates that a pre-allocated neighbor matrix or COO segment
    is too small to hold all discovered neighbors. Users should increase the
    relevant ``max_neighbors`` / segment-capacity parameter or provide a
    larger pre-allocated tensor.

    Parameters
    ----------
    max_neighbors : int
        The maximum number of neighbors or COO entries the output can hold.
    num_neighbors : int
        The actual number of neighbors or COO entries found.
    system_index : int, optional
        System index for segmented batched outputs.
    """

    def __init__(
        self, max_neighbors: int, num_neighbors: int, system_index: int | None = None
    ):
        if system_index is None:
            message = (
                "The number of neighbors is larger than the maximum allowed: "
                f"{num_neighbors} > {max_neighbors}."
            )
        else:
            message = (
                f"The number of neighbors in segment {system_index} is larger "
                f"than the maximum allowed: {num_neighbors} > {max_neighbors}."
            )
        super().__init__(message)
        self.max_neighbors = max_neighbors
        self.num_neighbors = num_neighbors
        self.system_index = system_index


__all__ = [
    "DTYPE_INFO_ALL",
    "NeighborOverflowError",
    "dtype_info",
    "empty_sentinel",
    "compute_naive_num_shifts",
    "compute_inv_cells",
    "estimate_max_neighbors",
    "fill_neighbor_matrix_tail",
    "get_compute_inv_cells_kernel",
    "get_compute_naive_num_shifts_kernel",
    "get_gather_positions_and_shifts_kernel",
    "get_update_ref_positions_kernel",
    "get_wrap_positions_kernel",
    "kernel_specialization_name",
    "require_supported_dtype",
    "resolve_buffer_alias",
    "selective_zero_num_neighbors",
    "selective_zero_num_neighbors_single",
    "set_fn_name",
    "update_ref_positions",
    "update_ref_positions_batch",
    "wrap_positions_single",
    "wrap_positions_batch",
    "wp_device_str",
    "zero_array",
]


def zero_array(array: wp.array, device: str) -> None:
    """Zero all elements of a Warp array in place.

    .. deprecated::
        Use ``array.zero_()`` directly.  This shim forwards to it and will be
        removed in a future release.

    Parameters
    ----------
    array : wp.array, dtype=Any
        OUTPUT: Array to be zeroed in place.
    device : str
        Accepted for backward compatibility and ignored; ``array.zero_()``
        runs on the array's own device.
    """
    warnings.warn(
        "nvalchemiops.neighbors.zero_array is deprecated; use array.zero_() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    del device  # retained only for signature compatibility
    array.zero_()


@wp.func
def _decode_shift_index(local_idx: int, shift_range: wp.vec3i) -> wp.vec3i:
    """Decode a flat shift index into (kx, ky, kz) lattice shift vector

    Decodes the half-shell enumeration used by the naive PBC kernels so
    shift vectors can be computed on-the-fly without materialising the
    full shifts array.

    Parameters
    ----------
    local_idx : int
        Zero-based index into the per-system shift enumeration.
    shift_range : wp.vec3i
        Shift range in each dimension (from ``_compute_naive_num_shifts``).

    Returns
    -------
    wp.vec3i
        The integer lattice shift vector ``(kx, ky, kz)``.
    """
    k2_size = 2 * shift_range[2] + 1
    k1_size = 2 * shift_range[1] + 1
    group0_size = shift_range[1] * k2_size + shift_range[2] + 1

    k0 = wp.int32(0)
    k1 = wp.int32(0)
    k2 = wp.int32(0)

    if local_idx < group0_size:
        if local_idx <= shift_range[2]:
            k2 = local_idx
        else:
            rem = local_idx - (shift_range[2] + 1)
            k1 = rem / k2_size + 1
            k2 = rem % k2_size - shift_range[2]
    else:
        rem = local_idx - group0_size
        k0 = rem / (k1_size * k2_size) + 1
        rem2 = rem % (k1_size * k2_size)
        k1 = rem2 / k2_size - shift_range[1]
        k2 = rem2 % k2_size - shift_range[2]

    return wp.vec3i(k0, k1, k2)


@wp.func
def _decode_full_shift_index(local_idx: int, shift_range: wp.vec3i) -> wp.vec3i:
    """Decode a flat full-shell index into ``(kx, ky, kz)``, excluding ``(0, 0, 0)``

    Companion to :func:`_decode_shift_index`.  Enumerates the FULL sphere
    of shift vectors at radius ``shift_range`` (not the half-shell), in
    the natural Cartesian order (``k0`` outer, ``k2`` inner), skipping the
    self entry at the centre.

    Parameters
    ----------
    local_idx : int
        Zero-based index in ``[0, (2*Rx+1)*(2*Ry+1)*(2*Rz+1) - 1)``.
    shift_range : wp.vec3i
        Per-axis radius ``(Rx, Ry, Rz)``.

    Returns
    -------
    wp.vec3i
        The integer lattice shift vector ``(kx, ky, kz)`` with
        ``-Rx <= kx <= Rx`` etc., and never ``(0, 0, 0)``.
    """
    k1_size = 2 * shift_range[1] + 1
    k2_size = 2 * shift_range[2] + 1
    plane = k1_size * k2_size
    self_pos = shift_range[0] * plane + shift_range[1] * k2_size + shift_range[2]

    raw_idx = local_idx
    if local_idx >= self_pos:
        raw_idx = local_idx + 1

    k0 = raw_idx / plane - shift_range[0]
    rem = raw_idx % plane
    k1 = rem / k2_size - shift_range[1]
    k2 = rem % k2_size - shift_range[2]
    return wp.vec3i(k0, k1, k2)


@wp.func
def _shifted_position(shift: wp.vec3i, cell: Any, position: Any):
    """Position translated by lattice shift ``shift`` under cell matrix ``cell``

    Parameters
    ----------
    shift : wp.vec3i
        Integer lattice shift vector.
    cell : wp.mat33*
        Cell matrix used to convert ``shift`` to Cartesian displacement.
    position : wp.vec3*
        Cartesian position to translate.

    Returns
    -------
    wp.vec3*
        Translated Cartesian position.

    Notes
    -----
    The cell-element scalar type is recovered from ``cell[0]`` so the cast of
    ``shift`` matches the position dtype.
    """
    return type(cell[0])(shift) * cell + position


@wp.func
def _update_dual_neighbor_matrix(
    i: int,
    j: int,
    dist_sq: Any,
    cutoff1_sq: Any,
    cutoff2_sq: Any,
    neighbor_matrix1: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts1: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors1: wp.array(dtype=wp.int32),
    max_neighbors1: int,
    neighbor_matrix2: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts2: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors2: wp.array(dtype=wp.int32),
    max_neighbors2: int,
    unit_shift: wp.vec3i,
    half_fill: bool,
    pbc: bool,
):
    """Update primary and secondary dual-cutoff neighbor matrices

    Parameters
    ----------
    i : int
        Source atom row.
    j : int
        Neighbor atom index.
    dist_sq : float
        Squared pair distance.
    cutoff1_sq : float
        Squared primary cutoff distance.
    cutoff2_sq : float
        Squared secondary cutoff distance.
    neighbor_matrix1 : wp.array, shape (rows, max_neighbors1), dtype=wp.int32
        OUTPUT: Primary cutoff neighbor matrix.
    neighbor_matrix_shifts1 : wp.array, shape (rows, max_neighbors1), dtype=wp.vec3i
        OUTPUT: Primary cutoff shift matrix for PBC mode.
    num_neighbors1 : wp.array, shape (rows,), dtype=wp.int32
        MODIFIED: Primary cutoff neighbor counts.
    max_neighbors1 : int
        Maximum primary cutoff neighbors per row.
    neighbor_matrix2 : wp.array, shape (rows, max_neighbors2), dtype=wp.int32
        OUTPUT: Secondary cutoff neighbor matrix.
    neighbor_matrix_shifts2 : wp.array, shape (rows, max_neighbors2), dtype=wp.vec3i
        OUTPUT: Secondary cutoff shift matrix for PBC mode.
    num_neighbors2 : wp.array, shape (rows,), dtype=wp.int32
        MODIFIED: Secondary cutoff neighbor counts.
    max_neighbors2 : int
        Maximum secondary cutoff neighbors per row.
    unit_shift : wp.vec3i
        Periodic unit shift stored with PBC neighbor pairs.
    half_fill : bool
        If True, store only one direction for each unordered pair.
    pbc : bool
        If True, write shift matrices alongside atom indices.

    Returns
    -------
    None
        This function modifies the input arrays in-place.

    Notes
    -----
    - Modifies: neighbor matrices, shift matrices in PBC mode, and neighbor counts.
    """
    if dist_sq < cutoff2_sq:
        _update_neighbor_matrix(
            i,
            j,
            neighbor_matrix2,
            neighbor_matrix_shifts2,
            num_neighbors2,
            unit_shift,
            max_neighbors2,
            half_fill,
            pbc,
        )
        if dist_sq < cutoff1_sq:
            _update_neighbor_matrix(
                i,
                j,
                neighbor_matrix1,
                neighbor_matrix_shifts1,
                num_neighbors1,
                unit_shift,
                max_neighbors1,
                half_fill,
                pbc,
            )


@wp.func
def _correct_shift(
    shift: wp.vec3i,
    offset_i: wp.vec3i,
    offset_j: wp.vec3i,
) -> wp.vec3i:
    """Apply wrap-on-entry shift correction

    Parameters
    ----------
    shift : wp.vec3i
        Periodic image shift before wrap-on-entry correction.
    offset_i : wp.vec3i
        Integer cell offset for the source atom.
    offset_j : wp.vec3i
        Integer cell offset for the neighbor atom.

    Returns
    -------
    wp.vec3i
        Corrected periodic shift vector.

    Notes
    -----
    The returned shift is adjusted by ``offset_i - offset_j`` so the reconstructed
    displacement matches the original unwrapped geometry.
    """
    return wp.vec3i(
        shift[0] - offset_j[0] + offset_i[0],
        shift[1] - offset_j[1] + offset_i[1],
        shift[2] - offset_j[2] + offset_i[2],
    )


@wp.func
def _update_neighbor_matrix(
    i: int,
    j: int,
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2),
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2),
    num_neighbors: wp.array(dtype=wp.int32),
    unit_shift: wp.vec3i,
    max_neighbors: int,
    half_fill: bool,
    pbc: bool,
):
    """Update the neighbor matrix with the given atom indices

    Parameters
    ----------
    i: int
        The index of the source atom.
    j: int
        The index of the target atom.
    neighbor_matrix: wp.array(dtype=wp.int32, ndim=2)
        OUTPUT: The neighbor matrix to be updated.
    neighbor_matrix_shifts: wp.array(dtype=wp.vec3i, ndim=2)
        OUTPUT: The neighbor matrix shifts to be updated when ``pbc`` is true.
    num_neighbors: wp.array(dtype=wp.int32)
        OUTPUT: The number of neighbors for each atom.
    unit_shift: wp.vec3i
        The unit shift vector for the periodic boundary.
    max_neighbors: int
        The maximum number of neighbors for each atom.
    half_fill: bool
        If True, only fill half of the neighbor matrix.
    pbc: bool
        If True, write periodic shift entries alongside atom indices.

    Returns
    -------
    None
        This function modifies the input arrays in-place.
    """
    pos = wp.atomic_add(num_neighbors, i, 1)
    if pos < max_neighbors:
        neighbor_matrix[i, pos] = j
        if pbc:
            neighbor_matrix_shifts[i, pos] = unit_shift
    if not half_fill and (pbc or i < j):
        pos = wp.atomic_add(num_neighbors, j, 1)
        if pos < max_neighbors:
            neighbor_matrix[j, pos] = i
            if pbc:
                neighbor_matrix_shifts[j, pos] = -unit_shift


@wp.kernel(enable_backward=False)
def _compute_naive_num_shifts(
    cell: wp.array(dtype=Any),
    cutoff: Any,
    pbc: wp.array2d(dtype=wp.bool),
    num_shifts: wp.array(dtype=int),
    shift_range: wp.array(dtype=wp.vec3i),
) -> None:
    """Compute periodic image shifts needed for neighbor searching

    Calculates the number and range of periodic boundary shifts required
    to ensure all atoms within the cutoff distance are found, taking into
    account the geometry of the simulation cell and minimum image convention.

    Parameters
    ----------
    cell : wp.array, shape (num_systems, 3, 3), dtype=wp.mat33*
        Cell matrices defining lattice vectors in Cartesian coordinates.
        Each 3x3 matrix represents one system's periodic cell.
    cutoff : float
        Cutoff distance for neighbor searching in Cartesian units.
        Must be positive and typically less than half the minimum cell dimension.
    pbc : wp.array, shape (num_systems, 3), dtype=wp.bool
        Periodic boundary condition flags for each dimension.
        True enables periodicity in that direction.
    num_shifts : wp.array, shape (num_systems,), dtype=int
        OUTPUT: Total number of periodic shifts needed for each system.
        Updated with calculated shift counts.
    shift_range : wp.array, shape (num_systems, 3), dtype=wp.vec3i
        OUTPUT: Maximum shift indices in each dimension for each system.
        Updated with calculated shift ranges.

    Returns
    -------
    None
        This function modifies the input arrays in-place:

        - num_shifts : Updated with total shift counts per system
        - shift_range : Updated with shift ranges per dimension

    Notes
    -----
    - Thread launch: see launcher-specific launch dimension.
    - Modifies: see OUTPUT or MODIFIED parameters.

    See Also
    --------
    get_compute_naive_num_shifts_kernel : Return the specialized periodic-image shift-count kernel.
    """
    tid = wp.tid()

    _cell = cell[tid]
    _pbc = pbc[tid]

    _cell_inv = wp.transpose(wp.inverse(_cell))
    _d_inv_0 = wp.length(_cell_inv[0]) if _pbc[0] else type(_cell_inv[0, 0])(0.0)
    _d_inv_1 = wp.length(_cell_inv[1]) if _pbc[1] else type(_cell_inv[1, 0])(0.0)
    _d_inv_2 = wp.length(_cell_inv[2]) if _pbc[2] else type(_cell_inv[2, 0])(0.0)
    _s = wp.vec3i(
        wp.int32(wp.ceil(_d_inv_0 * type(_d_inv_0)(cutoff))),
        wp.int32(wp.ceil(_d_inv_1 * type(_d_inv_1)(cutoff))),
        wp.int32(wp.ceil(_d_inv_2 * type(_d_inv_2)(cutoff))),
    )
    k1 = 2 * _s[1] + 1
    k2 = 2 * _s[2] + 1
    shift_range[tid] = _s
    num_shifts[tid] = _s[0] * k1 * k2 + _s[1] * k2 + _s[2] + 1


@lru_cache(maxsize=None)
def get_compute_naive_num_shifts_kernel(wp_dtype: type) -> wp.Kernel:
    """Return the specialized periodic-image shift-count kernel."""
    _vec_dtype, mat_dtype = dtype_info(wp_dtype)
    kernel = wp.overload(
        _compute_naive_num_shifts,
        [
            wp.array(dtype=mat_dtype),
            wp_dtype,
            wp.array2d(dtype=wp.bool),
            wp.array(dtype=int),
            wp.array(dtype=wp.vec3i),
        ],
    )
    name = kernel_specialization_name("_compute_naive_num_shifts", wp_dtype=wp_dtype)
    return set_fn_doc(
        set_fn_name(kernel, name),
        _append_specialization_doc(
            kernel.__doc__,
            dtype=wp_dtype,
            entries=(("operation", "compute_naive_num_shifts"),),
        ),
    )


def _make_selective_zero_num_neighbors_kernel(*, batched: bool):
    """Build the selective ``num_neighbors`` zeroing kernel."""
    BATCHED = wp.constant(bool(batched))

    @wp.kernel(enable_backward=False, module="unique")
    def _kernel(
        num_neighbors: wp.array(dtype=wp.int32),
        batch_idx: wp.array(dtype=wp.int32),
        rebuild_flags: wp.array(dtype=wp.bool),
    ) -> None:
        """Zero neighbor counts for atoms in rebuilt systems

        Parameters
        ----------
        num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
            OUTPUT: Per-atom neighbor counts to zero selectively.
        batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
            System index for each atom. Zero-size sentinel in single-system
            specializations.
        rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool
            Rebuild flags controlling which atoms have their counts reset.

        Returns
        -------
        None
            This function modifies the input arrays in-place.

        Notes
        -----
        - Thread launch: One thread per atom.
        - Modifies: ``num_neighbors`` entries for rebuilt systems.
        ``BATCHED`` is a static specialization. Single-system launchers pass a
        zero-size ``batch_idx`` sentinel that is not read.

        See Also
        --------
        _get_selective_zero_num_neighbors_kernel : Return the specialized selective neighbor-count zeroing kernel.
        """
        tid = wp.tid()
        isys = wp.int32(0)
        if BATCHED:
            isys = batch_idx[tid]
        if rebuild_flags[isys]:
            num_neighbors[tid] = 0

    base = (
        "_selective_zero_num_neighbors"
        if batched
        else "_selective_zero_num_neighbors_single"
    )
    name = kernel_specialization_name(base)
    return set_fn_doc(
        set_fn_name(_kernel, name),
        _append_specialization_doc(
            _kernel.__doc__,
            entries=(
                ("batched", bool(batched)),
                ("operation", "selective_zero_num_neighbors"),
            ),
        ),
    )


@lru_cache(maxsize=None)
def _get_selective_zero_num_neighbors_kernel(*, batched: bool) -> wp.Kernel:
    """Return the selective ``num_neighbors`` zeroing kernel for batching."""
    return _make_selective_zero_num_neighbors_kernel(batched=bool(batched))


def selective_zero_num_neighbors(
    num_neighbors: wp.array,
    batch_idx: wp.array,
    rebuild_flags: wp.array,
    device: str,
) -> None:
    """Core warp launcher for selectively zeroing num_neighbors.

    Zeros the num_neighbors count for atoms belonging to systems where
    rebuild_flags is True, preserving counts for non-rebuilt systems.

    Parameters
    ----------
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Per-atom neighbor counts; selectively zeroed.
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        System index for each atom.
    rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool
        Per-system flags indicating which systems need rebuilding.
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').

    See Also
    --------
    _get_selective_zero_num_neighbors_kernel : Selects the batched kernel
    """
    total_atoms = num_neighbors.shape[0]
    wp.launch(
        kernel=_get_selective_zero_num_neighbors_kernel(batched=True),
        dim=total_atoms,
        inputs=[num_neighbors, batch_idx, rebuild_flags],
        device=device,
    )


def selective_zero_num_neighbors_single(
    num_neighbors: wp.array,
    rebuild_flags: wp.array,
    device: str,
) -> None:
    """Core warp launcher for selectively zeroing num_neighbors for a single system.

    Zeros all num_neighbors entries when rebuild_flags[0] is True.  When False
    the kernel returns immediately — no CPU-GPU synchronization occurs.

    Parameters
    ----------
    num_neighbors : wp.array, shape (total_atoms,), dtype=wp.int32
        OUTPUT: Per-atom neighbor counts; zeroed when rebuild is needed.
    rebuild_flags : wp.array, shape (1,) or shape (), dtype=wp.bool
        Single-system rebuild flag.
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').

    See Also
    --------
    _get_selective_zero_num_neighbors_kernel : Selects the single-system kernel
    selective_zero_num_neighbors : Batch variant using per-atom batch_idx
    """
    total_atoms = num_neighbors.shape[0]
    wp.launch(
        kernel=_get_selective_zero_num_neighbors_kernel(batched=False),
        dim=total_atoms,
        inputs=[
            num_neighbors,
            _empty_sentinel(1, wp.int32, device),
            rebuild_flags,
        ],
        device=device,
    )


@wp.kernel(enable_backward=False)
def _compute_inv_cells_kernel(
    cell: wp.array(dtype=Any),
    inv_cell: wp.array(dtype=Any),
) -> None:
    """Compute the inverse of each cell matrix

    Parameters
    ----------
    cell : wp.array, shape (num_systems,), dtype=wp.mat33*
        Input cell matrices.
    inv_cell : wp.array, shape (num_systems,), dtype=wp.mat33*
        OUTPUT: Inverse of each cell matrix.

    Returns
    -------
    None
        This function modifies the input arrays in-place.

    Notes
    -----
    - Modifies: see OUTPUT or MODIFIED parameters.
    - Thread launch: One thread per system (dim=num_systems)

    See Also
    --------
    get_compute_inv_cells_kernel : Return the specialized inverse-cell kernel.
    """
    tid = wp.tid()
    inv_cell[tid] = wp.inverse(cell[tid])


@lru_cache(maxsize=None)
def get_compute_inv_cells_kernel(wp_dtype: type) -> wp.Kernel:
    """Return the specialized inverse-cell kernel."""
    _vec_dtype, mat_dtype = dtype_info(wp_dtype)
    kernel = wp.overload(
        _compute_inv_cells_kernel,
        [wp.array(dtype=mat_dtype), wp.array(dtype=mat_dtype)],
    )
    name = kernel_specialization_name("_compute_inv_cells_kernel", wp_dtype=wp_dtype)
    return set_fn_doc(
        set_fn_name(kernel, name),
        _append_specialization_doc(
            kernel.__doc__,
            dtype=wp_dtype,
            entries=(("operation", "compute_inv_cells"),),
        ),
    )


def compute_inv_cells(
    cell: wp.array,
    inv_cell: wp.array,
    wp_dtype: type,
    device: str,
) -> None:
    """Core warp launcher for computing inverse cell matrices.

    Inverts each cell matrix in the batch using pure warp operations.
    Call this once before launching naive PBC neighbor-list kernels to
    avoid redundant per-thread inversions inside those kernels.

    Parameters
    ----------
    cell : wp.array, shape (num_systems,), dtype=wp.mat33*
        Input cell matrices.
    inv_cell : wp.array, shape (num_systems,), dtype=wp.mat33*
        OUTPUT: Inverse of each cell matrix. Must be pre-allocated
        with the same shape and dtype as *cell*.
    wp_dtype : type
        Warp scalar dtype (wp.float32, wp.float64, or wp.float16).
    device : str
        Warp device string (e.g., ``'cuda:0'``, ``'cpu'``).

    See Also
    --------
    get_compute_inv_cells_kernel : Factory-selected inverse-cell kernel
    """
    num_systems = cell.shape[0]
    wp.launch(
        kernel=get_compute_inv_cells_kernel(wp_dtype),
        dim=num_systems,
        inputs=[cell, inv_cell],
        device=device,
    )


def compute_naive_num_shifts(
    cell: wp.array,
    cutoff: float,
    pbc: wp.array,
    num_shifts: wp.array,
    shift_range: wp.array,
    wp_dtype: type,
    device: str,
) -> None:
    """Core warp launcher for computing periodic image shifts.

    Calculates the number and range of periodic boundary shifts required
    to ensure all atoms within the cutoff distance are found, using pure
    warp operations.

    Parameters
    ----------
    cell : wp.array, shape (num_systems, 3, 3), dtype=wp.mat33*
        Cell matrices defining lattice vectors in Cartesian coordinates.
        Each 3x3 matrix represents one system's periodic cell.
    cutoff : float
        Cutoff distance for neighbor searching in Cartesian units.
        Must be positive and typically less than half the minimum cell dimension.
    pbc : wp.array, shape (num_systems, 3), dtype=wp.bool
        Periodic boundary condition flags for each dimension.
        True enables periodicity in that direction.
    num_shifts : wp.array, shape (num_systems,), dtype=wp.int32
        OUTPUT: Total number of periodic shifts needed for each system.
        Updated with calculated shift counts.
    shift_range : wp.array, shape (num_systems, 3), dtype=wp.vec3i
        OUTPUT: Maximum shift indices in each dimension for each system.
        Updated with calculated shift ranges.
    wp_dtype : type
        Warp dtype (wp.float32, wp.float64, or wp.float16).
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').

    Notes
    -----
    - This is a low-level warp interface. For framework bindings, use torch/jax wrappers.
    - Output arrays (num_shifts, shift_range) must be pre-allocated by caller.

    See Also
    --------
    get_compute_naive_num_shifts_kernel : Factory-selected shift-count kernel
    """
    num_systems = cell.shape[0]

    wp.launch(
        kernel=get_compute_naive_num_shifts_kernel(wp_dtype),
        dim=num_systems,
        inputs=[
            cell,
            wp_dtype(cutoff),
            pbc,
            num_shifts,
            shift_range,
        ],
        device=device,
    )


def estimate_max_neighbors(
    cutoff: float,
    atomic_density: float = 0.2,
    safety_factor: float = 1.0,
) -> int:
    r"""Estimate maximum neighbors per atom based on volume calculations.

    Uses atomic density and cutoff volume to estimate a conservative upper bound
    on the number of neighbors any atom could have. This is a pure Python function
    with no framework dependencies.

    Parameters
    ----------
    cutoff : float
        Maximum distance for considering atoms as neighbors.
    atomic_density : float, optional
        Atomic density in atoms per unit volume. Default is 0.2.
    safety_factor : float
        Safety factor to multiply the estimated number of neighbors. Default is 1.0.

    Returns
    -------
    max_neighbors_estimate : int
        Conservative estimate of maximum neighbors per atom. Returns 0 for
        empty systems.

    Notes
    -----
    The estimation uses the formula:

    .. math::

        \text{neighbors} = \text{safety\_factor} \times \text{density} \times V_{\text{sphere}}

    where the cutoff sphere volume is:

    .. math::

        V_{\text{sphere}} = \frac{4}{3}\pi r^3

    The result is rounded up to the multiple of 16 for memory alignment.
    """
    if cutoff <= 0:
        return 0
    cutoff_sphere_volume = atomic_density * (4.0 / 3.0) * math.pi * (cutoff**3)

    # Estimate neighbors based on density and cutoff volume
    expected_neighbors = max(1, safety_factor * cutoff_sphere_volume)

    # Round up to multiple of 16 for memory alignment and safety
    max_neighbors_estimate = int(math.ceil(expected_neighbors / 16)) * 16
    return max_neighbors_estimate


###########################################################################################
########################### Position Wrapping Kernels ####################################
###########################################################################################


def _make_wrap_positions_kernel(
    wp_dtype: type, *, batched: bool, pbc_aware: bool = False
):
    """Build a position-wrapping kernel for one dtype and batching mode."""
    require_supported_dtype(wp_dtype)
    vec_dtype, mat_dtype = dtype_info(wp_dtype)
    BATCHED = wp.constant(bool(batched))

    if pbc_aware:

        @wp.kernel(enable_backward=False, module="unique")
        def _kernel(
            positions: wp.array(dtype=vec_dtype),
            cell: wp.array(dtype=mat_dtype),
            inv_cell: wp.array(dtype=mat_dtype),
            pbc: wp.array2d(dtype=wp.bool),
            batch_idx: wp.array(dtype=wp.int32),
            positions_wrapped: wp.array(dtype=vec_dtype),
            per_atom_cell_offsets: wp.array(dtype=wp.vec3i),
        ) -> None:
            """Wrap positions into periodic axes and store integer offsets.

            Notes
            -----
            - Thread launch: One thread per atom.
            - Modifies: ``positions_wrapped`` and ``per_atom_cell_offsets``.
            """
            i = wp.tid()
            isys = wp.int32(0)
            if BATCHED:
                isys = batch_idx[i]
            _cell = cell[isys]
            _inv_cell = inv_cell[isys]
            _pbc = pbc[isys]
            _pos = positions[i]
            _frac = _pos * _inv_cell
            _int = wp.vec3i(
                wp.int32(wp.floor(_frac[0])) if _pbc[0] else wp.int32(0),
                wp.int32(wp.floor(_frac[1])) if _pbc[1] else wp.int32(0),
                wp.int32(wp.floor(_frac[2])) if _pbc[2] else wp.int32(0),
            )
            positions_wrapped[i] = _pos - type(_pos)(_int) * _cell
            per_atom_cell_offsets[i] = _int

    else:

        @wp.kernel(enable_backward=False, module="unique")
        def _kernel(
            positions: wp.array(dtype=vec_dtype),
            cell: wp.array(dtype=mat_dtype),
            inv_cell: wp.array(dtype=mat_dtype),
            batch_idx: wp.array(dtype=wp.int32),
            positions_wrapped: wp.array(dtype=vec_dtype),
            per_atom_cell_offsets: wp.array(dtype=wp.vec3i),
        ) -> None:
            """Wrap positions into the primary cell and store integer offsets.

            Notes
            -----
            - Thread launch: One thread per atom.
            - Modifies: ``positions_wrapped`` and ``per_atom_cell_offsets``.
            """
            i = wp.tid()
            isys = wp.int32(0)
            if BATCHED:
                isys = batch_idx[i]
            _cell = cell[isys]
            _inv_cell = inv_cell[isys]
            _pos = positions[i]
            _frac = _pos * _inv_cell
            _int = wp.vec3i(
                wp.int32(wp.floor(_frac[0])),
                wp.int32(wp.floor(_frac[1])),
                wp.int32(wp.floor(_frac[2])),
            )
            positions_wrapped[i] = _pos - type(_pos)(_int) * _cell
            per_atom_cell_offsets[i] = _int

    base = (
        "_wrap_positions_batch_kernel" if batched else "_wrap_positions_single_kernel"
    )
    name = kernel_specialization_name(base, wp_dtype=wp_dtype)
    return set_fn_doc(
        set_fn_name(_kernel, name),
        _append_specialization_doc(
            _kernel.__doc__,
            dtype=wp_dtype,
            entries=(
                ("batched", bool(batched)),
                ("pbc_aware", bool(pbc_aware)),
            ),
        ),
    )


@lru_cache(maxsize=None)
def get_wrap_positions_kernel(
    wp_dtype: type, *, batched: bool = False, pbc_aware: bool = False
) -> wp.Kernel:
    """Return the specialized position-wrapping kernel.

    Parameters
    ----------
    wp_dtype : type
        Warp scalar dtype (wp.float32, wp.float64, or wp.float16).
    batched : bool, optional
        Whether to build the batched kernel variant.
    pbc_aware : bool, optional
        If ``False``, build the existing fold-all-axes kernel signature. If
        ``True``, build the variant that accepts ``pbc`` and skips wrapping on
        non-periodic axes.

    Returns
    -------
    wp.Kernel
        Specialized position-wrapping kernel.
    """
    return _make_wrap_positions_kernel(
        wp_dtype, batched=bool(batched), pbc_aware=bool(pbc_aware)
    )


def _launch_wrap_positions(
    positions: wp.array,
    cell: wp.array,
    inv_cell: wp.array,
    pbc: wp.array | None,
    batch_idx: wp.array,
    positions_wrapped: wp.array,
    per_atom_cell_offsets: wp.array,
    wp_dtype: type,
    device: str,
    *,
    batched: bool,
) -> None:
    """Launch the shared position-wrapping kernel."""
    pbc_aware = pbc is not None
    inputs = [positions, cell, inv_cell]
    if pbc_aware:
        inputs.append(pbc)
    inputs.extend(
        [
            batch_idx if batched else _empty_sentinel(1, wp.int32, device),
            positions_wrapped,
            per_atom_cell_offsets,
        ]
    )
    wp.launch(
        kernel=get_wrap_positions_kernel(
            wp_dtype, batched=batched, pbc_aware=pbc_aware
        ),
        dim=positions.shape[0],
        inputs=inputs,
        device=device,
    )


def wrap_positions_single(
    positions: wp.array,
    cell: wp.array,
    inv_cell: wp.array,
    positions_wrapped: wp.array,
    per_atom_cell_offsets: wp.array,
    wp_dtype: type,
    device: str,
    pbc: wp.array | None = None,
) -> None:
    """Core warp launcher for wrapping positions into the primary cell (single system).

    Computes per-atom integer cell offsets and wrapped positions in a single
    GPU pass. Call this before naive PBC neighbor-list kernels to move the
    wrapping out of the hot ishift × iatom loop.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        Atomic coordinates in Cartesian space. May be unwrapped.
    cell : wp.array, shape (1,), dtype=wp.mat33*
        Cell matrix defining lattice vectors.
    inv_cell : wp.array, shape (1,), dtype=wp.mat33*
        Pre-computed inverse cell matrix. Must be pre-allocated with the
        same shape and dtype as *cell*.
    positions_wrapped : wp.array, shape (total_atoms,), dtype=wp.vec3*
        OUTPUT: Wrapped positions. Must be pre-allocated with the same shape
        and dtype as *positions*.
    per_atom_cell_offsets : wp.array, shape (total_atoms,), dtype=wp.vec3i
        OUTPUT: Integer cell offsets per atom. Must be pre-allocated.
    wp_dtype : type
        Warp scalar dtype (wp.float32, wp.float64, or wp.float16).
    device : str
        Warp device string (e.g., ``'cuda:0'``, ``'cpu'``).
    pbc : wp.array, shape (1, 3), dtype=wp.bool, optional
        Per-axis periodicity flags. If omitted, all axes are wrapped.
        Non-periodic axes are left unwrapped when provided.

    See Also
    --------
    get_wrap_positions_kernel : Factory-selected wrapping kernel.
    wrap_positions_batch : Batch variant for multiple systems
    """
    _launch_wrap_positions(
        positions,
        cell,
        inv_cell,
        pbc,
        _empty_sentinel(1, wp.int32, device),
        positions_wrapped,
        per_atom_cell_offsets,
        wp_dtype,
        device,
        batched=False,
    )


def wrap_positions_batch(
    positions: wp.array,
    cell: wp.array,
    inv_cell: wp.array,
    batch_idx: wp.array,
    positions_wrapped: wp.array,
    per_atom_cell_offsets: wp.array,
    wp_dtype: type,
    device: str,
    pbc: wp.array | None = None,
) -> None:
    """Core warp launcher for wrapping positions into the primary cell (batch of systems).

    Each atom uses the cell matrix of its system (indexed via batch_idx).
    Computes per-atom integer cell offsets and wrapped positions in a single
    GPU pass. Call this before batch naive PBC neighbor-list kernels to move
    the wrapping out of the hot ishift × iatom loop.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        Concatenated atomic coordinates for all systems. May be unwrapped.
    cell : wp.array, shape (num_systems,), dtype=wp.mat33*
        Cell matrices for each system.
    inv_cell : wp.array, shape (num_systems,), dtype=wp.mat33*
        Pre-computed inverse cell matrices. Must be pre-allocated with the
        same shape and dtype as *cell*.
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        System index for each atom.
    positions_wrapped : wp.array, shape (total_atoms,), dtype=wp.vec3*
        OUTPUT: Wrapped positions. Must be pre-allocated with the same shape
        and dtype as *positions*.
    per_atom_cell_offsets : wp.array, shape (total_atoms,), dtype=wp.vec3i
        OUTPUT: Integer cell offsets per atom. Must be pre-allocated.
    wp_dtype : type
        Warp scalar dtype (wp.float32, wp.float64, or wp.float16).
    device : str
        Warp device string (e.g., ``'cuda:0'``, ``'cpu'``).
    pbc : wp.array, shape (num_systems, 3), dtype=wp.bool, optional
        Per-system periodicity flags. If omitted, all axes are wrapped.
        Non-periodic axes are left unwrapped when provided.

    See Also
    --------
    get_wrap_positions_kernel : Factory-selected wrapping kernel.
    wrap_positions_single : Single-system variant
    """
    _launch_wrap_positions(
        positions,
        cell,
        inv_cell,
        pbc,
        batch_idx,
        positions_wrapped,
        per_atom_cell_offsets,
        wp_dtype,
        device,
        batched=True,
    )


###########################################################################################
########################### Reference Position Update Kernels ############################
###########################################################################################


def _make_update_ref_positions_kernel(wp_dtype: type, *, batched: bool):
    """Build a conditional reference-position update kernel."""
    require_supported_dtype(wp_dtype, (wp.float32, wp.float64))
    vec_dtype, _mat_dtype = dtype_info(wp_dtype, (wp.float32, wp.float64))
    BATCHED = wp.constant(bool(batched))

    @wp.kernel(enable_backward=False, module="unique")
    def _kernel(
        positions: wp.array(dtype=vec_dtype),
        rebuild_flags: wp.array(dtype=wp.bool),
        batch_idx: wp.array(dtype=wp.int32),
        ref_positions: wp.array(dtype=vec_dtype),
    ) -> None:
        """Copy current positions into reference positions when rebuilding

        Parameters
        ----------
        positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
            Current Cartesian coordinates.
        rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool
            Rebuild flags controlling which systems update their references.
        batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
            System index for each atom. Zero-size sentinel in single-system
            specializations.
        ref_positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
            OUTPUT: Reference coordinates updated for rebuilt systems.

        Returns
        -------
        None
            This function modifies the input arrays in-place.

        Notes
        -----
        - Thread launch: One thread per atom.
        - Modifies: ``ref_positions`` entries for rebuilt systems.
        ``BATCHED`` is a static specialization. Single-system launchers pass a
        zero-size ``batch_idx`` sentinel that is not read.

        See Also
        --------
        get_update_ref_positions_kernel : Return the specialized reference-position update kernel.
        """
        i = wp.tid()
        isys = wp.int32(0)
        if BATCHED:
            isys = batch_idx[i]
        if rebuild_flags[isys]:
            ref_positions[i] = positions[i]

    base = (
        "_update_ref_positions_batch_kernel"
        if batched
        else "_update_ref_positions_kernel"
    )
    name = kernel_specialization_name(base, wp_dtype=wp_dtype)
    return set_fn_doc(
        set_fn_name(_kernel, name),
        _append_specialization_doc(
            _kernel.__doc__,
            dtype=wp_dtype,
            entries=(("batched", bool(batched)),),
        ),
    )


@lru_cache(maxsize=None)
def get_update_ref_positions_kernel(
    wp_dtype: type, *, batched: bool = False
) -> wp.Kernel:
    """Return the specialized conditional reference-position update kernel."""
    return _make_update_ref_positions_kernel(wp_dtype, batched=bool(batched))


def _launch_update_ref_positions(
    positions: wp.array,
    rebuild_flags: wp.array,
    batch_idx: wp.array,
    ref_positions: wp.array,
    wp_dtype: type,
    device: str,
    *,
    batched: bool,
) -> None:
    """Launch the shared conditional reference-position update kernel."""
    wp.launch(
        kernel=get_update_ref_positions_kernel(wp_dtype, batched=batched),
        dim=positions.shape[0],
        inputs=[
            positions,
            rebuild_flags,
            batch_idx if batched else _empty_sentinel(1, wp.int32, device),
            ref_positions,
        ],
        device=device,
    )


def update_ref_positions(
    positions: wp.array,
    rebuild_flag: wp.array,
    ref_positions: wp.array,
    wp_dtype: type,
    device: str,
) -> None:
    """Core warp launcher for conditionally updating reference positions (single system).

    Copies current positions into reference positions only when rebuild_flag[0] is True.
    No CPU-GPU synchronization required.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        Current atomic coordinates.
    rebuild_flag : wp.array, shape (1,), dtype=wp.bool
        Single-system rebuild flag.
    ref_positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        OUTPUT: Reference positions to update selectively.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').

    See Also
    --------
    get_update_ref_positions_kernel : Factory-selected update kernel.
    update_ref_positions_batch : Batch variant
    """
    _launch_update_ref_positions(
        positions,
        rebuild_flag,
        _empty_sentinel(1, wp.int32, device),
        ref_positions,
        wp_dtype,
        device,
        batched=False,
    )


def update_ref_positions_batch(
    positions: wp.array,
    rebuild_flags: wp.array,
    batch_idx: wp.array,
    ref_positions: wp.array,
    wp_dtype: type,
    device: str,
) -> None:
    """Core warp launcher for conditionally updating reference positions (batch).

    Updates reference positions only for atoms in systems where rebuild_flags is True.
    No CPU-GPU synchronization required.

    Parameters
    ----------
    positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        Current atomic coordinates for all systems.
    rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool
        Per-system rebuild flags.
    batch_idx : wp.array, shape (total_atoms,), dtype=wp.int32
        System index for each atom.
    ref_positions : wp.array, shape (total_atoms,), dtype=wp.vec3*
        OUTPUT: Reference positions to update selectively.
    wp_dtype : type
        Warp scalar dtype (wp.float32 or wp.float64).
    device : str
        Warp device string (e.g., 'cuda:0', 'cpu').

    See Also
    --------
    get_update_ref_positions_kernel : Factory-selected update kernel.
    update_ref_positions : Single-system variant
    """
    _launch_update_ref_positions(
        positions,
        rebuild_flags,
        batch_idx,
        ref_positions,
        wp_dtype,
        device,
        batched=True,
    )


# =============================================================================
# Fused gather kernels for cell-list pair-centric layout
# =============================================================================


def _make_gather_positions_and_shifts_kernel(wp_dtype: type):
    """Build the fused position/shift gather kernel for ``wp_dtype``.

    Parameters
    ----------
    wp_dtype : type
        Warp scalar dtype (``wp.float32`` or ``wp.float64``).

    Returns
    -------
    wp.Kernel
        Kernel that writes ``dst_pos[i] = src_pos[perm[i]]`` and
        ``dst_shifts[i] = src_shifts[perm[i]]``.
    """
    require_supported_dtype(wp_dtype, (wp.float32, wp.float64))
    vec_dtype, _mat_dtype = dtype_info(wp_dtype, (wp.float32, wp.float64))

    @wp.kernel(enable_backward=False, module="unique")
    def _kernel(
        src_pos: wp.array(dtype=vec_dtype),
        src_shifts: wp.array(dtype=wp.vec3i),
        perm: wp.array(dtype=wp.int32),
        dst_pos: wp.array(dtype=vec_dtype),
        dst_shifts: wp.array(dtype=wp.vec3i),
    ) -> None:
        """Gather positions and shifts under one permutation

        Parameters
        ----------
        src_pos : wp.array, shape (total_atoms,), dtype=wp.vec3*
            Source positions in original ordering.
        src_shifts : wp.array, shape (total_atoms,), dtype=wp.vec3i
            Source periodic shifts in original ordering.
        perm : wp.array, shape (total_atoms,), dtype=wp.int32
            Permutation mapping destination slots to source atom indices.
        dst_pos : wp.array, shape (total_atoms,), dtype=wp.vec3*
            OUTPUT: Gathered positions.
        dst_shifts : wp.array, shape (total_atoms,), dtype=wp.vec3i
            OUTPUT: Gathered periodic shifts.

        Returns
        -------
        None
            This function modifies the input arrays in-place.

        Notes
        -----
        - Thread launch: One thread per destination slot.
        - Modifies: ``dst_pos`` and ``dst_shifts``.

        See Also
        --------
        get_gather_positions_and_shifts_kernel : Return the specialized fused gather kernel.
        """
        i = wp.tid()
        idx = perm[i]
        dst_pos[i] = src_pos[idx]
        dst_shifts[i] = src_shifts[idx]

    name = kernel_specialization_name("_gather_positions_and_shifts", wp_dtype=wp_dtype)
    return set_fn_doc(
        set_fn_name(_kernel, name),
        _append_specialization_doc(
            _kernel.__doc__,
            dtype=wp_dtype,
            entries=(("operation", "gather_positions_and_shifts"),),
        ),
    )


@lru_cache(maxsize=None)
def get_gather_positions_and_shifts_kernel(wp_dtype: type) -> wp.Kernel:
    """Return the specialized fused position/shift gather kernel."""
    return _make_gather_positions_and_shifts_kernel(wp_dtype)


# =============================================================================
# Neighbor matrix tail-fill kernel + launcher (used by cluster_tile + cell_list)
# =============================================================================
FILL_TAIL_BLOCK_DIM = 128


def _make_fill_neighbor_matrix_tail_kernel(block_dim: int):
    """Build a tiled kernel that fills unused neighbor-matrix columns.

    Parameters
    ----------
    block_dim : int
        Static tile width used by ``wp.tile_arange`` and ``wp.launch_tiled``.

    Returns
    -------
    wp.Kernel
        Tail-fill kernel specialized to ``block_dim``.
    """
    block_dim = int(block_dim)
    if block_dim <= 0:
        raise ValueError("block_dim must be positive")
    block_dim_const = wp.constant(block_dim)

    @wp.kernel(enable_backward=False, module=f"tail_fill_block_{block_dim}")
    def _kernel(
        num_neighbors: wp.array(dtype=wp.int32),
        natom: wp.int32,
        max_neighbors: wp.int32,
        fill_value: wp.int32,
        neighbor_matrix: wp.array2d(dtype=wp.int32),
    ) -> None:
        """Fill unused neighbor-matrix columns with ``fill_value``

        Parameters
        ----------
        num_neighbors : wp.array, shape (natom,), dtype=wp.int32
            Active-slot count for each atom row.
        natom : wp.int32
            Number of atom rows to process.
        max_neighbors : wp.int32
            Number of columns in ``neighbor_matrix``.
        fill_value : wp.int32
            Value written to unused columns.
        neighbor_matrix : wp.array, shape (natom, max_neighbors), dtype=wp.int32
            OUTPUT: Neighbor matrix whose inactive tail columns are filled.

        Returns
        -------
        None
            This function modifies the input arrays in-place.

        Notes
        -----
        - Thread launch: Tiled launch with one tile per atom row.
        - Modifies: Unused columns in ``neighbor_matrix``.
        ``block_dim`` is a static specialization used by ``wp.tile_arange``.

        See Also
        --------
        fill_neighbor_matrix_tail : Launch the specialized neighbor-matrix tail fill kernel.
        """
        row = wp.tid()
        if row >= natom:
            return
        nn = num_neighbors[row]
        if nn >= max_neighbors:
            return
        lane_tile = wp.tile_arange(block_dim_const, dtype=wp.int32)
        lane = wp.untile(lane_tile)
        k = nn + lane
        while k < max_neighbors:
            neighbor_matrix[row, k] = fill_value
            k += block_dim_const

    name = kernel_specialization_name(
        "_fill_neighbor_matrix_tail",
        features=(f"block_{block_dim}",),
    )
    return set_fn_doc(
        set_fn_name(_kernel, name),
        _append_specialization_doc(
            _kernel.__doc__,
            entries=(
                ("operation", "fill_neighbor_matrix_tail"),
                ("block_dim", block_dim),
            ),
        ),
    )


@lru_cache(maxsize=None)
def _get_fill_neighbor_matrix_tail_kernel(block_dim: int):
    """Return the cached tail-fill kernel for ``block_dim``."""
    return _make_fill_neighbor_matrix_tail_kernel(int(block_dim))


def fill_neighbor_matrix_tail(
    num_neighbors: wp.array,
    natom: int,
    max_neighbors: int,
    fill_value: int,
    neighbor_matrix: wp.array,
    device: str,
    block_dim: int = FILL_TAIL_BLOCK_DIM,
) -> None:
    """Core warp launcher for coalesced tail-fill of the neighbor matrix.

    Writes ``fill_value`` into every column of ``neighbor_matrix`` that lies
    past the active-slot range ``[0, num_neighbors[i])``.  Pairs with
    always-write neighbor-matrix builders (e.g.
    :func:`nvalchemiops.neighbors.cluster_tile.query_cluster_tile`, pair-centric
    cell-list queries) so callers can skip the per-step
    ``neighbor_matrix.fill_(fill_value)`` prefill.

    Parameters
    ----------
    num_neighbors : wp.array, shape (natom,), dtype=wp.int32
        Per-atom active-slot counts.
    natom : int
        Number of atoms.
    max_neighbors : int
        Column count of ``neighbor_matrix``.
    fill_value : int
        Value written into unused columns.
    neighbor_matrix : wp.array, shape (natom, max_neighbors), dtype=wp.int32
        OUTPUT: tail columns filled with ``fill_value``.
    device : str
        Warp device string (e.g. ``"cuda:0"``).
    block_dim : int
        Static tile width for the specialized tail-fill kernel.

    Returns
    -------
    None
        Modifies ``neighbor_matrix`` in-place; see
        :func:`_make_fill_neighbor_matrix_tail_kernel`.

    Notes
    -----
    - This is a low-level warp interface.  Framework bindings should call
      it through :mod:`nvalchemiops.torch.neighbors` /
      :mod:`nvalchemiops.jax.neighbors`.

    See Also
    --------
    _make_fill_neighbor_matrix_tail_kernel : Factory for the fill kernel.
    """
    block_dim = int(block_dim)
    wp.launch_tiled(
        kernel=_get_fill_neighbor_matrix_tail_kernel(block_dim),
        dim=[int(natom)],
        inputs=[
            num_neighbors,
            int(natom),
            int(max_neighbors),
            int(fill_value),
            neighbor_matrix,
        ],
        block_dim=block_dim,
        device=device,
    )
