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


"""Generated Warp kernels and factories for cluster-tile neighbor lists."""

from functools import lru_cache

import warp as wp

from nvalchemiops.neighbors.neighbor_utils import (
    _append_specialization_doc,
    kernel_specialization_name,
    set_fn_doc,
    set_fn_name,
)

__all__ = [
    "TILE_GROUP_SIZE",
    "_compute_morton_kernel",
    "_get_reset_cluster_tile_counts_kernel",
    "_get_build_cluster_tiles_kernel",
    "_get_tile_sort_specialization",
    "_group_buffer",
    "_group_scratch_size",
    "_permute_gather_soa_kernel",
    "_rank_to_group_kernel",
    "_require_f32",
    "get_batch_query_cluster_tile_coo_kernel",
    "get_batch_query_cluster_tile_kernel",
    "get_query_cluster_tile_coo_kernel",
    "get_query_cluster_tile_kernel",
]


TILE_GROUP_SIZE = 32
TILE = wp.constant(TILE_GROUP_SIZE)

# JAX callbacks can force-load this module before entering the Python launcher,
# so Warp compiles it with the module default rather than the launcher's explicit
# block dimension. All tiled kernels in this module use one 32-lane block per
# tile group; compiling them with Warp's default of 256 makes ``wp.untile``
# reject their 32-wide tiles before the callback can run.
wp.set_module_options({"block_dim": TILE_GROUP_SIZE})

_MORTON_PADDING_SENTINEL = 0x40000000


def _pair_output_features(
    *,
    selective: bool = False,
    dual_cutoff: bool = False,
    coo_segmented: bool = False,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
) -> tuple[str, ...]:
    """Return specialization-name tokens for cluster-tile pair outputs."""
    return tuple(
        feature
        for feature in (
            "selective" if selective else "",
            "dual_cutoff" if dual_cutoff else "",
            "coo_segmented" if coo_segmented else "",
            "vectors" if return_vectors else "",
            "distances" if return_distances else "",
            "pair_fn" if pair_fn is not None else "",
        )
        if feature
    )


def _require_f32(wp_dtype: type) -> None:
    """Reject non-float32 dtypes for cluster-tile public launchers."""
    if wp_dtype is not wp.float32:
        raise ValueError(
            f"cluster_tile kernels are float32-only; got {wp_dtype!r}",
        )


def _group_scratch_size(ngroup: int) -> int:
    """Return padded group scratch length for tile-load safety."""
    ngroup_padded = (
        (ngroup + TILE_GROUP_SIZE - 1) // TILE_GROUP_SIZE
    ) * TILE_GROUP_SIZE
    if ngroup_padded == ngroup:
        return ngroup + TILE_GROUP_SIZE
    return ngroup_padded


def _group_buffer(buffer: wp.array | None, size: int, device: str) -> wp.array:
    """Return caller-owned or transient float32 group scratch."""
    if buffer is not None:
        return buffer
    return wp.empty(size, dtype=wp.float32, device=device)


@wp.struct
class _TriclinicQR:
    """Cell-only factors reused by all pairs in a cluster tile."""

    axis_aligned: wp.bool
    q0: wp.vec3f
    q1: wp.vec3f
    q2: wp.vec3f
    r00: wp.float32
    r01: wp.float32
    r02: wp.float32
    r11: wp.float32
    r12: wp.float32
    r22: wp.float32


@wp.func
def _prepare_triclinic_qr(cell: wp.mat33f) -> _TriclinicQR:
    """Prepare QR factors for the cell's lattice basis."""
    qr = _TriclinicQR()
    qr.axis_aligned = (
        cell[0, 1] == 0.0
        and cell[0, 2] == 0.0
        and cell[1, 0] == 0.0
        and cell[1, 2] == 0.0
        and cell[2, 0] == 0.0
        and cell[2, 1] == 0.0
    )
    qr.q0 = wp.vec3f()
    qr.q1 = wp.vec3f()
    qr.q2 = wp.vec3f()
    qr.r00 = 1.0
    qr.r01 = 0.0
    qr.r02 = 0.0
    qr.r11 = 1.0
    qr.r12 = 0.0
    qr.r22 = 1.0
    if qr.axis_aligned:
        return qr

    b0 = wp.vec3f(1.0, 0.0, 0.0) * cell
    b1 = wp.vec3f(0.0, 1.0, 0.0) * cell
    b2 = wp.vec3f(0.0, 0.0, 1.0) * cell
    qr.r00 = wp.sqrt(wp.dot(b0, b0))
    qr.q0 = b0 / qr.r00
    qr.r01 = wp.dot(qr.q0, b1)
    u1 = b1 - qr.q0 * qr.r01
    qr.r11 = wp.sqrt(wp.dot(u1, u1))
    qr.q1 = u1 / qr.r11
    qr.r02 = wp.dot(qr.q0, b2)
    qr.r12 = wp.dot(qr.q1, b2)
    u2 = b2 - qr.q0 * qr.r02 - qr.q1 * qr.r12
    qr.r22 = wp.sqrt(wp.dot(u2, u2))
    qr.q2 = u2 / qr.r22
    return qr


@wp.func
def _triclinic_cutoff_is_certified(
    inv_cell: wp.mat33f, outer_cutoff: wp.float32
) -> wp.bool:
    """Check whether fractional rounding is exact within the query cutoff.

    If ``v`` is an image within ``outer_cutoff``, its fractional coordinates
    have magnitudes bounded by ``outer_cutoff * ||g_i||``, where ``g_i`` are
    the reciprocal lattice vectors. Requiring each bound to remain below one
    half certifies that rounding selects that image and that no second image
    can also be in range. The small margin is conservative for float32 input.
    """
    reciprocal_a = inv_cell * wp.vec3f(1.0, 0.0, 0.0)
    reciprocal_b = inv_cell * wp.vec3f(0.0, 1.0, 0.0)
    reciprocal_c = inv_cell * wp.vec3f(0.0, 0.0, 1.0)
    certified_limit = 0.5 - 1.0e-4
    return (
        outer_cutoff * wp.length(reciprocal_a) < certified_limit
        and outer_cutoff * wp.length(reciprocal_b) < certified_limit
        and outer_cutoff * wp.length(reciprocal_c) < certified_limit
    )


@wp.func
def _prepare_triclinic_qr_if_needed(
    cell: wp.mat33f, fractional_rounding_certified: wp.bool
) -> _TriclinicQR:
    """Prepare QR only when the cutoff certificate cannot select the image."""
    qr = _TriclinicQR()
    if not fractional_rounding_certified:
        qr = _prepare_triclinic_qr(cell)
    return qr


@wp.func
def _triclinic_qr_height_is_certified(
    qr: _TriclinicQR, outer_cutoff: wp.float32
) -> wp.bool:
    """Certify that the Babai candidate is the only possible in-cutoff image.

    For a displacement shorter than ``outer_cutoff``, the q2 projection can
    identify at most one integer c when the cutoff is below half ``r22``.
    With c fixed, the same argument identifies b below half ``r11``; nearest
    plane then chooses the minimizing a. The relative margin protects the
    float32 comparisons. Axis-aligned cells use their direct componentwise
    path and do not have QR heights populated.
    """
    if qr.axis_aligned:
        return wp.bool(False)
    min_height = qr.r11
    if qr.r22 < min_height:
        min_height = qr.r22
    return outer_cutoff < (0.5 - 1.0e-4) * min_height


@wp.func
def _wrap_triclinic_fractional(
    d: wp.vec3f, cell: wp.mat33f, inv_cell: wp.mat33f
) -> tuple[wp.vec3f, wp.vec3i]:
    """Apply componentwise rounding after the cutoff certificate succeeds."""
    fractional = wp.transpose(inv_cell) * d
    s_a = -wp.int32(wp.floor(fractional[0] + 0.5))
    s_b = -wp.int32(wp.floor(fractional[1] + 0.5))
    s_c = -wp.int32(wp.floor(fractional[2] + 0.5))
    shift = wp.vec3i(s_a, s_b, s_c)
    displacement = d + wp.transpose(cell) * wp.vec3f(
        wp.float32(s_a), wp.float32(s_b), wp.float32(s_c)
    )
    return displacement, shift


@wp.func
def _wrap_triclinic_prepared(
    d: wp.vec3f,
    cell: wp.mat33f,
    inv_cell: wp.mat33f,
    qr: _TriclinicQR,
):
    """Search the closest lattice image using prepared QR factors.

    Parameters
    ----------
    d : wp.vec3f
        Cartesian displacement before wrapping.
    cell : wp.mat33f
        Cell matrix.
    inv_cell : wp.mat33f
        Inverse cell matrix.
    qr : _TriclinicQR
        Cell-only QR factors prepared outside the pair loop.

    Returns
    -------
    tuple[wp.vec3f, wp.vec3i]
        Closest-image displacement and the shift added to ``d``.

    Notes
    -----
    Fractional component rounding is exact for pairwise-orthogonal lattice
    vectors. The fast special case below covers axis-aligned orthorhombic
    cells. Otherwise, a Babai candidate seeds a finite sphere, and QR bounds
    enumerate lattice coordinates within it. In exact arithmetic this covers
    the true minimum; float32 arithmetic limits certainty for extremely
    ill-conditioned cells.
    """
    inv_cell_T = wp.transpose(inv_cell)
    cell_T = wp.transpose(cell)

    # Keep the common aligned orthorhombic case to a fractional round and one
    # lattice multiply. The general path below is complete for skewed cells.
    if qr.axis_aligned:
        f = inv_cell_T * d
        s_a = -wp.int32(wp.floor(f[0] + 0.5))
        s_b = -wp.int32(wp.floor(f[1] + 0.5))
        s_c = -wp.int32(wp.floor(f[2] + 0.5))
        shift = wp.vec3i(s_a, s_b, s_c)
        d_new = d + cell_T * wp.vec3f(wp.float32(s_a), wp.float32(s_b), wp.float32(s_c))
        return d_new, shift

    y0 = wp.dot(qr.q0, d)
    y1 = wp.dot(qr.q1, d)
    y2 = wp.dot(qr.q2, d)
    s_c = -wp.int32(wp.floor(y2 / qr.r22 + 0.5))
    s_b = -wp.int32(wp.floor((y1 + qr.r12 * wp.float32(s_c)) / qr.r11 + 0.5))
    s_a = -wp.int32(
        wp.floor(
            (y0 + qr.r01 * wp.float32(s_b) + qr.r02 * wp.float32(s_c)) / qr.r00 + 0.5
        )
    )
    best_shift = wp.vec3i(s_a, s_b, s_c)
    best_displacement = d + cell_T * wp.vec3f(
        wp.float32(s_a), wp.float32(s_b), wp.float32(s_c)
    )
    best_distance_sq = wp.dot(best_displacement, best_displacement)

    center_c = -y2 / qr.r22
    range_slack = 1.0e-6 * wp.max(best_distance_sq, 1.0)
    radius_c = wp.sqrt(best_distance_sq + range_slack) / qr.r22
    c_min = wp.int32(wp.floor(center_c - radius_c))
    c_max = wp.int32(wp.ceil(center_c + radius_c))
    for n_c in range(c_min, c_max + 1):
        residual_c = y2 + qr.r22 * wp.float32(n_c)
        partial_c_sq = residual_c * residual_c
        best_slack = 1.0e-6 * wp.max(best_distance_sq, 1.0)
        if partial_c_sq <= best_distance_sq + best_slack:
            remaining_b_sq = wp.max(
                best_distance_sq + best_slack - partial_c_sq,
                0.0,
            )
            center_b = -(y1 + qr.r12 * wp.float32(n_c)) / qr.r11
            radius_b = wp.sqrt(remaining_b_sq) / qr.r11
            b_min = wp.int32(wp.floor(center_b - radius_b))
            b_max = wp.int32(wp.ceil(center_b + radius_b))
            for n_b in range(b_min, b_max + 1):
                residual_b = y1 + qr.r11 * wp.float32(n_b) + qr.r12 * wp.float32(n_c)
                partial_b_sq = partial_c_sq + residual_b * residual_b
                best_slack = 1.0e-6 * wp.max(best_distance_sq, 1.0)
                if partial_b_sq <= best_distance_sq + best_slack:
                    center_a = (
                        -(y0 + qr.r01 * wp.float32(n_b) + qr.r02 * wp.float32(n_c))
                        / qr.r00
                    )
                    n_a = wp.int32(wp.floor(center_a + 0.5))
                    candidate_shift = wp.vec3i(n_a, n_b, n_c)
                    candidate_displacement = d + cell_T * wp.vec3f(
                        wp.float32(n_a), wp.float32(n_b), wp.float32(n_c)
                    )
                    candidate_distance_sq = wp.dot(
                        candidate_displacement, candidate_displacement
                    )
                    if candidate_distance_sq < best_distance_sq:
                        best_shift = candidate_shift
                        best_displacement = candidate_displacement
                        best_distance_sq = candidate_distance_sq

    return best_displacement, best_shift


@wp.func
def _wrap_triclinic_babai(
    d: wp.vec3f, cell: wp.mat33f, qr: _TriclinicQR
) -> tuple[wp.vec3f, wp.vec3i]:
    """Return the nearest-plane image for a QR-height-certified pair query."""
    y0 = wp.dot(qr.q0, d)
    y1 = wp.dot(qr.q1, d)
    y2 = wp.dot(qr.q2, d)
    s_c = -wp.int32(wp.floor(y2 / qr.r22 + 0.5))
    s_b = -wp.int32(wp.floor((y1 + qr.r12 * wp.float32(s_c)) / qr.r11 + 0.5))
    s_a = -wp.int32(
        wp.floor(
            (y0 + qr.r01 * wp.float32(s_b) + qr.r02 * wp.float32(s_c)) / qr.r00 + 0.5
        )
    )
    shift = wp.vec3i(s_a, s_b, s_c)
    displacement = d + wp.transpose(cell) * wp.vec3f(
        wp.float32(s_a), wp.float32(s_b), wp.float32(s_c)
    )
    return displacement, shift


@wp.func
def _wrap_triclinic_pair(
    d: wp.vec3f,
    cell: wp.mat33f,
    inv_cell: wp.mat33f,
    qr: _TriclinicQR,
    fractional_rounding_certified: wp.bool,
    qr_height_certified: wp.bool,
) -> tuple[wp.vec3f, wp.vec3i]:
    """Use the strongest available pair certificate before complete search."""
    if fractional_rounding_certified:
        return _wrap_triclinic_fractional(d, cell, inv_cell)
    if qr_height_certified:
        return _wrap_triclinic_babai(d, cell, qr)
    return _wrap_triclinic_prepared(d, cell, inv_cell, qr)


@wp.func
def _wrap_triclinic(
    d: wp.vec3f,
    cell: wp.mat33f,
    inv_cell: wp.mat33f,
):
    """Prepare and apply a closest-image search for one displacement."""
    qr = _prepare_triclinic_qr(cell)
    return _wrap_triclinic_prepared(d, cell, inv_cell, qr)


@wp.func
def _direct_coo_pair_is_accepted(
    i_sorted: wp.int32,
    j_sorted: wp.int32,
    i_orig: wp.int32,
    j_orig: wp.int32,
    natom: wp.int32,
    distance_sq: wp.float32,
    cutoff_sq: wp.float32,
) -> wp.bool:
    """Return whether a sorted tile lane defines an accepted pair

    Parameters
    ----------
    i_sorted, j_sorted : wp.int32
        Sorted-layout indices used to keep one orientation of each pair.
    i_orig, j_orig : wp.int32
        Original atom indices used to address the outputs.
    natom : wp.int32
        Number of real atoms; padded indices are rejected.
    distance_sq : wp.float32
        Squared minimum-image distance.
    cutoff_sq : wp.float32
        Squared query cutoff.

    Returns
    -------
    wp.bool
        True when the pair is unique, real, and strictly inside the cutoff.
    """
    return (
        i_sorted < j_sorted
        and i_orig < natom
        and j_orig < natom
        and distance_sq < cutoff_sq
    )


@wp.kernel(enable_backward=False, module="morton_build")
def _compute_morton_kernel(
    positions: wp.array(dtype=wp.vec3f),
    inv_cell: wp.array(dtype=wp.mat33f),
    natom: wp.int32,
    morton_codes: wp.array(dtype=wp.int32),
    sorted_atom_index: wp.array(dtype=wp.int32),
    num_neighbors: wp.array(dtype=wp.int32),
    num_tiles: wp.array(dtype=wp.int32),
) -> None:
    """Compute per-atom Morton codes and initialize tile scratch

    Parameters
    ----------
    positions : wp.array, shape (natom_padded,), dtype=wp.vec3f
        Atomic positions.
    inv_cell : wp.array, shape (1,), dtype=wp.mat33f
        Inverse cell matrix.
    natom : wp.int32
        Number of real atoms before padding.
    morton_codes : wp.array, shape (natom_padded,), dtype=wp.int32
        OUTPUT: Morton code per atom or padding sentinel.
    sorted_atom_index : wp.array, shape (natom_padded,), dtype=wp.int32
        OUTPUT: Initial atom-index permutation.
    num_neighbors : wp.array, shape (natom,), dtype=wp.int32
        OUTPUT: Neighbor counts initialized to zero.
    num_tiles : wp.array, shape (1,), dtype=wp.int32
        OUTPUT: Tile counter initialized to zero.

    Returns
    -------
    None
        This function modifies the input arrays in-place.

    Notes
    -----
    - Thread launch: 1D over the padded atom count.
    - Modifies: morton_codes, sorted_atom_index, num_neighbors, num_tiles.

    See Also
    --------
    _compute_morton : Launch Morton-code generation for cluster-tile construction.
    """
    i = wp.tid()
    if i == 0:
        num_tiles[0] = 0
    if i >= natom:
        morton_codes[i] = wp.int32(_MORTON_PADDING_SENTINEL)
        sorted_atom_index[i] = i
        return
    pos = positions[i]
    inv_cell_mat = inv_cell[0]
    inv_cell_T = wp.transpose(inv_cell_mat)
    f = inv_cell_T * pos
    fx = f[0] - wp.floor(f[0])
    fy = f[1] - wp.floor(f[1])
    fz = f[2] - wp.floor(f[2])
    ix = wp.int32(fx * 1024.0)
    iy = wp.int32(fy * 1024.0)
    iz = wp.int32(fz * 1024.0)
    if ix < 0:
        ix = 0
    if ix > 1023:
        ix = 1023
    if iy < 0:
        iy = 0
    if iy > 1023:
        iy = 1023
    if iz < 0:
        iz = 0
    if iz > 1023:
        iz = 1023
    ix = ix & 0x000003FF
    ix = (ix | (ix << 16)) & 0x030000FF
    ix = (ix | (ix << 8)) & 0x0300F00F
    ix = (ix | (ix << 4)) & 0x030C30C3
    ix = (ix | (ix << 2)) & 0x09249249
    iy = iy & 0x000003FF
    iy = (iy | (iy << 16)) & 0x030000FF
    iy = (iy | (iy << 8)) & 0x0300F00F
    iy = (iy | (iy << 4)) & 0x030C30C3
    iy = (iy | (iy << 2)) & 0x09249249
    iz = iz & 0x000003FF
    iz = (iz | (iz << 16)) & 0x030000FF
    iz = (iz | (iz << 8)) & 0x0300F00F
    iz = (iz | (iz << 4)) & 0x030C30C3
    iz = (iz | (iz << 2)) & 0x09249249
    morton_codes[i] = (iz << 2) | (iy << 1) | ix
    sorted_atom_index[i] = i
    num_neighbors[i] = 0


@wp.kernel(enable_backward=False, module="morton_build")
def _permute_gather_soa_kernel(
    positions: wp.array(dtype=wp.vec3f),
    sorted_atom_index: wp.array(dtype=wp.int32),
    natom: wp.int32,
    sorted_pos_x: wp.array(dtype=wp.float32),
    sorted_pos_y: wp.array(dtype=wp.float32),
    sorted_pos_z: wp.array(dtype=wp.float32),
) -> None:
    """Gather AoS positions into Morton-sorted SoA arrays

    Parameters
    ----------
    positions : wp.array, shape (natom,), dtype=wp.vec3f
        Atomic positions in original ordering.
    sorted_atom_index : wp.array, shape (natom,), dtype=wp.int32
        Source atom index for each sorted slot.
    natom : wp.int32
        Number of real atoms.
    sorted_pos_x : wp.array, shape (natom,), dtype=wp.float32
        OUTPUT: Sorted x coordinates.
    sorted_pos_y : wp.array, shape (natom,), dtype=wp.float32
        OUTPUT: Sorted y coordinates.
    sorted_pos_z : wp.array, shape (natom,), dtype=wp.float32
        OUTPUT: Sorted z coordinates.

    Returns
    -------
    None
        This function modifies the input arrays in-place.

    Notes
    -----
    - Thread launch: 1D over ``natom``.
    - Modifies: sorted_pos_x, sorted_pos_y, sorted_pos_z.

    See Also
    --------
    _permute_gather_soa : Launch sorted-position and tile-coordinate gathering.
    """
    i = wp.tid()
    if i >= natom:
        return
    src = sorted_atom_index[i]
    pos = positions[src]
    sorted_pos_x[i] = pos[0]
    sorted_pos_y[i] = pos[1]
    sorted_pos_z[i] = pos[2]


TILE_SORT_N_1024 = wp.constant(1024)
TILE_SORT_N_2048 = wp.constant(2048)


@wp.kernel(enable_backward=False, module="warp_tile_sort_1024")
def _tile_sort_kernel_1024(
    keys: wp.array(dtype=wp.int32),
    values: wp.array(dtype=wp.int32),
) -> None:
    """Sort 1024 int32 key/value pairs in-place with one tiled block

    Parameters
    ----------
    keys : wp.array, shape (1024,), dtype=wp.int32
        MODIFIED: Sort keys.
    values : wp.array, shape (1024,), dtype=wp.int32
        MODIFIED: Values permuted with ``keys``.

    Returns
    -------
    None
        This function modifies the input arrays in-place.

    Notes
    -----
    - Thread launch: one tiled block with fixed size 1024.
    - Modifies: keys and values.

    See Also
    --------
    _tile_sort_pairs : Launch tile-pair sorting for cluster-tile construction.
    """
    keys_tile = wp.tile_load(keys, shape=TILE_SORT_N_1024, storage="shared")
    values_tile = wp.tile_load(values, shape=TILE_SORT_N_1024, storage="shared")
    wp.tile_sort(keys_tile, values_tile)
    wp.tile_store(keys, keys_tile)
    wp.tile_store(values, values_tile)


@wp.kernel(enable_backward=False, module="warp_tile_sort_2048")
def _tile_sort_kernel_2048(
    keys: wp.array(dtype=wp.int32),
    values: wp.array(dtype=wp.int32),
) -> None:
    """Sort 2048 int32 key/value pairs in-place with one tiled block

    Parameters
    ----------
    keys : wp.array, shape (2048,), dtype=wp.int32
        MODIFIED: Sort keys.
    values : wp.array, shape (2048,), dtype=wp.int32
        MODIFIED: Values permuted with ``keys``.

    Returns
    -------
    None
        This function modifies the input arrays in-place.

    Notes
    -----
    - Thread launch: one tiled block with fixed size 2048.
    - Modifies: keys and values.

    See Also
    --------
    _tile_sort_pairs : Launch tile-pair sorting for cluster-tile construction.
    """
    keys_tile = wp.tile_load(keys, shape=TILE_SORT_N_2048, storage="shared")
    values_tile = wp.tile_load(values, shape=TILE_SORT_N_2048, storage="shared")
    wp.tile_sort(keys_tile, values_tile)
    wp.tile_store(keys, keys_tile)
    wp.tile_store(values, values_tile)


_TILE_SORT_SPECIALIZATIONS = {
    1024: (_tile_sort_kernel_1024, 512),
    2048: (_tile_sort_kernel_2048, 512),
}


def _get_tile_sort_specialization(natom: int) -> tuple[wp.Kernel, int] | None:
    """Return the fixed-size tile-sort specialization for ``natom``."""
    return _TILE_SORT_SPECIALIZATIONS.get(int(natom))


@wp.kernel(enable_backward=False)
def _rank_to_group_kernel(
    sorted_pos_x: wp.array(dtype=wp.float32),
    sorted_pos_y: wp.array(dtype=wp.float32),
    sorted_pos_z: wp.array(dtype=wp.float32),
    group_ctr_x: wp.array(dtype=wp.float32),
    group_ctr_y: wp.array(dtype=wp.float32),
    group_ctr_z: wp.array(dtype=wp.float32),
    group_ext_x: wp.array(dtype=wp.float32),
    group_ext_y: wp.array(dtype=wp.float32),
    group_ext_z: wp.array(dtype=wp.float32),
) -> None:
    """Compute one axis-aligned bounding box for each 32-atom cluster

    Parameters
    ----------
    sorted_pos_x : wp.array, shape (natom_padded,), dtype=wp.float32
        Sorted atom x coordinates.
    sorted_pos_y : wp.array, shape (natom_padded,), dtype=wp.float32
        Sorted atom y coordinates.
    sorted_pos_z : wp.array, shape (natom_padded,), dtype=wp.float32
        Sorted atom z coordinates.
    group_ctr_x : wp.array, shape (ngroup,), dtype=wp.float32
        OUTPUT: Group center x coordinates.
    group_ctr_y : wp.array, shape (ngroup,), dtype=wp.float32
        OUTPUT: Group center y coordinates.
    group_ctr_z : wp.array, shape (ngroup,), dtype=wp.float32
        OUTPUT: Group center z coordinates.
    group_ext_x : wp.array, shape (ngroup,), dtype=wp.float32
        OUTPUT: Group half-extent x coordinates.
    group_ext_y : wp.array, shape (ngroup,), dtype=wp.float32
        OUTPUT: Group half-extent y coordinates.
    group_ext_z : wp.array, shape (ngroup,), dtype=wp.float32
        OUTPUT: Group half-extent z coordinates.

    Returns
    -------
    None
        This function modifies the input arrays in-place.

    Notes
    -----
    - Thread launch: tiled with one block per group and block dimension ``TILE_GROUP_SIZE``.
    - Modifies: group center and extent arrays.

    See Also
    --------
    _tile_sort_pairs : Launch tile-group rank construction after sorting.
    """
    g = wp.tid()
    x_tile = wp.tile_load(sorted_pos_x, shape=TILE, offset=g * TILE)
    y_tile = wp.tile_load(sorted_pos_y, shape=TILE, offset=g * TILE)
    z_tile = wp.tile_load(sorted_pos_z, shape=TILE, offset=g * TILE)

    x_min = wp.tile_extract(wp.tile_min(x_tile), 0)
    x_max = wp.tile_extract(wp.tile_max(x_tile), 0)
    y_min = wp.tile_extract(wp.tile_min(y_tile), 0)
    y_max = wp.tile_extract(wp.tile_max(y_tile), 0)
    z_min = wp.tile_extract(wp.tile_min(z_tile), 0)
    z_max = wp.tile_extract(wp.tile_max(z_tile), 0)

    lane_tile = wp.tile_arange(TILE, dtype=wp.int32)
    lane = wp.untile(lane_tile)
    if lane == 0:
        group_ctr_x[g] = wp.float32(0.5) * (x_max + x_min)
        group_ctr_y[g] = wp.float32(0.5) * (y_max + y_min)
        group_ctr_z[g] = wp.float32(0.5) * (z_max + z_min)
        group_ext_x[g] = wp.float32(0.5) * (x_max - x_min)
        group_ext_y[g] = wp.float32(0.5) * (y_max - y_min)
        group_ext_z[g] = wp.float32(0.5) * (z_max - z_min)


@wp.func
def _bbox_distance_sq(
    d: wp.vec3f,
    rg_ext: wp.vec3f,
    cg_ext: wp.vec3f,
) -> wp.float32:
    """Return squared distance between two Cartesian axis-aligned boxes.

    Parameters
    ----------
    d : wp.vec3f
        Cartesian displacement between the two box centers.
    rg_ext : wp.vec3f
        Half-extents of the row-group bounding box.
    cg_ext : wp.vec3f
        Half-extents of the column-group bounding box.

    Returns
    -------
    wp.float32
        Squared gap distance between the two boxes, or zero when they overlap.
    """
    dx = wp.max(wp.abs(d[0]) - rg_ext[0] - cg_ext[0], 0.0)
    dy = wp.max(wp.abs(d[1]) - rg_ext[1] - cg_ext[1], 0.0)
    dz = wp.max(wp.abs(d[2]) - rg_ext[2] - cg_ext[2], 0.0)
    return dx * dx + dy * dy + dz * dz


@wp.func
def _bbox_valid(
    col_idx: wp.int32,
    row_group: wp.int32,
    col_limit: wp.int32,
    rg_ctr: wp.vec3f,
    rg_ext: wp.vec3f,
    cg_ctr: wp.vec3f,
    cg_ext: wp.vec3f,
    cell: wp.mat33f,
    inv_cell: wp.mat33f,
    qr: _TriclinicQR,
    cutoff_sq: wp.float32,
) -> wp.int32:
    """Return whether a group pair can contain an in-cutoff atom pair

    Parameters
    ----------
    col_idx : wp.int32
        Candidate column group index.
    row_group : wp.int32
        Row group index.
    col_limit : wp.int32
        Exclusive upper bound for valid column groups.
    rg_ctr : wp.vec3f
        Row-group bounding-box center.
    rg_ext : wp.vec3f
        Row-group bounding-box half extents.
    cg_ctr : wp.vec3f
        Column-group bounding-box center.
    cg_ext : wp.vec3f
        Column-group bounding-box half extents.
    cell : wp.mat33f
        Cell matrix.
    inv_cell : wp.mat33f
        Inverse cell matrix.
    cutoff_sq : wp.float32
        Squared cutoff distance.

    Returns
    -------
    wp.int32
        One when the group pair can contain an in-cutoff atom pair, otherwise zero.
    """
    if col_idx < row_group:
        return 0
    if col_idx >= col_limit:
        return 0
    d_ctr = wp.vec3f(
        rg_ctr[0] - cg_ctr[0],
        rg_ctr[1] - cg_ctr[1],
        rg_ctr[2] - cg_ctr[2],
    )
    d_wrapped, _s = _wrap_triclinic_prepared(d_ctr, cell, inv_cell, qr)
    if _bbox_distance_sq(d_wrapped, rg_ext, cg_ext) < cutoff_sq:
        return 1

    # For an axis-aligned orthorhombic cell, fractional rounding selects the
    # componentwise closest center image. That image also minimizes AABB gap
    # distance, so no other periodic image can make this group pair valid.
    if (
        cell[0, 1] == 0.0
        and cell[0, 2] == 0.0
        and cell[1, 0] == 0.0
        and cell[1, 2] == 0.0
        and cell[2, 0] == 0.0
        and cell[2, 1] == 0.0
    ):
        return 0

    # If any translated pair of group boxes is within cutoff, each Cartesian
    # component of the translated center delta is bounded by the summed box
    # half-extents plus cutoff. Applying each reciprocal row gives a finite,
    # complete integer-shift interval. The nearest-center image above is a
    # cheap common-case check; this enumeration covers all other images too.
    cutoff = wp.sqrt(cutoff_sq)
    box_extent = rg_ext + cg_ext
    fractional = d_ctr * inv_cell
    reciprocal_a = inv_cell * wp.vec3f(1.0, 0.0, 0.0)
    reciprocal_b = inv_cell * wp.vec3f(0.0, 1.0, 0.0)
    reciprocal_c = inv_cell * wp.vec3f(0.0, 0.0, 1.0)
    bound_a = (
        wp.abs(reciprocal_a[0]) * box_extent[0]
        + wp.abs(reciprocal_a[1]) * box_extent[1]
        + wp.abs(reciprocal_a[2]) * box_extent[2]
        + cutoff * wp.length(reciprocal_a)
    )
    bound_b = (
        wp.abs(reciprocal_b[0]) * box_extent[0]
        + wp.abs(reciprocal_b[1]) * box_extent[1]
        + wp.abs(reciprocal_b[2]) * box_extent[2]
        + cutoff * wp.length(reciprocal_b)
    )
    bound_c = (
        wp.abs(reciprocal_c[0]) * box_extent[0]
        + wp.abs(reciprocal_c[1]) * box_extent[1]
        + wp.abs(reciprocal_c[2]) * box_extent[2]
        + cutoff * wp.length(reciprocal_c)
    )
    a_min = wp.int32(wp.floor(-fractional[0] - bound_a))
    a_max = wp.int32(wp.ceil(-fractional[0] + bound_a))
    b_min = wp.int32(wp.floor(-fractional[1] - bound_b))
    b_max = wp.int32(wp.ceil(-fractional[1] + bound_b))
    c_min = wp.int32(wp.floor(-fractional[2] - bound_c))
    c_max = wp.int32(wp.ceil(-fractional[2] + bound_c))
    cell_T = wp.transpose(cell)
    for shift_a in range(a_min, a_max + 1):
        for shift_b in range(b_min, b_max + 1):
            for shift_c in range(c_min, c_max + 1):
                if shift_a != _s[0] or shift_b != _s[1] or shift_c != _s[2]:
                    d_image = d_wrapped + cell_T * wp.vec3f(
                        wp.float32(shift_a - _s[0]),
                        wp.float32(shift_b - _s[1]),
                        wp.float32(shift_c - _s[2]),
                    )
                    if _bbox_distance_sq(d_image, rg_ext, cg_ext) < cutoff_sq:
                        return 1
    return 0


@wp.func
def _coo_segment_is_valid(
    start: wp.int32,
    stop: wp.int32,
    max_pairs: wp.int32,
) -> wp.int32:
    """Return whether a segmented COO interval lies within physical storage.

    Parameters
    ----------
    start : wp.int32
        Inclusive start offset of the segment.
    stop : wp.int32
        Exclusive stop offset of the segment.
    max_pairs : wp.int32
        Physical capacity of the COO pair buffer.

    Returns
    -------
    wp.int32
        One when ``0 <= start <= stop <= max_pairs``; otherwise zero.
    """
    if start >= 0 and stop >= start and stop <= max_pairs:
        return wp.int32(1)
    return wp.int32(0)


@lru_cache(maxsize=None)
def _get_reset_cluster_tile_counts_kernel(*, selective: bool) -> wp.Kernel:
    """Build a counter-reset kernel for cluster-tile segmented state."""
    SELECTIVE = wp.constant(bool(selective))

    @wp.kernel(enable_backward=False)
    def _kernel(
        counts: wp.array(dtype=wp.int32),
        rebuild_flags: wp.array(dtype=wp.bool),
    ) -> None:
        """Reset cluster-tile per-system counters

        Parameters
        ----------
        counts : wp.array, shape (num_systems,), dtype=wp.int32
            OUTPUT: Counters to reset before a selective or segmented launch.
        rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool
            Per-system rebuild flags. Read only in selective specializations.

        Returns
        -------
        None
            This function modifies the input arrays in-place.

        Notes
        -----
        - Thread launch: One thread per counter entry.
        - Modifies: ``counts`` entries selected by ``rebuild_flags`` or all entries.
        ``SELECTIVE`` is a static specialization. The reset is a separate
        pre-pass so main kernels never race by zeroing and atomically
        incrementing the same per-system counter.

        See Also
        --------
        _get_reset_cluster_tile_counts_kernel : Return the specialized counter-reset kernel.
        """
        isys = wp.tid()
        if SELECTIVE:
            if rebuild_flags[isys]:
                counts[isys] = 0
        else:
            counts[isys] = 0

    name = kernel_specialization_name(
        "_reset_cluster_tile_counts",
        features=("selective" if selective else "",),
    )
    return set_fn_doc(
        set_fn_name(_kernel, name),
        _append_specialization_doc(
            _kernel.__doc__,
            entries=(
                ("operation", "reset_cluster_tile_counts"),
                ("selective", bool(selective)),
            ),
        ),
    )


@lru_cache(maxsize=None)
def _get_build_cluster_tiles_kernel(
    *, batched: bool, segmented: bool = False, selective: bool = False
) -> wp.Kernel:
    """Build the single/batched group-to-tile enumeration kernel."""
    BATCHED = wp.constant(bool(batched))
    SEGMENTED = wp.constant(bool(segmented))
    SELECTIVE = wp.constant(bool(selective))

    @wp.kernel(enable_backward=False)
    def _kernel(
        group_ctr_x: wp.array(dtype=wp.float32),
        group_ctr_y: wp.array(dtype=wp.float32),
        group_ctr_z: wp.array(dtype=wp.float32),
        group_ext_x: wp.array(dtype=wp.float32),
        group_ext_y: wp.array(dtype=wp.float32),
        group_ext_z: wp.array(dtype=wp.float32),
        group_system: wp.array(dtype=wp.int32),
        group_ptr: wp.array(dtype=wp.int32),
        cell: wp.array(dtype=wp.mat33f),
        inv_cell: wp.array(dtype=wp.mat33f),
        cutoff_sq: wp.float32,
        ngroup: wp.int32,
        num_tiles: wp.array(dtype=wp.int32),
        tile_offsets: wp.array(dtype=wp.int32),
        tile_counts: wp.array(dtype=wp.int32),
        rebuild_flags: wp.array(dtype=wp.bool),
        tile_row_group: wp.array(dtype=wp.int32),
        tile_col_group: wp.array(dtype=wp.int32),
        tile_system: wp.array(dtype=wp.int32),
        max_tiles: wp.int32,
    ) -> None:
        """Build cluster-tile metadata

        Parameters
        ----------
        group_ctr_x, group_ctr_y, group_ctr_z : wp.array, shape (ngroup,), dtype=wp.float32
            Group center coordinates.
        group_ext_x, group_ext_y, group_ext_z : wp.array, shape (ngroup,), dtype=wp.float32
            Group half-extent coordinates.
        group_system : wp.array, shape (ngroup,), dtype=wp.int32
            System index per group for batched mode. Sentinel in single-system mode.
        group_ptr : wp.array, shape (num_systems + 1,), dtype=wp.int32
            Group prefix offsets per system for batched mode.
        cell, inv_cell : wp.array, shape (num_systems,), dtype=wp.mat33f
            Cell and inverse-cell matrices.
        cutoff_sq : wp.float32
            Squared cutoff distance.
        ngroup : wp.int32
            Total number of groups.
        num_tiles : wp.array, shape (1,), dtype=wp.int32
            OUTPUT: Compact tile count for non-segmented launches.
        tile_offsets : wp.array, shape (num_systems + 1,), dtype=wp.int32
            Caller-owned fixed output offsets for segmented batched launches.
        tile_counts : wp.array, shape (num_systems,), dtype=wp.int32
            OUTPUT: Per-system tile counts for segmented batched launches.
        rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool
            Per-system selective rebuild flags. Sentinel when inactive.
        tile_row_group, tile_col_group : wp.array, shape (max_tiles,), dtype=wp.int32
            OUTPUT: Row and column group index for each emitted tile.
        tile_system : wp.array, shape (max_tiles,), dtype=wp.int32
            OUTPUT: System index for each emitted tile in batched mode.
        max_tiles : wp.int32
            Compact tile capacity for non-segmented launches.

        Returns
        -------
        None
            This function modifies the input arrays in-place.

        Notes
        -----
        - Thread launch: Tiled launch with one block per row group and block dimension ``TILE_GROUP_SIZE``.
        - Modifies: ``num_tiles`` or ``tile_counts`` plus ``tile_row_group``, ``tile_col_group``, and ``tile_system``.
        ``BATCHED``, ``SEGMENTED``, and ``SELECTIVE`` are static specializations. Segmented
        batched mode writes each system into ``[tile_offsets[isys],
        tile_offsets[isys + 1])`` using ``tile_counts[isys]`` as the local
        atomic counter.

        See Also
        --------
        build_cluster_tile_list : Launch cluster-tile list construction.
        """
        row_group = wp.tid()
        lane_tile = wp.tile_arange(TILE, dtype=wp.int32)
        lane = wp.untile(lane_tile)

        system_idx = wp.int32(0)
        col_limit = ngroup
        if BATCHED:
            system_idx = group_system[row_group]
            col_limit = group_ptr[system_idx + 1]

        if SELECTIVE:
            if not rebuild_flags[system_idx]:
                return

        cell_mat = cell[system_idx]
        inv_cell_mat = inv_cell[system_idx]
        triclinic_qr = _prepare_triclinic_qr(cell_mat)

        rg_ctr = wp.vec3f(
            group_ctr_x[row_group],
            group_ctr_y[row_group],
            group_ctr_z[row_group],
        )
        rg_ext = wp.vec3f(
            group_ext_x[row_group],
            group_ext_y[row_group],
            group_ext_z[row_group],
        )

        col_start_aligned = (row_group // TILE) * TILE
        for col_start in range(col_start_aligned, col_limit, TILE):
            cg_ctr_x_tile = wp.tile_load(group_ctr_x, shape=TILE, offset=col_start)
            cg_ctr_y_tile = wp.tile_load(group_ctr_y, shape=TILE, offset=col_start)
            cg_ctr_z_tile = wp.tile_load(group_ctr_z, shape=TILE, offset=col_start)
            cg_ext_x_tile = wp.tile_load(group_ext_x, shape=TILE, offset=col_start)
            cg_ext_y_tile = wp.tile_load(group_ext_y, shape=TILE, offset=col_start)
            cg_ext_z_tile = wp.tile_load(group_ext_z, shape=TILE, offset=col_start)

            cg = col_start + lane
            cg_ctr = wp.vec3f(
                wp.untile(cg_ctr_x_tile),
                wp.untile(cg_ctr_y_tile),
                wp.untile(cg_ctr_z_tile),
            )
            cg_ext = wp.vec3f(
                wp.untile(cg_ext_x_tile),
                wp.untile(cg_ext_y_tile),
                wp.untile(cg_ext_z_tile),
            )

            valid = _bbox_valid(
                cg,
                row_group,
                col_limit,
                rg_ctr,
                rg_ext,
                cg_ctr,
                cg_ext,
                cell_mat,
                inv_cell_mat,
                triclinic_qr,
                cutoff_sq,
            )

            valid_tile = wp.tile(valid)
            scan_tile = wp.tile_scan_inclusive(valid_tile)
            tile_total = wp.tile_extract(scan_tile, TILE - 1)

            if tile_total > 0:
                base = wp.int32(0)
                capacity = max_tiles
                offset = wp.int32(0)
                if SEGMENTED:
                    capacity = tile_offsets[system_idx + 1] - tile_offsets[system_idx]
                    offset = tile_offsets[system_idx]
                    if lane == 0:
                        base = wp.atomic_add(tile_counts, system_idx, tile_total)
                else:
                    if lane == 0:
                        base = wp.atomic_add(num_tiles, 0, tile_total)
                base_tile = wp.tile_from_thread(
                    shape=TILE,
                    value=base,
                    thread_idx=0,
                    storage="shared",
                )
                base_b = wp.untile(base_tile)
                scan_v = wp.untile(scan_tile)
                local_slot = base_b + scan_v - 1
                slot = offset + local_slot

                if valid == 1 and local_slot < capacity:
                    tile_row_group[slot] = row_group
                    tile_col_group[slot] = cg
                    if BATCHED:
                        tile_system[slot] = system_idx

    name = kernel_specialization_name(
        "_build_cluster_tiles",
        wp_dtype=wp.float32,
        features=(
            "batched" if batched else "",
            "segmented" if segmented else "",
            "selective" if selective else "",
        ),
    )
    return set_fn_doc(
        set_fn_name(_kernel, name),
        _append_specialization_doc(
            _kernel.__doc__,
            dtype=wp.float32,
            entries=(
                ("operation", "build_cluster_tiles"),
                ("batched", bool(batched)),
                ("segmented", bool(segmented)),
                ("selective", bool(selective)),
            ),
        ),
    )


@lru_cache(maxsize=None)
def _make_query_cluster_tile_kernel(
    *,
    batched: bool,
    tile_segmented: bool,
    selective: bool,
    dual_cutoff: bool,
    return_vectors: bool,
    return_distances: bool,
    pair_fn: wp.Function | None,
) -> wp.Kernel:
    """Build the tile-to-matrix kernel for one batching/feature combination."""
    BATCHED = wp.constant(bool(batched))
    TILE_SEGMENTED = wp.constant(bool(tile_segmented))
    SELECTIVE = wp.constant(bool(selective))
    DUAL_CUTOFF = wp.constant(bool(dual_cutoff))
    RETURN_VECTORS = wp.constant(bool(return_vectors))
    RETURN_DISTANCES = wp.constant(bool(return_distances))
    HAS_PAIR_FN = wp.constant(pair_fn is not None)

    @wp.kernel(enable_backward=False, module="unique")
    def _kernel(
        sorted_pos_x: wp.array(dtype=wp.float32),
        sorted_pos_y: wp.array(dtype=wp.float32),
        sorted_pos_z: wp.array(dtype=wp.float32),
        sorted_atom_index: wp.array(dtype=wp.int32),
        cell: wp.array(dtype=wp.mat33f),
        inv_cell: wp.array(dtype=wp.mat33f),
        cutoff_sq: wp.float32,
        cutoff_sq2: wp.float32,
        natom: wp.int32,
        max_neighbors: wp.int32,
        max_neighbors2: wp.int32,
        tile_row_group: wp.array(dtype=wp.int32),
        tile_col_group: wp.array(dtype=wp.int32),
        tile_system: wp.array(dtype=wp.int32),
        num_tiles: wp.array(dtype=wp.int32),
        tile_offsets: wp.array(dtype=wp.int32),
        tile_counts: wp.array(dtype=wp.int32),
        rebuild_flags: wp.array(dtype=wp.bool),
        neighbor_matrix: wp.array2d(dtype=wp.int32),
        num_neighbors: wp.array(dtype=wp.int32),
        neighbor_matrix_shifts: wp.array3d(dtype=wp.int32),
        neighbor_matrix2: wp.array2d(dtype=wp.int32),
        num_neighbors2: wp.array(dtype=wp.int32),
        neighbor_matrix_shifts2: wp.array3d(dtype=wp.int32),
        neighbor_vectors: wp.array2d(dtype=wp.vec3f),
        neighbor_distances: wp.array2d(dtype=wp.float32),
        pair_params: wp.array2d(dtype=wp.float32),
        pair_energies: wp.array2d(dtype=wp.float32),
        pair_forces: wp.array2d(dtype=wp.vec3f),
    ) -> None:
        """Fill cluster-tile neighbor-matrix rows

        Parameters
        ----------
        sorted_pos_x, sorted_pos_y, sorted_pos_z : wp.array, shape (natom_padded,), dtype=wp.float32
            Sorted atom coordinates.
        sorted_atom_index : wp.array, shape (natom_padded,), dtype=wp.int32
            Original atom index for each sorted slot.
        cell, inv_cell : wp.array, shape (num_systems,), dtype=wp.mat33f
            Cell and inverse-cell matrices.
        cutoff_sq, cutoff_sq2 : wp.float32
            Squared primary and secondary cutoffs. ``cutoff_sq2`` is read only
            in dual-cutoff specializations.
        natom : wp.int32
            Number of real atoms before padding.
        max_neighbors, max_neighbors2 : wp.int32
            Maximum row capacities for the primary and secondary matrices.
        tile_row_group, tile_col_group, tile_system : wp.array, shape (num_tiles,), dtype=wp.int32
            Cluster-tile records from the build launcher.
        num_tiles : wp.array, shape (1,), dtype=wp.int32
            Compact tile count for non-segmented tile lists.
        tile_offsets : wp.array, shape (num_systems + 1,), dtype=wp.int32
            Fixed per-system tile offsets for segmented tile lists.
        tile_counts : wp.array, shape (num_systems,), dtype=wp.int32
            Per-system tile counts for segmented tile lists.
        rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool
            Per-system selective rebuild flags. Sentinel when inactive.
        neighbor_matrix, neighbor_matrix2 : wp.array, shape (natom, max_neighbors*), dtype=wp.int32
            OUTPUT: Primary and optional secondary neighbor atom indices.
        num_neighbors, num_neighbors2 : wp.array, shape (natom,), dtype=wp.int32
            OUTPUT: Primary and optional secondary row counts, updated atomically.
        neighbor_matrix_shifts, neighbor_matrix_shifts2 : wp.array, shape (natom, max_neighbors*, 3), dtype=wp.int32
            OUTPUT: Periodic shift vectors for each emitted neighbor.
        neighbor_vectors : wp.array, shape (natom, max_neighbors), dtype=wp.vec3f
            OUTPUT: Optional primary displacement vectors. Sentinel when disabled.
        neighbor_distances : wp.array, shape (natom, max_neighbors), dtype=wp.float32
            OUTPUT: Optional primary pair distances. Sentinel when disabled.
        pair_params : wp.array, shape (natom, K), dtype=wp.float32
            Pair-function parameters. Sentinel when no pair function is active.
        pair_energies : wp.array, shape (natom, max_neighbors), dtype=wp.float32
            OUTPUT: Optional primary pair-function energies. Sentinel when disabled.
        pair_forces : wp.array, shape (natom, max_neighbors), dtype=wp.vec3f
            OUTPUT: Optional primary pair-function forces. Sentinel when disabled.

        Returns
        -------
        None
            This function modifies the input arrays in-place.

        Notes
        -----
        - Thread launch: Tiled launch with one block per allocated tile slot and block dimension ``TILE_GROUP_SIZE``.
        - Modifies: neighbor matrices, neighbor counts, shift matrices, and enabled pair-output buffers.
        Batched, segmented-tile, selective, dual-cutoff, and pair-output modes
        are static specializations. Selective kernels skip unflagged systems;
        their row counters must be reset by a separate pre-pass.

        See Also
        --------
        query_cluster_tile : Launch cluster-tile neighbor search with matrix-style outputs.
        """
        tile = wp.tid()
        system_idx = wp.int32(0)

        if TILE_SEGMENTED:
            if BATCHED:
                system_idx = tile_system[tile]
            local_tile = tile - tile_offsets[system_idx]
            # Bound to the per-system region capacity (mirrors the build's
            # ``local_slot < capacity`` write clamp).  ``tile_counts`` is
            # incremented unclamped, so on a tile-capacity overflow it can
            # exceed ``capacity``; without this clamp an over-counted system
            # plus zeroed ``tile_system`` gap slots can misattribute tiles
            # from a later region.
            capacity = tile_offsets[system_idx + 1] - tile_offsets[system_idx]
            if (
                local_tile < 0
                or local_tile >= tile_counts[system_idx]
                or local_tile >= capacity
            ):
                return
        else:
            if tile >= num_tiles[0]:
                return
            if BATCHED:
                system_idx = tile_system[tile]

        if SELECTIVE:
            if not rebuild_flags[system_idx]:
                return

        lane_tile = wp.tile_arange(TILE, dtype=wp.int32)
        lane = wp.untile(lane_tile)

        row_group = tile_row_group[tile]
        col_group = tile_col_group[tile]

        cell_mat = cell[system_idx]
        inv_cell_mat = inv_cell[system_idx]
        outer_cutoff_sq = cutoff_sq
        if DUAL_CUTOFF and cutoff_sq2 > outer_cutoff_sq:
            outer_cutoff_sq = cutoff_sq2
        fractional_rounding_certified = _triclinic_cutoff_is_certified(
            inv_cell_mat, wp.sqrt(outer_cutoff_sq)
        )
        triclinic_qr = _prepare_triclinic_qr_if_needed(
            cell_mat, fractional_rounding_certified
        )
        qr_height_certified = wp.bool(False)
        if not fractional_rounding_certified:
            qr_height_certified = _triclinic_qr_height_is_certified(
                triclinic_qr, wp.sqrt(outer_cutoff_sq)
            )

        j_sorted = col_group * TILE + lane
        j_orig_t = sorted_atom_index[j_sorted]
        pj_x = sorted_pos_x[j_sorted]
        pj_y = sorted_pos_y[j_sorted]
        pj_z = sorted_pos_z[j_sorted]

        pi_x_tile = wp.tile_load(sorted_pos_x, shape=TILE, offset=row_group * TILE)
        pi_y_tile = wp.tile_load(sorted_pos_y, shape=TILE, offset=row_group * TILE)
        pi_z_tile = wp.tile_load(sorted_pos_z, shape=TILE, offset=row_group * TILE)
        i_orig_tile = wp.tile_load(
            sorted_atom_index, shape=TILE, offset=row_group * TILE
        )

        for i_local in range(TILE_GROUP_SIZE):
            i_sorted = row_group * TILE + i_local
            i_orig = wp.tile_extract(i_orig_tile, i_local)
            pi_x = wp.tile_extract(pi_x_tile, i_local)
            pi_y = wp.tile_extract(pi_y_tile, i_local)
            pi_z = wp.tile_extract(pi_z_tile, i_local)

            d = wp.vec3f(pj_x - pi_x, pj_y - pi_y, pj_z - pi_z)
            d_wrapped, s_shift = _wrap_triclinic_pair(
                d,
                cell_mat,
                inv_cell_mat,
                triclinic_qr,
                fractional_rounding_certified,
                qr_height_certified,
            )
            dist_sq = (
                d_wrapped[0] * d_wrapped[0]
                + d_wrapped[1] * d_wrapped[1]
                + d_wrapped[2] * d_wrapped[2]
            )

            in_real_atoms = wp.bool(False)
            if BATCHED:
                if i_sorted < j_sorted and i_orig < natom and j_orig_t < natom:
                    in_real_atoms = True
            else:
                if i_sorted < j_sorted and i_sorted < natom and j_sorted < natom:
                    in_real_atoms = True

            valid = wp.int32(0)
            if in_real_atoms and dist_sq < cutoff_sq:
                valid = wp.int32(1)

            valid_tile = wp.tile(valid)
            scan_tile = wp.tile_scan_inclusive(valid_tile)
            row_count = wp.tile_extract(scan_tile, TILE - 1)

            if row_count > 0:
                base = wp.int32(0)
                if lane == 0:
                    base = wp.atomic_add(num_neighbors, i_orig, row_count)
                base_tile = wp.tile_from_thread(
                    shape=TILE,
                    value=base,
                    thread_idx=0,
                    storage="shared",
                )
                base_b = wp.untile(base_tile)
                scan_v = wp.untile(scan_tile)
                write_col = base_b + scan_v - 1

                if valid == 1 and write_col < max_neighbors:
                    neighbor_matrix[i_orig, write_col] = j_orig_t
                    neighbor_matrix_shifts[i_orig, write_col, 0] = s_shift[0]
                    neighbor_matrix_shifts[i_orig, write_col, 1] = s_shift[1]
                    neighbor_matrix_shifts[i_orig, write_col, 2] = s_shift[2]
                    if RETURN_VECTORS:
                        neighbor_vectors[i_orig, write_col] = d_wrapped
                    if RETURN_DISTANCES or HAS_PAIR_FN:
                        distance = wp.sqrt(dist_sq)
                        if RETURN_DISTANCES:
                            neighbor_distances[i_orig, write_col] = distance
                        if HAS_PAIR_FN:
                            pair_energy, pair_force = pair_fn(
                                d_wrapped,
                                distance,
                                pair_params,
                                i_orig,
                                j_orig_t,
                            )
                            pair_energies[i_orig, write_col] = pair_energy
                            pair_forces[i_orig, write_col] = pair_force

            # Full-fill: emit the reverse (j, i) pair into atom j's row with
            # negated shift/vector and the (j, i) pair-function result.  Each
            # unordered pair is visited once (i_sorted < j_sorted), so forward
            # plus reverse yields each ordered pair exactly once -- a symmetric
            # full matrix matching cell_list / naive (half_fill=False).
            if valid == 1:
                pos_j = wp.atomic_add(num_neighbors, j_orig_t, 1)
                if pos_j < max_neighbors:
                    neighbor_matrix[j_orig_t, pos_j] = i_orig
                    neighbor_matrix_shifts[j_orig_t, pos_j, 0] = -s_shift[0]
                    neighbor_matrix_shifts[j_orig_t, pos_j, 1] = -s_shift[1]
                    neighbor_matrix_shifts[j_orig_t, pos_j, 2] = -s_shift[2]
                    if RETURN_VECTORS:
                        neighbor_vectors[j_orig_t, pos_j] = -d_wrapped
                    if RETURN_DISTANCES or HAS_PAIR_FN:
                        distance_r = wp.sqrt(dist_sq)
                        if RETURN_DISTANCES:
                            neighbor_distances[j_orig_t, pos_j] = distance_r
                        if HAS_PAIR_FN:
                            pair_energy_r, pair_force_r = pair_fn(
                                -d_wrapped,
                                distance_r,
                                pair_params,
                                j_orig_t,
                                i_orig,
                            )
                            pair_energies[j_orig_t, pos_j] = pair_energy_r
                            pair_forces[j_orig_t, pos_j] = pair_force_r

            if DUAL_CUTOFF:
                valid2 = wp.int32(0)
                if in_real_atoms and dist_sq < cutoff_sq2:
                    valid2 = wp.int32(1)
                valid2_tile = wp.tile(valid2)
                scan2_tile = wp.tile_scan_inclusive(valid2_tile)
                row_count2 = wp.tile_extract(scan2_tile, TILE - 1)

                if row_count2 > 0:
                    base2 = wp.int32(0)
                    if lane == 0:
                        base2 = wp.atomic_add(num_neighbors2, i_orig, row_count2)
                    base2_tile = wp.tile_from_thread(
                        shape=TILE,
                        value=base2,
                        thread_idx=0,
                        storage="shared",
                    )
                    base2_b = wp.untile(base2_tile)
                    scan2_v = wp.untile(scan2_tile)
                    write_col2 = base2_b + scan2_v - 1

                    if valid2 == 1 and write_col2 < max_neighbors2:
                        neighbor_matrix2[i_orig, write_col2] = j_orig_t
                        neighbor_matrix_shifts2[i_orig, write_col2, 0] = s_shift[0]
                        neighbor_matrix_shifts2[i_orig, write_col2, 1] = s_shift[1]
                        neighbor_matrix_shifts2[i_orig, write_col2, 2] = s_shift[2]

                # Full-fill reverse for the secondary (cutoff2) matrix.
                if valid2 == 1:
                    pos_j2 = wp.atomic_add(num_neighbors2, j_orig_t, 1)
                    if pos_j2 < max_neighbors2:
                        neighbor_matrix2[j_orig_t, pos_j2] = i_orig
                        neighbor_matrix_shifts2[j_orig_t, pos_j2, 0] = -s_shift[0]
                        neighbor_matrix_shifts2[j_orig_t, pos_j2, 1] = -s_shift[1]
                        neighbor_matrix_shifts2[j_orig_t, pos_j2, 2] = -s_shift[2]

    name = kernel_specialization_name(
        "_query_cluster_tile",
        wp_dtype=wp.float32,
        features=(
            "batched" if batched else "",
            "tile_segmented" if tile_segmented else "",
            *_pair_output_features(
                selective=selective,
                dual_cutoff=dual_cutoff,
                return_vectors=return_vectors,
                return_distances=return_distances,
                pair_fn=pair_fn,
            ),
        ),
    )
    return set_fn_doc(
        set_fn_name(_kernel, name),
        _append_specialization_doc(
            _kernel.__doc__,
            dtype=wp.float32,
            entries=(
                ("operation", "query_cluster_tile"),
                ("output", "matrix"),
                ("batched", bool(batched)),
                ("tile_segmented", bool(tile_segmented)),
                ("selective", bool(selective)),
                ("dual_cutoff", bool(dual_cutoff)),
                ("return_vectors", bool(return_vectors)),
                ("return_distances", bool(return_distances)),
                ("pair_fn", pair_fn is not None),
            ),
        ),
    )


def get_query_cluster_tile_kernel(
    *,
    tile_segmented: bool = False,
    selective: bool = False,
    dual_cutoff: bool = False,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
) -> wp.Kernel:
    """Return a cached single-system tile-to-matrix kernel.

    The cache key stores the Warp function object directly, so module-scope
    ``@wp.func`` singletons reuse the same compiled kernel across calls.
    """
    return _make_query_cluster_tile_kernel(
        batched=False,
        tile_segmented=bool(tile_segmented),
        selective=bool(selective),
        dual_cutoff=bool(dual_cutoff),
        return_vectors=bool(return_vectors),
        return_distances=bool(return_distances),
        pair_fn=pair_fn,
    )


def get_batch_query_cluster_tile_kernel(
    *,
    tile_segmented: bool = False,
    selective: bool = False,
    dual_cutoff: bool = False,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
) -> wp.Kernel:
    """Return a cached batched tile-to-matrix kernel."""
    return _make_query_cluster_tile_kernel(
        batched=True,
        tile_segmented=bool(tile_segmented),
        selective=bool(selective),
        dual_cutoff=bool(dual_cutoff),
        return_vectors=bool(return_vectors),
        return_distances=bool(return_distances),
        pair_fn=pair_fn,
    )


@lru_cache(maxsize=None)
def _get_query_cluster_tile_direct_csr_count_kernel(*, batched: bool) -> wp.Kernel:
    """Build the count pass for compact, source-owned exact COO."""
    BATCHED = wp.constant(bool(batched))

    @wp.kernel(enable_backward=False, module="unique")
    def _kernel(
        sorted_pos_x: wp.array(dtype=wp.float32),
        sorted_pos_y: wp.array(dtype=wp.float32),
        sorted_pos_z: wp.array(dtype=wp.float32),
        sorted_atom_index: wp.array(dtype=wp.int32),
        cell: wp.array(dtype=wp.mat33f),
        inv_cell: wp.array(dtype=wp.mat33f),
        cutoff_sq: wp.float32,
        natom: wp.int32,
        num_tiles: wp.array(dtype=wp.int32),
        tile_row_group: wp.array(dtype=wp.int32),
        tile_col_group: wp.array(dtype=wp.int32),
        tile_system: wp.array(dtype=wp.int32),
        row_counts: wp.array(dtype=wp.int32),
    ) -> None:
        """Count accepted direct-CSR neighbors for each source atom

        Parameters
        ----------
        sorted_pos_x, sorted_pos_y, sorted_pos_z : wp.array, shape (natom_padded,), dtype=wp.float32
            Morton-sorted position components in padded group layout.
        sorted_atom_index : wp.array, shape (natom_padded,), dtype=wp.int32
            Original atom index for each sorted slot.
        cell, inv_cell : wp.array, shape (num_systems,), dtype=wp.mat33f
            Per-system cell matrices and inverse matrices.
        cutoff_sq : wp.float32
            Squared query cutoff.
        natom : wp.int32
            Total number of real atoms.
        num_tiles : wp.array, shape (1,), dtype=wp.int32
            Number of active tile pairs.
        tile_row_group, tile_col_group : wp.array, shape (tile_capacity,), dtype=wp.int32
            Row and column group for each tile pair.
        tile_system : wp.array, shape (tile_capacity,), dtype=wp.int32
            System index for each tile pair. Ignored by the single-system
            specialization.
        row_counts : wp.array, shape (natom,), dtype=wp.int32
            OUTPUT: Accepted neighbor count for each source atom.

        Returns
        -------
        None
            This function updates ``row_counts`` in-place.

        Notes
        -----
        - Thread launch: One tiled thread block per launched tile slot; slots at
          or beyond ``num_tiles[0]`` return without writing.
        - Modifies: ``row_counts`` through one atomic increment per occupied
          forward row and one per occupied reverse lane in each tile.

        See Also
        --------
        _get_query_cluster_tile_direct_csr_fill_kernel : Build the matching CSR fill pass.
        """
        tid = wp.tid()
        if tid >= num_tiles[0]:
            return
        system_idx = wp.int32(0)
        if BATCHED:
            system_idx = tile_system[tid]
        lane_tile = wp.tile_arange(TILE, dtype=wp.int32)
        lane = wp.untile(lane_tile)
        row_group = tile_row_group[tid]
        col_group = tile_col_group[tid]
        j_sorted = col_group * TILE + lane
        j_orig = sorted_atom_index[j_sorted]
        pj_x = sorted_pos_x[j_sorted]
        pj_y = sorted_pos_y[j_sorted]
        pj_z = sorted_pos_z[j_sorted]
        pi_x_tile = wp.tile_load(sorted_pos_x, shape=TILE, offset=row_group * TILE)
        pi_y_tile = wp.tile_load(sorted_pos_y, shape=TILE, offset=row_group * TILE)
        pi_z_tile = wp.tile_load(sorted_pos_z, shape=TILE, offset=row_group * TILE)
        i_orig_tile = wp.tile_load(
            sorted_atom_index, shape=TILE, offset=row_group * TILE
        )
        cell_mat = cell[system_idx]
        inv_cell_mat = inv_cell[system_idx]
        fractional_rounding_certified = _triclinic_cutoff_is_certified(
            inv_cell_mat, wp.sqrt(cutoff_sq)
        )
        triclinic_qr = _prepare_triclinic_qr_if_needed(
            cell_mat, fractional_rounding_certified
        )
        qr_height_certified = wp.bool(False)
        if not fractional_rounding_certified:
            qr_height_certified = _triclinic_qr_height_is_certified(
                triclinic_qr, wp.sqrt(cutoff_sq)
            )
        reverse_count = wp.int32(0)
        for i_local in range(TILE_GROUP_SIZE):
            i_sorted = row_group * TILE + i_local
            i_orig = wp.tile_extract(i_orig_tile, i_local)
            d = wp.vec3f(
                pj_x - wp.tile_extract(pi_x_tile, i_local),
                pj_y - wp.tile_extract(pi_y_tile, i_local),
                pj_z - wp.tile_extract(pi_z_tile, i_local),
            )
            wrapped, _ = _wrap_triclinic_pair(
                d,
                cell_mat,
                inv_cell_mat,
                triclinic_qr,
                fractional_rounding_certified,
                qr_height_certified,
            )
            distance_sq = wp.dot(wrapped, wrapped)
            valid = wp.int32(0)
            if _direct_coo_pair_is_accepted(
                i_sorted, j_sorted, i_orig, j_orig, natom, distance_sq, cutoff_sq
            ):
                valid = wp.int32(1)
                reverse_count += wp.int32(1)

            valid_tile = wp.tile(valid)
            scan_tile = wp.tile_scan_inclusive(valid_tile)
            forward_count = wp.tile_extract(scan_tile, TILE - 1)
            if lane == 0 and forward_count > 0:
                wp.atomic_add(row_counts, i_orig, forward_count)

        if reverse_count > 0:
            wp.atomic_add(row_counts, j_orig, reverse_count)

    return _kernel


@lru_cache(maxsize=None)
def _get_query_cluster_tile_direct_csr_fill_kernel(
    *,
    batched: bool,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
) -> wp.Kernel:
    """Build the fill pass for compact, source-owned exact COO."""
    BATCHED = wp.constant(bool(batched))
    RETURN_VECTORS = wp.constant(bool(return_vectors))
    RETURN_DISTANCES = wp.constant(bool(return_distances))
    HAS_PAIR_FN = wp.constant(pair_fn is not None)

    @wp.kernel(enable_backward=False, module="unique")
    def _kernel(
        sorted_pos_x: wp.array(dtype=wp.float32),
        sorted_pos_y: wp.array(dtype=wp.float32),
        sorted_pos_z: wp.array(dtype=wp.float32),
        sorted_atom_index: wp.array(dtype=wp.int32),
        cell: wp.array(dtype=wp.mat33f),
        inv_cell: wp.array(dtype=wp.mat33f),
        cutoff_sq: wp.float32,
        natom: wp.int32,
        physical_capacity: wp.int32,
        num_tiles: wp.array(dtype=wp.int32),
        tile_row_group: wp.array(dtype=wp.int32),
        tile_col_group: wp.array(dtype=wp.int32),
        tile_system: wp.array(dtype=wp.int32),
        cursors: wp.array(dtype=wp.int32),
        coo_list: wp.array2d(dtype=wp.int32),
        coo_shifts: wp.array2d(dtype=wp.int32),
        neighbor_vectors: wp.array(dtype=wp.vec3f),
        neighbor_distances: wp.array(dtype=wp.float32),
        pair_params: wp.array2d(dtype=wp.float32),
        pair_energies: wp.array(dtype=wp.float32),
        pair_forces: wp.array(dtype=wp.vec3f),
    ) -> None:
        """Fill source-owned CSR rows and aligned pair outputs

        Parameters
        ----------
        sorted_pos_x, sorted_pos_y, sorted_pos_z : wp.array, shape (natom_padded,), dtype=wp.float32
            Morton-sorted position components in padded group layout.
        sorted_atom_index : wp.array, shape (natom_padded,), dtype=wp.int32
            Original atom index for each sorted slot.
        cell, inv_cell : wp.array, shape (num_systems,), dtype=wp.mat33f
            Per-system cell matrices and inverse matrices.
        cutoff_sq : wp.float32
            Squared query cutoff.
        natom : wp.int32
            Total number of real atoms.
        physical_capacity : wp.int32
            Number of writable pair slots in each output buffer.
        num_tiles : wp.array, shape (1,), dtype=wp.int32
            Number of active tile pairs.
        tile_row_group, tile_col_group : wp.array, shape (tile_capacity,), dtype=wp.int32
            Row and column group for each tile pair.
        tile_system : wp.array, shape (tile_capacity,), dtype=wp.int32
            System index for each tile pair. Ignored by the single-system
            specialization.
        cursors : wp.array, shape (natom,), dtype=wp.int32
            OUTPUT: Per-source insertion cursors initialized from row pointers.
        coo_list : wp.array, shape (physical_capacity, 2), dtype=wp.int32
            OUTPUT: Directed source-target pairs in CSR row ownership.
        coo_shifts : wp.array, shape (physical_capacity, 3), dtype=wp.int32
            OUTPUT: Periodic shifts aligned with ``coo_list``.
        neighbor_vectors : wp.array, shape (physical_capacity,), dtype=wp.vec3f
            OUTPUT: Optional displacement vectors aligned with pairs. Sentinel
            when disabled.
        neighbor_distances : wp.array, shape (physical_capacity,), dtype=wp.float32
            OUTPUT: Optional distances aligned with pairs. Sentinel when
            disabled.
        pair_params : wp.array, shape (natom, K), dtype=wp.float32
            Pair-function parameters. Sentinel when no pair function is active.
        pair_energies : wp.array, shape (physical_capacity,), dtype=wp.float32
            OUTPUT: Optional pair-function energies aligned with pairs.
            Sentinel when disabled.
        pair_forces : wp.array, shape (physical_capacity,), dtype=wp.vec3f
            OUTPUT: Optional pair-function forces aligned with pairs. Sentinel
            when disabled.

        Returns
        -------
        None
            This function updates insertion cursors and enabled output buffers
            in-place.

        Notes
        -----
        - Thread launch: One tiled thread block per launched tile slot; slots at
          or beyond ``num_tiles[0]`` return without writing.
        - Modifies: ``cursors``, topology buffers, and enabled pair-output buffers.
          A first tile traversal reserves one reverse-row range per occupied
          lane. A second traversal reserves one forward-row range per occupied
          row and writes through block-local cursors.

        See Also
        --------
        _get_query_cluster_tile_direct_csr_count_kernel : Build the preceding CSR count pass.
        """
        tid = wp.tid()
        if tid >= num_tiles[0]:
            return
        system_idx = wp.int32(0)
        if BATCHED:
            system_idx = tile_system[tid]
        lane_tile = wp.tile_arange(TILE, dtype=wp.int32)
        lane = wp.untile(lane_tile)
        row_group = tile_row_group[tid]
        col_group = tile_col_group[tid]
        j_sorted = col_group * TILE + lane
        j_orig = sorted_atom_index[j_sorted]
        pj_x = sorted_pos_x[j_sorted]
        pj_y = sorted_pos_y[j_sorted]
        pj_z = sorted_pos_z[j_sorted]
        pi_x_tile = wp.tile_load(sorted_pos_x, shape=TILE, offset=row_group * TILE)
        pi_y_tile = wp.tile_load(sorted_pos_y, shape=TILE, offset=row_group * TILE)
        pi_z_tile = wp.tile_load(sorted_pos_z, shape=TILE, offset=row_group * TILE)
        i_orig_tile = wp.tile_load(
            sorted_atom_index, shape=TILE, offset=row_group * TILE
        )
        cell_mat = cell[system_idx]
        inv_cell_mat = inv_cell[system_idx]
        fractional_rounding_certified = _triclinic_cutoff_is_certified(
            inv_cell_mat, wp.sqrt(cutoff_sq)
        )
        triclinic_qr = _prepare_triclinic_qr_if_needed(
            cell_mat, fractional_rounding_certified
        )
        qr_height_certified = wp.bool(False)
        if not fractional_rounding_certified:
            qr_height_certified = _triclinic_qr_height_is_certified(
                triclinic_qr, wp.sqrt(cutoff_sq)
            )
        reverse_count = wp.int32(0)
        for i_local in range(TILE_GROUP_SIZE):
            i_sorted = row_group * TILE + i_local
            i_orig = wp.tile_extract(i_orig_tile, i_local)
            d = wp.vec3f(
                pj_x - wp.tile_extract(pi_x_tile, i_local),
                pj_y - wp.tile_extract(pi_y_tile, i_local),
                pj_z - wp.tile_extract(pi_z_tile, i_local),
            )
            wrapped, _ = _wrap_triclinic_pair(
                d,
                cell_mat,
                inv_cell_mat,
                triclinic_qr,
                fractional_rounding_certified,
                qr_height_certified,
            )
            if _direct_coo_pair_is_accepted(
                i_sorted,
                j_sorted,
                i_orig,
                j_orig,
                natom,
                wp.dot(wrapped, wrapped),
                cutoff_sq,
            ):
                reverse_count += wp.int32(1)

        reverse_base = wp.int32(0)
        if reverse_count > 0:
            reverse_base = wp.atomic_add(cursors, j_orig, reverse_count)
        reverse_cursor = wp.int32(0)

        for i_local in range(TILE_GROUP_SIZE):
            i_sorted = row_group * TILE + i_local
            i_orig = wp.tile_extract(i_orig_tile, i_local)
            d = wp.vec3f(
                pj_x - wp.tile_extract(pi_x_tile, i_local),
                pj_y - wp.tile_extract(pi_y_tile, i_local),
                pj_z - wp.tile_extract(pi_z_tile, i_local),
            )
            wrapped, shift = _wrap_triclinic_pair(
                d,
                cell_mat,
                inv_cell_mat,
                triclinic_qr,
                fractional_rounding_certified,
                qr_height_certified,
            )
            distance_sq = wp.dot(wrapped, wrapped)
            valid = wp.int32(0)
            if _direct_coo_pair_is_accepted(
                i_sorted,
                j_sorted,
                i_orig,
                j_orig,
                natom,
                distance_sq,
                cutoff_sq,
            ):
                valid = wp.int32(1)

            valid_tile = wp.tile(valid)
            scan_tile = wp.tile_scan_inclusive(valid_tile)
            forward_count = wp.tile_extract(scan_tile, TILE - 1)
            forward_base = wp.int32(0)
            if lane == 0 and forward_count > 0:
                forward_base = wp.atomic_add(cursors, i_orig, forward_count)
            forward_base_tile = wp.tile_from_thread(
                shape=TILE,
                value=forward_base,
                thread_idx=0,
                storage="shared",
            )
            forward_base_lane = wp.untile(forward_base_tile)
            scan_lane = wp.untile(scan_tile)

            if valid == 1:
                forward = forward_base_lane + scan_lane - 1
                reverse = reverse_base + reverse_cursor
                reverse_cursor += wp.int32(1)
                distance = wp.float32(0.0)
                if RETURN_DISTANCES or HAS_PAIR_FN:
                    distance = wp.sqrt(distance_sq)

                if forward >= 0 and forward < physical_capacity:
                    coo_list[forward, 0] = i_orig
                    coo_list[forward, 1] = j_orig
                    coo_shifts[forward, 0] = shift[0]
                    coo_shifts[forward, 1] = shift[1]
                    coo_shifts[forward, 2] = shift[2]
                    if RETURN_VECTORS:
                        neighbor_vectors[forward] = wrapped
                    if RETURN_DISTANCES or HAS_PAIR_FN:
                        if RETURN_DISTANCES:
                            neighbor_distances[forward] = distance
                        if HAS_PAIR_FN:
                            energy, force = pair_fn(
                                wrapped, distance, pair_params, i_orig, j_orig
                            )
                            pair_energies[forward] = energy
                            pair_forces[forward] = force
                if reverse >= 0 and reverse < physical_capacity:
                    coo_list[reverse, 0] = j_orig
                    coo_list[reverse, 1] = i_orig
                    coo_shifts[reverse, 0] = -shift[0]
                    coo_shifts[reverse, 1] = -shift[1]
                    coo_shifts[reverse, 2] = -shift[2]
                    if RETURN_VECTORS:
                        neighbor_vectors[reverse] = -wrapped
                    if RETURN_DISTANCES or HAS_PAIR_FN:
                        if RETURN_DISTANCES:
                            neighbor_distances[reverse] = distance
                        if HAS_PAIR_FN:
                            energy, force = pair_fn(
                                -wrapped, distance, pair_params, j_orig, i_orig
                            )
                            pair_energies[reverse] = energy
                            pair_forces[reverse] = force

    return _kernel


@lru_cache(maxsize=None)
def _make_query_cluster_tile_coo_kernel(
    *,
    batched: bool,
    tile_segmented: bool = False,
    coo_segmented: bool = False,
    selective: bool = False,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
) -> wp.Kernel:
    """Build the tile-to-COO conversion kernel for one feature combination."""
    BATCHED = wp.constant(bool(batched))
    TILE_SEGMENTED = wp.constant(bool(tile_segmented))
    COO_SEGMENTED = wp.constant(bool(coo_segmented))
    SELECTIVE = wp.constant(bool(selective))
    RETURN_VECTORS = wp.constant(bool(return_vectors))
    RETURN_DISTANCES = wp.constant(bool(return_distances))
    HAS_PAIR_FN = wp.constant(pair_fn is not None)

    @wp.kernel(enable_backward=False, module="unique")
    def _kernel(
        sorted_pos_x: wp.array(dtype=wp.float32),
        sorted_pos_y: wp.array(dtype=wp.float32),
        sorted_pos_z: wp.array(dtype=wp.float32),
        sorted_atom_index: wp.array(dtype=wp.int32),
        cell: wp.array(dtype=wp.mat33f),
        inv_cell: wp.array(dtype=wp.mat33f),
        cutoff_sq: wp.float32,
        natom: wp.int32,
        max_pairs: wp.int32,
        tile_row_group: wp.array(dtype=wp.int32),
        tile_col_group: wp.array(dtype=wp.int32),
        tile_system: wp.array(dtype=wp.int32),
        num_tiles: wp.array(dtype=wp.int32),
        tile_offsets: wp.array(dtype=wp.int32),
        tile_counts: wp.array(dtype=wp.int32),
        rebuild_flags: wp.array(dtype=wp.bool),
        pair_counter: wp.array(dtype=wp.int32),
        pair_offsets: wp.array(dtype=wp.int32),
        pair_counts: wp.array(dtype=wp.int32),
        coo_list: wp.array2d(dtype=wp.int32),
        coo_shifts: wp.array2d(dtype=wp.int32),
        neighbor_vectors: wp.array(dtype=wp.vec3f),
        neighbor_distances: wp.array(dtype=wp.float32),
        pair_params: wp.array2d(dtype=wp.float32),
        pair_energies: wp.array(dtype=wp.float32),
        pair_forces: wp.array(dtype=wp.vec3f),
    ) -> None:
        """Convert cluster-tile pairs into COO rows

        Parameters
        ----------
        sorted_pos_x, sorted_pos_y, sorted_pos_z : wp.array, shape (natom_padded,), dtype=wp.float32
            Sorted atom coordinates.
        sorted_atom_index : wp.array, shape (natom_padded,), dtype=wp.int32
            Original atom index for each sorted slot.
        cell, inv_cell : wp.array, shape (num_systems,), dtype=wp.mat33f
            Cell and inverse-cell matrices.
        cutoff_sq : wp.float32
            Squared cutoff distance.
        natom : wp.int32
            Number of real atoms before padding.
        max_pairs : wp.int32
            Physical COO output capacity for both compact and segmented modes.
            Segmented offsets must describe intervals within this capacity.
        tile_row_group, tile_col_group, tile_system : wp.array, shape (num_tiles,), dtype=wp.int32
            Cluster-tile records from the build launcher.
        num_tiles : wp.array, shape (1,), dtype=wp.int32
            Compact tile count for non-segmented tile lists.
        tile_offsets : wp.array, shape (num_systems + 1,), dtype=wp.int32
            Fixed per-system tile offsets for segmented tile lists.
        tile_counts : wp.array, shape (num_systems,), dtype=wp.int32
            Per-system tile counts for segmented tile lists.
        rebuild_flags : wp.array, shape (num_systems,), dtype=wp.bool
            Per-system selective rebuild flags. Sentinel when inactive.
        pair_counter : wp.array, shape (1,), dtype=wp.int32
            OUTPUT: Compact COO pair count. Sentinel in segmented COO mode.
        pair_offsets : wp.array, shape (num_systems + 1,), dtype=wp.int32
            Caller-owned fixed output offsets for segmented COO mode.
        pair_counts : wp.array, shape (num_systems,), dtype=wp.int32
            OUTPUT: Per-system pair counts for segmented COO mode.
        coo_list : wp.array, shape (max_pairs, 2), dtype=wp.int32
            OUTPUT: COO atom-index pairs.
        coo_shifts : wp.array, shape (max_pairs, 3), dtype=wp.int32
            OUTPUT: Periodic shift vectors for COO pairs.
        neighbor_vectors : wp.array, shape (max_pairs,), dtype=wp.vec3f
            OUTPUT: Optional displacement vectors. Sentinel when disabled.
        neighbor_distances : wp.array, shape (max_pairs,), dtype=wp.float32
            OUTPUT: Optional pair distances. Sentinel when disabled.
        pair_params : wp.array, shape (natom, K), dtype=wp.float32
            Pair-function parameters. Sentinel when no pair function is active.
        pair_energies : wp.array, shape (max_pairs,), dtype=wp.float32
            OUTPUT: Optional pair-function energies. Sentinel when disabled.
        pair_forces : wp.array, shape (max_pairs,), dtype=wp.vec3f
            OUTPUT: Optional pair-function forces. Sentinel when disabled.

        Returns
        -------
        None
            This function modifies the input arrays in-place.

        Notes
        -----
        - Thread launch: Tiled launch with one block per allocated tile slot and block dimension ``TILE_GROUP_SIZE``.
        - Modifies: ``pair_counter`` or ``pair_counts``, ``coo_list``, ``coo_shifts``, and enabled pair-output buffers.
        Segmented COO writes reserve from ``pair_counts`` only for valid output
        intervals. Non-batched queries require the complete physical interval
        ``[0, max_pairs]``; batched queries accept physically bounded
        per-system subsegments. Valid segments retain attempted counts for
        wrapper overflow reporting; invalid segments retain zero active pairs.

        See Also
        --------
        query_cluster_tile_coo : Launch cluster-tile neighbor search with COO-style outputs.
        """
        tile = wp.tid()
        system_idx = wp.int32(0)

        if TILE_SEGMENTED:
            if BATCHED:
                system_idx = tile_system[tile]
            local_tile = tile - tile_offsets[system_idx]
            # Bound to the per-system region capacity (mirrors the build's
            # ``local_slot < capacity`` write clamp).  ``tile_counts`` is
            # incremented unclamped, so on a tile-capacity overflow it can
            # exceed ``capacity``; without this clamp an over-counted system
            # plus zeroed ``tile_system`` gap slots can misattribute tiles
            # from a later region.
            capacity = tile_offsets[system_idx + 1] - tile_offsets[system_idx]
            if (
                local_tile < 0
                or local_tile >= tile_counts[system_idx]
                or local_tile >= capacity
            ):
                return
        else:
            if tile >= num_tiles[0]:
                return
            if BATCHED:
                system_idx = tile_system[tile]

        if SELECTIVE:
            if not rebuild_flags[system_idx]:
                return

        lane_tile = wp.tile_arange(TILE, dtype=wp.int32)
        lane = wp.untile(lane_tile)

        row_group = tile_row_group[tile]
        col_group = tile_col_group[tile]

        cell_mat = cell[system_idx]
        inv_cell_mat = inv_cell[system_idx]
        fractional_rounding_certified = _triclinic_cutoff_is_certified(
            inv_cell_mat, wp.sqrt(cutoff_sq)
        )
        triclinic_qr = _prepare_triclinic_qr_if_needed(
            cell_mat, fractional_rounding_certified
        )
        qr_height_certified = wp.bool(False)
        if not fractional_rounding_certified:
            qr_height_certified = _triclinic_qr_height_is_certified(
                triclinic_qr, wp.sqrt(cutoff_sq)
            )

        j_sorted = col_group * TILE + lane
        j_orig_t = sorted_atom_index[j_sorted]
        pj_x = sorted_pos_x[j_sorted]
        pj_y = sorted_pos_y[j_sorted]
        pj_z = sorted_pos_z[j_sorted]

        pi_x_tile = wp.tile_load(sorted_pos_x, shape=TILE, offset=row_group * TILE)
        pi_y_tile = wp.tile_load(sorted_pos_y, shape=TILE, offset=row_group * TILE)
        pi_z_tile = wp.tile_load(sorted_pos_z, shape=TILE, offset=row_group * TILE)
        i_orig_tile = wp.tile_load(
            sorted_atom_index, shape=TILE, offset=row_group * TILE
        )

        for i_local in range(TILE_GROUP_SIZE):
            i_sorted = row_group * TILE + i_local
            i_orig = wp.tile_extract(i_orig_tile, i_local)
            pi_x = wp.tile_extract(pi_x_tile, i_local)
            pi_y = wp.tile_extract(pi_y_tile, i_local)
            pi_z = wp.tile_extract(pi_z_tile, i_local)

            d = wp.vec3f(pj_x - pi_x, pj_y - pi_y, pj_z - pi_z)
            d_wrapped, s_shift = _wrap_triclinic_pair(
                d,
                cell_mat,
                inv_cell_mat,
                triclinic_qr,
                fractional_rounding_certified,
                qr_height_certified,
            )
            dist_sq = (
                d_wrapped[0] * d_wrapped[0]
                + d_wrapped[1] * d_wrapped[1]
                + d_wrapped[2] * d_wrapped[2]
            )

            valid = wp.int32(0)
            if BATCHED:
                if (
                    i_sorted < j_sorted
                    and i_orig < natom
                    and j_orig_t < natom
                    and dist_sq < cutoff_sq
                ):
                    valid = wp.int32(1)
            else:
                if (
                    i_sorted < j_sorted
                    and i_sorted < natom
                    and j_sorted < natom
                    and dist_sq < cutoff_sq
                ):
                    valid = wp.int32(1)

            valid_tile = wp.tile(valid)
            scan_tile = wp.tile_scan_inclusive(valid_tile)
            i_count = wp.tile_extract(scan_tile, TILE - 1)

            if i_count > 0:
                base = wp.int32(0)
                capacity = max_pairs
                offset = wp.int32(0)
                segment_valid = wp.int32(1)
                # Full-fill: reserve two slots per valid pair (forward (i, j)
                # and reverse (j, i)), so the COO list carries each ordered
                # pair once -- matching cell_list / naive (half_fill=False).
                if COO_SEGMENTED:
                    start = pair_offsets[system_idx]
                    stop = pair_offsets[system_idx + 1]
                    if BATCHED:
                        segment_valid = _coo_segment_is_valid(
                            start,
                            stop,
                            max_pairs,
                        )
                    else:
                        if start == 0 and stop == max_pairs:
                            segment_valid = wp.int32(1)
                        else:
                            segment_valid = wp.int32(0)
                    if segment_valid == 1:
                        capacity = stop - start
                        offset = start
                        if lane == 0:
                            base = wp.atomic_add(
                                pair_counts,
                                system_idx,
                                2 * i_count,
                            )
                    else:
                        capacity = wp.int32(0)
                else:
                    if lane == 0:
                        base = wp.atomic_add(pair_counter, 0, 2 * i_count)
                base_tile = wp.tile_from_thread(
                    shape=TILE,
                    value=base,
                    thread_idx=0,
                    storage="shared",
                )
                base_b = wp.untile(base_tile)
                scan_v = wp.untile(scan_tile)
                local_pos = base_b + scan_v - 1
                local_rev = base_b + i_count + scan_v - 1
                write_pos = offset + local_pos
                write_rev = offset + local_rev

                if (
                    valid == 1
                    and segment_valid == 1
                    and local_pos < capacity
                    and write_pos >= 0
                    and write_pos < max_pairs
                ):
                    coo_list[write_pos, 0] = i_orig
                    coo_list[write_pos, 1] = j_orig_t
                    coo_shifts[write_pos, 0] = s_shift[0]
                    coo_shifts[write_pos, 1] = s_shift[1]
                    coo_shifts[write_pos, 2] = s_shift[2]
                    if RETURN_VECTORS:
                        neighbor_vectors[write_pos] = d_wrapped
                    if RETURN_DISTANCES or HAS_PAIR_FN:
                        distance = wp.sqrt(dist_sq)
                        if RETURN_DISTANCES:
                            neighbor_distances[write_pos] = distance
                        if HAS_PAIR_FN:
                            pair_energy, pair_force = pair_fn(
                                d_wrapped,
                                distance,
                                pair_params,
                                i_orig,
                                j_orig_t,
                            )
                            pair_energies[write_pos] = pair_energy
                            pair_forces[write_pos] = pair_force

                if (
                    valid == 1
                    and segment_valid == 1
                    and local_rev < capacity
                    and write_rev >= 0
                    and write_rev < max_pairs
                ):
                    coo_list[write_rev, 0] = j_orig_t
                    coo_list[write_rev, 1] = i_orig
                    coo_shifts[write_rev, 0] = -s_shift[0]
                    coo_shifts[write_rev, 1] = -s_shift[1]
                    coo_shifts[write_rev, 2] = -s_shift[2]
                    if RETURN_VECTORS:
                        neighbor_vectors[write_rev] = -d_wrapped
                    if RETURN_DISTANCES or HAS_PAIR_FN:
                        distance_r = wp.sqrt(dist_sq)
                        if RETURN_DISTANCES:
                            neighbor_distances[write_rev] = distance_r
                        if HAS_PAIR_FN:
                            pair_energy_r, pair_force_r = pair_fn(
                                -d_wrapped,
                                distance_r,
                                pair_params,
                                j_orig_t,
                                i_orig,
                            )
                            pair_energies[write_rev] = pair_energy_r
                            pair_forces[write_rev] = pair_force_r

    name = kernel_specialization_name(
        "_query_cluster_tile_coo",
        wp_dtype=wp.float32,
        features=(
            "batched" if batched else "",
            "tile_segmented" if tile_segmented else "",
            *_pair_output_features(
                selective=selective,
                coo_segmented=coo_segmented,
                return_vectors=return_vectors,
                return_distances=return_distances,
                pair_fn=pair_fn,
            ),
        ),
    )
    return set_fn_doc(
        set_fn_name(_kernel, name),
        _append_specialization_doc(
            _kernel.__doc__,
            dtype=wp.float32,
            entries=(
                ("operation", "query_cluster_tile_coo"),
                ("output", "coo"),
                ("batched", bool(batched)),
                ("tile_segmented", bool(tile_segmented)),
                ("coo_segmented", bool(coo_segmented)),
                ("selective", bool(selective)),
                ("return_vectors", bool(return_vectors)),
                ("return_distances", bool(return_distances)),
                ("pair_fn", pair_fn is not None),
            ),
        ),
    )


def get_query_cluster_tile_coo_kernel(
    *,
    tile_segmented: bool = False,
    coo_segmented: bool = False,
    selective: bool = False,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
) -> wp.Kernel:
    """Return a cached single-system tile-to-COO kernel."""
    return _make_query_cluster_tile_coo_kernel(
        batched=False,
        tile_segmented=bool(tile_segmented),
        coo_segmented=bool(coo_segmented),
        selective=bool(selective),
        return_vectors=bool(return_vectors),
        return_distances=bool(return_distances),
        pair_fn=pair_fn,
    )


def get_batch_query_cluster_tile_coo_kernel(
    *,
    tile_segmented: bool = False,
    coo_segmented: bool = False,
    selective: bool = False,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: wp.Function | None = None,
) -> wp.Kernel:
    """Return a cached batched tile-to-COO kernel."""
    return _make_query_cluster_tile_coo_kernel(
        batched=True,
        tile_segmented=bool(tile_segmented),
        coo_segmented=bool(coo_segmented),
        selective=bool(selective),
        return_vectors=bool(return_vectors),
        return_distances=bool(return_distances),
        pair_fn=pair_fn,
    )
