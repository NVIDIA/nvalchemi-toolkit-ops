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


"""Pair-centric grids selected from cell visits and source-atom passes.

For C cells, stencil volume S and occupancy q = ceil(N/C), count setup warps
and active source-warp passes, including a partly occupied final pass. The batched
estimate is C*S*(setup_warps + active_warp_passes*q). Alternatives must preserve
or reduce source passes. Compare the box/cutoff grid, the existing minimum,
and a minimum derived from N/block_dim. Single-system alternatives also preserve
estimated candidate pairs, neighbor-loop depth, cell storage and logical blocks.
Ties retain the configured grid.
The model uses current inputs; it assumes approximately uniform occupancy.
"""

from functools import lru_cache

import warp as wp

__all__ = []


@wp.kernel(enable_backward=False)
def _validated_cell_total(
    counts: wp.array(dtype=wp.int32),
    total: wp.array(dtype=wp.int64),
) -> None:
    """Reduce positive per-system cell counts to one allocation size.

    Parameters
    ----------
    counts : wp.array, shape (num_systems,)
        Cell counts for each system.
    total : wp.array, shape (1,)
        Output allocation size.

    Returns
    -------
    None
        The result is written to ``total[0]``.

    Notes
    -----
    Thread launch: One tiled block; lanes traverse strided system indices.
    Modifies: ``total[0]`` receives the int64 sum, or zero if any count is
    nonpositive. An empty input also produces zero.
    """
    _, lane = wp.tid()
    partial = wp.int64(0)
    valid = wp.int32(1)
    for i in range(lane, counts.shape[0], wp.block_dim()):
        count = counts[i]
        partial += wp.int64(count)
        if count <= 0:
            valid = 0
    summed = wp.tile_sum(wp.tile(partial))
    all_valid = wp.tile_min(wp.tile(valid))
    if lane == 0:
        value = wp.int64(0)
        if wp.tile_extract(all_valid, 0) != 0:
            value = wp.tile_extract(summed, 0)
        total[0] = value


def _validate_grid_policy(grid_policy: str) -> None:
    """Require a supported cell-grid policy using host metadata only."""
    if grid_policy not in {"configured", "adaptive"}:
        raise ValueError(
            f"grid_policy must be 'configured' or 'adaptive', got {grid_policy!r}"
        )


@lru_cache(maxsize=None)
def _get_pair_grid_kernel(
    dtype: type,
    block_dim: int,
    single_system: bool = False,
    scalar_population: bool = False,
    write_counts: bool = True,
) -> wp.Kernel:
    """Create the pair-grid sizing kernel for a query block width.

    Parameters
    ----------
    dtype : type
        Warp floating-point type used for cell geometry.
    block_dim : int
        Thread block width of the pair-centric query.
    single_system : bool, default False
        Apply constraints for a single-system query.
    scalar_population : bool, default False
        Read the system population from ``num_atoms`` instead of boundaries.
    write_counts : bool, default True
        Write per-system cell counts to the output array.

    Returns
    -------
    wp.Kernel
        Kernel that selects grids, stencil radii, and cell counts from the
        current geometry and atom populations.

    Notes
    -----
    Each batch system is scored using its own population and stencil radius.
    The batch query launch uses the maximum stencil radius across systems, so
    this per-system estimate does not model the extra work that maximum can
    impose on systems with shorter stencils.
    """
    mat = wp.mat33 if dtype == wp.float32 else wp.mat33d
    vec = wp.vec3 if dtype == wp.float32 else wp.vec3d
    block = wp.constant(block_dim)
    # CUDA execution groups contain 32 lanes; this is a hardware invariant.
    warp_width = wp.constant(32)
    block_warps = wp.constant((block_dim + 31) // 32)
    single = wp.constant(single_system)
    scalar = wp.constant(scalar_population)
    store_counts = wp.constant(write_counts)
    flat = wp.constant(single_system and not write_counts)
    pbc_array = wp.array(dtype=wp.bool) if flat else wp.array2d(dtype=wp.bool)
    grid_array = wp.array(dtype=wp.int32) if flat else wp.array(dtype=wp.vec3i)

    @wp.kernel(enable_backward=False)
    def kernel(
        cell: wp.array(dtype=mat),
        pbc: pbc_array,
        boundaries: wp.array(dtype=wp.int32),
        num_atoms: wp.int32,
        cutoff: dtype,
        cap: wp.int32,
        compatibility_minimum: wp.int32,
        grids: grid_array,
        radii: grid_array,
        counts: wp.array(dtype=wp.int32),
    ):
        """Select a grid using cell visits and estimated source-atom passes.

        Parameters
        ----------
        cell : wp.array
            Cell matrices, shape (num_systems,), in the factory's precision.
        pbc : wp.array2d
            Periodic flags, shape (num_systems, 3), or (3,) for flat outputs.
        boundaries : wp.array
            Cumulative atom offsets, shape (num_systems + 1,).
        num_atoms : int
            Single-system atom count for the scalar-population specialization.
        cutoff : float
            Neighbor cutoff in the same length units as cell.
        cap : int
            Available cell capacity per system.
        compatibility_minimum : int
            Existing per-axis minimum used as one candidate grid.
        grids : wp.array
            OUTPUT: Selected per-axis cell counts, shape (num_systems,),
            or three scalar entries for flat outputs.
        radii : wp.array
            OUTPUT: Selected per-axis stencil radii, matching grids.
        counts : wp.array
            OUTPUT: Total cell counts, shape (num_systems,).

        Returns
        -------
        None
            Results are written into the output arrays.

        Notes
        -----
        - Thread launch: One thread per system.
        - Modifies: grids, radii, and counts for each system.
        The score assumes approximately uniform occupancy. Single-system alternatives
        preserve or reduce candidate pairs, neighbor-loop depth, cell storage, and
        logical blocks relative to the configured grid. Equal scores retain that grid.

        See Also
        --------
        _get_pair_grid_kernel : Specialize this kernel for a query block width.
        """
        system = wp.tid()
        atoms = num_atoms
        if not scalar:
            atoms = boundaries[system + 1] - boundaries[system]
        if wp.determinant(cell[system]) == dtype(0.0):
            if flat:
                for axis in range(3):
                    grids[axis] = 1
                    radii[axis] = 0
            else:
                grids[system] = wp.vec3i(1)
                radii[system] = wp.vec3i(0)
            if store_counts:
                counts[system] = 0
            return
        inverse = wp.transpose(wp.inverse(cell[system]))
        faces = vec(dtype(0.0))
        natural = wp.vec3i(0)
        periodic = wp.vec3i(0)
        for axis in range(3):
            if flat:
                periodic[axis] = wp.int32(pbc[axis])
            else:
                periodic[axis] = wp.int32(pbc[system, axis])
            faces[axis] = dtype(1.0) / wp.length(inverse[axis])
            natural[axis] = wp.max(
                wp.int32(wp.min(faces[axis] / cutoff, dtype(cap))), 1
            )
        occupancy_minimum = wp.int32(
            wp.ceil(
                wp.pow(
                    wp.float64(wp.max(atoms, 1)) / wp.float64(block),
                    wp.float64(1.0 / 3.0),
                )
            )
        )
        while wp.int64(occupancy_minimum) * wp.int64(occupancy_minimum) * wp.int64(
            occupancy_minimum
        ) * wp.int64(block) < wp.int64(atoms):
            occupancy_minimum += 1
        while occupancy_minimum > 1 and wp.int64(occupancy_minimum - 1) * wp.int64(
            occupancy_minimum - 1
        ) * wp.int64(occupancy_minimum - 1) * wp.int64(block) >= wp.int64(atoms):
            occupancy_minimum -= 1
        best_score = wp.float64(0.0)
        best_valid = False
        best_grid = wp.vec3i(1)
        best_radius = wp.vec3i(0)
        best_count = int(1)
        baseline_cells = wp.int64(1)
        baseline_visits = wp.int64(1)
        baseline_depth = wp.int64(1)
        baseline_passes = wp.int64(1)
        for iteration in range(3):
            candidate = iteration
            # Evaluate the configured grid first so it bounds every alternative.
            if iteration == 0:
                candidate = 1
            elif iteration == 1:
                candidate = 0
            minimum = int(1)
            if candidate == 1:
                minimum = compatibility_minimum
            elif candidate == 2:
                minimum = occupancy_minimum
            grid = natural
            for axis in range(3):
                if periodic[axis] != 0 or grid[axis] > 1:
                    if candidate == 1:
                        # Retain the configured minimum's existing doubling rule.
                        while grid[axis] < minimum:
                            grid[axis] *= 2
                    else:
                        grid[axis] = wp.max(grid[axis], minimum)
            cells = wp.int64(grid[0]) * wp.int64(grid[1]) * wp.int64(grid[2])
            while cells > wp.int64(cap):
                for axis in range(3):
                    grid[axis] = wp.max(grid[axis] // 2, 1)
                cells = wp.int64(grid[0]) * wp.int64(grid[1]) * wp.int64(grid[2])
            radius = wp.vec3i(0)
            visits = wp.int64(1)
            for axis in range(3):
                if periodic[axis] != 0 or grid[axis] > 1:
                    radius[axis] = wp.int32(
                        wp.ceil(cutoff * dtype(grid[axis]) / faces[axis])
                    )
                visits *= wp.int64(2 * radius[axis] + 1)
            passes = (wp.int64(atoms) + wp.int64(block) * cells - wp.int64(1)) // (
                wp.int64(block) * cells
            )
            occupancy = (wp.int64(atoms) + cells - wp.int64(1)) // cells
            # Count active warp passes, including the partly occupied last pass.
            active_warps = (occupancy // wp.int64(block)) * wp.int64(block_warps)
            active_warps += (
                occupancy % wp.int64(block) + wp.int64(warp_width) - wp.int64(1)
            ) // wp.int64(warp_width)
            score = (
                wp.float64(visits)
                * wp.float64(cells)
                * (
                    wp.float64(block_warps)
                    + wp.float64(active_warps) * wp.float64(occupancy)
                )
            )
            if single:
                score = wp.float64(visits * passes)
            if candidate == 1:
                baseline_passes = passes
            # Additional source passes lengthen the serial loop within a block.
            eligible = passes <= baseline_passes
            if single:
                # The inner neighbor loop is serial within each source lane.
                depth = passes * occupancy
                if candidate == 1:
                    baseline_cells = cells
                    baseline_visits = visits
                    baseline_depth = depth
                else:
                    # Candidate pairs scale as N^2 * visits/cells. Compare ratios
                    # without multiplying by N^2, which can overflow int64.
                    eligible = (
                        eligible
                        and cells <= baseline_cells
                        and depth <= baseline_depth
                        and wp.float64(visits) / wp.float64(cells)
                        <= wp.float64(baseline_visits) / wp.float64(baseline_cells)
                        and wp.float64(cells) * wp.float64(visits)
                        <= wp.float64(baseline_cells) * wp.float64(baseline_visits)
                    )
            if eligible and (not best_valid or score < best_score):
                best_valid = True
                best_score = score
                best_grid = grid
                best_radius = radius
                best_count = wp.int32(cells)
        if flat:
            for axis in range(3):
                grids[axis] = best_grid[axis]
                radii[axis] = best_radius[axis]
        else:
            grids[system] = best_grid
            radii[system] = best_radius
        if store_counts:
            counts[system] = best_count

    return kernel
