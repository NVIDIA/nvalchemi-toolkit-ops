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
import math

import torch

# Mathematical constants
PI = math.pi
TWOPI = 2.0 * PI

__all__ = [
    "generate_ewald_miller_indices",
    "generate_k_vectors_ewald_summation",
    "generate_k_vectors_pme",
    "k_vectors_from_miller_indices",
]


def _prepare_k_cutoff(
    cell: torch.Tensor,
    k_cutoff: float | torch.Tensor,
) -> torch.Tensor:
    """Normalize k_cutoff for shared batch Miller bounds."""
    k_cutoff_tensor = torch.as_tensor(k_cutoff, device=cell.device, dtype=cell.dtype)
    if k_cutoff_tensor.ndim == 0 or k_cutoff_tensor.numel() == 1:
        return k_cutoff_tensor.reshape(())
    return k_cutoff_tensor.max()


def _generate_miller_indices(
    cell: torch.Tensor,
    k_cutoff: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate Miller indices for Ewald summation.

    Parameters
    ----------
    cell : torch.Tensor, shape (N, 3, 3)
        Unit cell matrices with lattice vectors as rows.
    k_cutoff : float | torch.Tensor
        Maximum magnitude of k-vectors to include in reciprocal summation.

    Notes
    -----
    For batch mode, one shared set of Miller bounds is used for all systems.
    If ``k_cutoff`` is provided per system, the maximum cutoff across the batch
    is used to build those shared bounds.
    """
    shared_k_cutoff = _prepare_k_cutoff(cell, k_cutoff)
    cell_lengths = (torch.norm(cell, dim=-1).max(dim=0).values) / (
        2 * torch.pi
    )  # Length of each reciprocal vector
    return torch.ceil(shared_k_cutoff * cell_lengths).long()


def _ewald_miller_grid(
    cell: torch.Tensor,
    k_cutoff: float | torch.Tensor,
    miller_bounds: tuple[int, int, int] | torch.Tensor | None,
) -> torch.Tensor:
    """Build the legacy floating-point FFT-grid Miller representation."""
    if cell.ndim == 2:
        cell = cell.unsqueeze(0)
    device = cell.device
    dtype = cell.dtype
    if miller_bounds is None:
        max_h, max_k, max_l = 2 * _generate_miller_indices(cell, k_cutoff) + 1
    else:
        if isinstance(miller_bounds, torch.Tensor):
            if miller_bounds.numel() != 3:
                raise ValueError("miller_bounds tensor must contain exactly 3 values")
            bounds = tuple(int(v) for v in miller_bounds.reshape(3).tolist())
        else:
            if len(miller_bounds) != 3:
                raise ValueError("miller_bounds must contain exactly 3 values")
            bounds = tuple(int(v) for v in miller_bounds)
        max_h, max_k, max_l = (2 * bounds[0] + 1, 2 * bounds[1] + 1, 2 * bounds[2] + 1)

    h_range = torch.fft.fftfreq(max_h, device=device, dtype=dtype) * max_h
    k_range = torch.fft.fftfreq(max_k, device=device, dtype=dtype) * max_k
    l_range = torch.fft.fftfreq(max_l, device=device, dtype=dtype) * max_l
    h_grid, k_grid, l_grid = torch.meshgrid(h_range, k_range, l_range, indexing="ij")
    miller_grid = torch.stack(
        [h_grid.flatten(), k_grid.flatten(), l_grid.flatten()], dim=1
    )
    h, k, m = miller_grid.unbind(dim=1)
    halfspace_mask = (h > 0) | ((h == 0) & (k > 0)) | ((h == 0) & (k == 0) & (m > 0))
    return miller_grid[halfspace_mask]


def generate_ewald_miller_indices(
    cell: torch.Tensor,
    k_cutoff: float | torch.Tensor,
    miller_bounds: tuple[int, int, int] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Generate the positive-half-space Miller topology for Ewald summation.

    The returned signed integer rows are caller-owned topology. They can be
    retained across cell updates and transformed with
    :func:`k_vectors_from_miller_indices`.

    Half-Space Optimization
    -----------------------
    This function generates k-vectors in the positive half-space only, exploiting
    the symmetry S(-k) = S*(k) where S(k) is the structure factor. For each pair
    of k-vectors (k, -k), only one is included.

    The half-space condition selects k-vectors where:
        - h > 0, OR
        - (h == 0 AND k > 0), OR
        - (h == 0 AND k == 0 AND l > 0)

    The kernels in ewald_kernels.py compensate by doubling the Green's function
    (using :math:`8\\pi` instead of :math:`4\\pi`), so energies, forces, and charge gradients are
    computed correctly.

    Mathematical Background
    -----------------------
    For a direct lattice defined by basis vectors {a, b, c} (rows of cell matrix),
    the reciprocal lattice vectors are:

    .. math::

        \\mathbf{a}^* &= \\frac{2\\pi (\\mathbf{b} \\times \\mathbf{c})}{V}

        \\mathbf{b}^* &= \\frac{2\\pi (\\mathbf{c} \\times \\mathbf{a})}{V}

        \\mathbf{c}^* &= \\frac{2\\pi (\\mathbf{a} \\times \\mathbf{b})}{V}

    where :math:`V = \\mathbf{a} \\cdot (\\mathbf{b} \\times \\mathbf{c})` is the cell volume.

    In matrix form: :math:`\\text{reciprocal_matrix} = 2\\pi \\cdot (\\text{cell}^T)^{-1}`

    Each k-vector is: :math:`\\mathbf{k} = h \\mathbf{a}^* + k \\mathbf{b}^* + l \\mathbf{c}^*`
    where (h, k, l) are Miller indices (integers).

    Parameters
    ----------
    cell : torch.Tensor
        Unit cell matrix with lattice vectors as rows.
        Shape (3, 3) for single system or (B, 3, 3) for batch.
    k_cutoff : float or torch.Tensor
        Reciprocal cutoff used to derive conservative per-axis Miller bounds.
        The resulting rectangular topology can contain vectors whose magnitude
        exceeds this value.
    miller_bounds : tuple[int, int, int] or torch.Tensor, optional
        Explicit Miller half-bounds ``(M_h, M_k, M_l)``. When supplied, their
        rectangle is enumerated directly and ``k_cutoff`` does not select
        individual rows. Python integer bounds avoid deriving range sizes from
        device tensors; tensor bounds are accepted for convenience.

    Returns
    -------
    torch.Tensor
        Signed integer Miller indices of shape ``(K, 3)``. The rows are
        nonzero, unique, and in the positive half-space.

    Examples
    --------
    Single system with explicit k_cutoff::

        >>> cell = torch.eye(3, dtype=torch.float64) * 10.0
        >>> indices = generate_ewald_miller_indices(cell, k_cutoff=8.0)
        >>> k_vectors = k_vectors_from_miller_indices(cell, indices)

    With automatic parameter estimation::

        >>> from nvalchemiops.torch.interactions.electrostatics import estimate_ewald_parameters
        >>> params = estimate_ewald_parameters(positions, cell)
        >>> k_vectors = generate_k_vectors_ewald_summation(cell, params.reciprocal_space_cutoff)

    Notes
    -----
    For batch mode, the maximum cutoff across the batch determines shared
    bounds. A retained topology must conservatively cover every cell state in
    which it will be used.

    See Also
    --------
    k_vectors_from_miller_indices : Transform retained indices for a live cell.
    estimate_ewald_parameters : Automatic parameter estimation including k_cutoff.
    """
    return torch.round(_ewald_miller_grid(cell, k_cutoff, miller_bounds)).to(
        torch.int64
    )


def k_vectors_from_miller_indices(
    cell: torch.Tensor,
    miller_indices: torch.Tensor,
) -> torch.Tensor:
    """Materialize reciprocal vectors from caller-retained Miller indices.

    The supplied ``cell`` defines the live reciprocal transform. Duplicate
    rows, zero rows, and half-space membership are caller preconditions and are
    not checked; :func:`generate_ewald_miller_indices` produces valid topology.

    Parameters
    ----------
    cell : torch.Tensor
        Unit cell of shape ``(3, 3)`` or ``(B, 3, 3)``.
    miller_indices : torch.Tensor
        Signed integer indices of shape ``(K, 3)`` on the same device as
        ``cell``. ``(0, 3)`` is valid.

    Returns
    -------
    torch.Tensor
        Reciprocal vectors of shape ``(K, 3)`` or ``(B, K, 3)``.
    """
    if cell.ndim not in (2, 3) or cell.shape[-2:] != (3, 3):
        raise ValueError("cell must have shape (3, 3) or (B, 3, 3)")
    if miller_indices.ndim != 2 or miller_indices.shape[-1] != 3:
        raise ValueError("miller_indices must have shape (K, 3)")
    if miller_indices.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise ValueError("miller_indices must have a signed integer dtype")
    if cell.device != miller_indices.device:
        raise ValueError("cell and miller_indices must be on the same device")

    if cell.ndim == 2:
        cell = cell.unsqueeze(0)
    reciprocal_cell = TWOPI * torch.linalg.inv_ex(cell.transpose(1, 2))[0]
    return (miller_indices.to(reciprocal_cell.dtype) @ reciprocal_cell).squeeze(0)


def generate_k_vectors_ewald_summation(
    cell: torch.Tensor,
    k_cutoff: float | torch.Tensor,
    miller_bounds: tuple[int, int, int] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Generate reciprocal lattice vectors for Ewald summation (half-space).

    This compatibility wrapper preserves Torch's historical float32 FFT-grid
    rounding. Use :func:`generate_ewald_miller_indices` and
    :func:`k_vectors_from_miller_indices` for caller-retained topology.
    """
    if cell.ndim == 2:
        cell = cell.unsqueeze(0)
    reciprocal_cell = TWOPI * torch.linalg.inv_ex(cell.transpose(1, 2))[0]
    return (
        _ewald_miller_grid(cell, k_cutoff, miller_bounds) @ reciprocal_cell
    ).squeeze(0)


def generate_k_vectors_pme(
    cell: torch.Tensor,
    mesh_dimensions: tuple[int, int, int],
    reciprocal_cell: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate reciprocal lattice vectors for Particle Mesh Ewald (PME).

    Creates k-vectors on a regular grid compatible with FFT-based reciprocal
    space calculations in PME. Uses rfft conventions (half-size in z-dimension)
    to exploit Hermitian symmetry of real-valued charge densities.

    Notes
    -----
    For a direct lattice defined by basis vectors {a, b, c} (rows of cell matrix),
    the reciprocal lattice vectors are:

    .. math::

        \\begin{aligned}
        \\mathbf{a}^* &= \\frac{2\\pi (\\mathbf{b} \\times \\mathbf{c})}{V} \\\\
        \\mathbf{b}^* &= \\frac{2\\pi (\\mathbf{c} \\times \\mathbf{a})}{V} \\\\
        \\mathbf{c}^* &= \\frac{2\\pi (\\mathbf{a} \\times \\mathbf{b})}{V}
        \\end{aligned}

    where :math:`V = \\mathbf{a} \\cdot (\\mathbf{b} \\times \\mathbf{c})` is the cell volume.

    In matrix form:

    .. math::

        \\text{reciprocal_matrix} = 2\\pi \\cdot (\\text{cell}^T)^{-1}

    Each k-vector is then:

    .. math::

        \\mathbf{k} = h \\mathbf{a}^* + k \\mathbf{b}^* + l \\mathbf{c}^*

    where (h, k, l) are Miller indices (integers).

    Parameters
    ----------
    cell : torch.Tensor
        Unit cell matrix with lattice vectors as rows.
        Shape (3, 3) for single system or (B, 3, 3) for batch.
    mesh_dimensions : tuple[int, int, int]
        PME mesh grid dimensions (nx, ny, nz). Should typically be chosen
        such that mesh spacing is :math:`\\sim 1 \\text{\\AA}` or finer. Power-of-2 dimensions
        are optimal for FFT performance.
    reciprocal_cell : torch.Tensor, optional
        Precomputed reciprocal cell matrix (:math:`2\\pi \\cdot \\text{cell}^{-1}`). If provided,
        skips the inverse computation. Shape (3, 3) or (B, 3, 3).

    Returns
    -------
    k_vectors : torch.Tensor, shape (nx, ny, nz//2+1, 3)
        Cartesian k-vectors at each grid point. Uses rfft convention
        where z-dimension is halved due to Hermitian symmetry.
    k_squared_safe : torch.Tensor, shape (nx, ny, nz//2+1)
        Squared magnitude :math:`|\\mathbf{k}|^2` for each k-vector, with k=0 set to a
        small positive value (1e-12) to avoid division by zero.

    Examples
    --------
    Basic usage::

        >>> cell = torch.eye(3, dtype=torch.float64) * 10.0
        >>> mesh_dims = (32, 32, 32)
        >>> k_vectors, k_squared = generate_k_vectors_pme(cell, mesh_dims)
        >>> k_vectors.shape
        torch.Size([32, 32, 17, 3])

    With precomputed reciprocal cell::

        >>> reciprocal_cell = 2 * torch.pi * torch.linalg.inv(cell)
        >>> k_vectors, k_squared = generate_k_vectors_pme(
        ...     cell, mesh_dims, reciprocal_cell=reciprocal_cell
        ... )

    Notes
    -----
    - The z-dimension output size is nz//2+1 due to rfft symmetry.
    - Miller indices follow torch.fft.fftfreq convention (0, 1, 2, ..., -2, -1).
    - k_squared_safe has k=0 replaced with 1e-12 to prevent division by zero
      in Green's function calculations.

    See Also
    --------
    pme_reciprocal_space : Uses these k-vectors for PME reciprocal space energy.
    pme_green_structure_factor : Computes Green's function using k_squared.
    """
    device = cell.device
    dtype = cell.dtype

    # Ensure cell has batch dimension
    cell_3d = cell if cell.dim() == 3 else cell.unsqueeze(0)

    # Compute reciprocal lattice vectors (2*pi times reciprocal of direct lattice)
    if reciprocal_cell is None:
        reciprocal_cell = TWOPI * torch.linalg.inv_ex(cell_3d)[0]

    # Generate all combinations of Miller indices
    mesh_grid_x, mesh_grid_y, mesh_grid_z = mesh_dimensions

    # Generate Miller indices (h, k, l) for each FFT grid point
    # fftfreq gives frequencies normalized to sampling rate
    # Multiplying by n gives actual Miller indices
    kx = torch.fft.fftfreq(mesh_grid_x, d=1.0, device=device, dtype=dtype) * mesh_grid_x
    ky = torch.fft.fftfreq(mesh_grid_y, d=1.0, device=device, dtype=dtype) * mesh_grid_y
    kz = (
        torch.fft.rfftfreq(mesh_grid_z, d=1.0, device=device, dtype=dtype) * mesh_grid_z
    )

    kx_grid, ky_grid, kz_grid = torch.meshgrid(kx, ky, kz, indexing="ij")

    # Stack into Miller indices array (nx, ny, nz/2+1, 3)
    k_grid = torch.stack([kx_grid, ky_grid, kz_grid], dim=-1)

    # Transform Miller indices to Cartesian k-vectors
    # k_cart = [h, k, l] @ reciprocal_matrix^T
    # where reciprocal_matrix has reciprocal lattice vectors as rows
    k_vectors = torch.einsum("ijkd,bcd->bijkc", k_grid, reciprocal_cell).squeeze(0)

    # Compute k^2 for Green's function
    k_squared = torch.sum(k_vectors**2, dim=-1)

    # Avoid division by zero at k=0
    k_squared_safe = torch.where(
        k_squared > 1e-12, k_squared, torch.tensor(1e-12, device=device)
    )

    return k_vectors, k_squared_safe
