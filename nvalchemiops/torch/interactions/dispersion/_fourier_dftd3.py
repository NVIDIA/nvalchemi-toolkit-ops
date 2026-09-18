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

r"""
PyTorch binding for FourierD3.

Evaluates the periodic DFT-D3(BJ) dispersion correction by particle-mesh summation, with no
real-space cutoff on the dispersion sum itself. See
:mod:`nvalchemiops.interactions.dispersion._fourier_dftd3` for the method and the pass
structure.

This layer supplies the two Fourier transforms, which Warp cannot perform on a full mesh, and
drives the Warp launchers around them. That division follows the electrostatics PME path in
this package rather than the real-space :func:`~nvalchemiops.torch.interactions.dispersion.dftd3`,
which never leaves Warp.

Units
-----
Every length must share one system: ``positions``, ``cell``, ``rcov``, ``r_cut`` and
``mesh_spacing``. The DFT-D3 reference parameters are conventionally atomic units, so a cutoff
quoted in Angstrom has to be converted before it is passed in. ``r_cut`` has no default for
that reason.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from functools import wraps

import numpy as np
import torch
import warp as wp

from nvalchemiops.interactions.dispersion._c6_decomposition import (
    C6Decomposition,
    decompose_c6_reference,
)
from nvalchemiops.interactions.dispersion._fourier_dftd3 import (
    fd3_cn_chain,
    fd3_cn_chain_matrix,
    fd3_coefficients,
    fd3_coordination_numbers,
    fd3_coordination_numbers_matrix,
    fd3_gather_and_force,
    fd3_kspace,
    fd3_self_energy,
    fd3_spread,
)
from nvalchemiops.torch import torch_custom_op
from nvalchemiops.torch.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

__all__ = [
    "FourierD3Parameters",
    "FourierD3Setup",
    "fourier_dftd3",
]

# Reference coordination numbers are stored per species with unused slots marked negative.
_UNUSED_REFERENCE = -1.0


@dataclass
class FourierD3Parameters:
    """Separable dispersion coefficients for the species in a system.

    Holds the low-rank factors of Grimme's reference tensor together with the element data
    the evaluation needs. Every field is a function of the reference tables, the species
    present and the requested tolerance.

    Notably absent are the damping parameters. They are call-time arguments to
    :func:`fourier_dftd3` and are not stored here, so a single instance is valid for every
    functional parametrisation and there is no way for a stored copy to disagree with the
    values actually used.

    Attributes
    ----------
    rcov : torch.Tensor, shape (max_z + 1,)
        Covalent radii indexed by atomic number.
    sqrt_q : torch.Tensor, shape (n_species,)
        Square root of the quadrupole-to-dipole ratio, per species channel.
    cnref : torch.Tensor, shape (n_species, n_ref)
        Reference coordination numbers, negative in unused slots.
    v_q : torch.Tensor, shape (n_species, n_ref, rank)
        Eigenvectors of the decomposed reference tensor.
    eigs : torch.Tensor, shape (rank,)
        Eigenvalues, which may be negative.
    species_map : torch.Tensor, shape (max_z + 1,), dtype=int32
        Atomic number to channel index; ``-1`` where a species is not covered.
    max_relative_error : float
        Largest relative error the truncated decomposition makes on the reference tensor.
    """

    rcov: torch.Tensor
    sqrt_q: torch.Tensor
    cnref: torch.Tensor
    v_q: torch.Tensor
    eigs: torch.Tensor
    species_map: torch.Tensor
    max_relative_error: float

    def __post_init__(self):
        """Validate shapes and device consistency."""
        tensors = {
            "rcov": self.rcov,
            "sqrt_q": self.sqrt_q,
            "cnref": self.cnref,
            "v_q": self.v_q,
            "eigs": self.eigs,
            "species_map": self.species_map,
        }
        for name, tensor in tensors.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)}.")
        devices = {tensor.device for tensor in tensors.values()}
        if len(devices) > 1:
            raise ValueError(
                f"All FourierD3Parameters tensors must share one device, got {devices}."
            )
        if self.cnref.ndim != 2:
            raise ValueError(f"cnref must be 2D, got shape {tuple(self.cnref.shape)}.")
        if self.v_q.ndim != 3:
            raise ValueError(f"v_q must be 3D, got shape {tuple(self.v_q.shape)}.")
        if self.v_q.shape[:2] != self.cnref.shape:
            raise ValueError(
                f"v_q and cnref disagree on (n_species, n_ref): "
                f"{tuple(self.v_q.shape[:2])} against {tuple(self.cnref.shape)}."
            )
        if self.eigs.shape[0] != self.v_q.shape[2]:
            raise ValueError(
                f"eigs has rank {self.eigs.shape[0]} but v_q has rank {self.v_q.shape[2]}."
            )
        if self.sqrt_q.shape[0] != self.cnref.shape[0]:
            raise ValueError(
                f"sqrt_q covers {self.sqrt_q.shape[0]} species but cnref covers "
                f"{self.cnref.shape[0]}."
            )

    @property
    def rank(self) -> int:
        """Number of retained rank slots, and so of mesh channels per species."""
        return int(self.eigs.shape[0])

    @property
    def n_species(self) -> int:
        """Number of species channels."""
        return int(self.cnref.shape[0])

    @property
    def device(self) -> torch.device:
        """Device the parameters live on."""
        return self.rcov.device

    def to(self, device=None, dtype=None) -> FourierD3Parameters:
        """Return a copy on the given device and floating dtype.

        ``species_map`` stays integral regardless of ``dtype``.
        """
        return FourierD3Parameters(
            rcov=self.rcov.to(device=device, dtype=dtype),
            sqrt_q=self.sqrt_q.to(device=device, dtype=dtype),
            cnref=self.cnref.to(device=device, dtype=dtype),
            v_q=self.v_q.to(device=device, dtype=dtype),
            eigs=self.eigs.to(device=device, dtype=dtype),
            species_map=self.species_map.to(device=device),
            max_relative_error=self.max_relative_error,
        )

    @classmethod
    def from_tables(
        cls,
        rcov: torch.Tensor,
        r4r2: torch.Tensor,
        c6ab: torch.Tensor,
        cn_ref: torch.Tensor,
        species: Sequence[int],
        tol: float = 1e-4,
        max_rank: int | None = None,
        device=None,
        dtype: torch.dtype = torch.float64,
    ) -> FourierD3Parameters:
        """Decompose Grimme's reference tables for a given set of species.

        Parameters
        ----------
        rcov, r4r2 : torch.Tensor, shape (max_z + 1,)
            Covalent radii and the quadrupole-to-dipole ratios, indexed by atomic number.
        c6ab, cn_ref : torch.Tensor, shape (max_z + 1, max_z + 1, n_ref, n_ref)
            Reference dispersion coefficients and coordination numbers.
        species : Sequence[int]
            Atomic numbers present. Order and duplicates are ignored.
        tol : float, default=1e-4
            Target maximum relative error of the reconstructed reference tensor.
        max_rank : int, optional
            Ceiling on the retained rank. When it prevents ``tol`` from being met the result
            is still returned and ``max_relative_error`` reports what was achieved.
        device : optional
            Device for the returned tensors. Defaults to that of ``rcov``.
        dtype : torch.dtype, default=torch.float64
            Floating dtype for the returned tensors.

        Returns
        -------
        FourierD3Parameters
        """
        decomposition = decompose_c6_reference(
            c6ab.detach().cpu().numpy().astype(np.float64),
            cn_ref.detach().cpu().numpy().astype(np.float64),
            species,
            tol=tol,
            max_rank=max_rank,
        )
        return cls._from_decomposition(
            decomposition, rcov, r4r2, device=device or rcov.device, dtype=dtype
        )

    @classmethod
    def _from_decomposition(
        cls,
        decomposition: C6Decomposition,
        rcov: torch.Tensor,
        r4r2: torch.Tensor,
        device,
        dtype: torch.dtype,
    ) -> FourierD3Parameters:
        """Wrap a host-side decomposition together with the element data."""

        def as_tensor(array):
            return torch.as_tensor(array, dtype=dtype, device=device)

        return cls(
            rcov=rcov.to(device=device, dtype=dtype),
            # Grimme stores the ratio already square-rooted, which is the form the damping
            # radius R0 = a1 * sqrt(3 * sqrt(Q_A Q_B)) + a2 consumes directly.
            sqrt_q=r4r2.to(device=device, dtype=dtype)[
                torch.as_tensor(
                    decomposition.species, dtype=torch.long, device=r4r2.device
                )
            ].to(device=device),
            cnref=as_tensor(decomposition.cnref),
            v_q=as_tensor(decomposition.v_q),
            eigs=as_tensor(decomposition.eigs),
            species_map=torch.as_tensor(
                decomposition.species_map, dtype=torch.int32, device=device
            ),
            max_relative_error=decomposition.max_relative_error,
        )


def _capturing() -> bool:
    """Whether a CUDA graph capture is in progress on the current stream."""
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _scoped_warp_stream(device):
    """Bind Warp launches to PyTorch's current CUDA stream, without synchronising.

    Warp otherwise launches on a stream of its own, which prevents ``torch.cuda.graph``
    capture and forces a cross-stream dependency on every call. ``wp.ScopedStream``
    synchronises on entry by default, which is itself illegal mid-capture, so that is
    disabled here; ordering is already guaranteed by both sides using the same stream.
    """
    if torch.device(device).type != "cuda":
        return nullcontext()
    torch_stream = torch.cuda.current_stream(device)
    if wp.get_stream(str(device)).cuda_stream == torch_stream.cuda_stream:
        return nullcontext()
    return wp.ScopedStream(wp.stream_from_torch(torch_stream), sync_enter=False)


def _on_torch_stream(function):
    """Run a Warp-launching op on PyTorch's current CUDA stream.

    Warp otherwise launches on a stream of its own. That prevents ``torch.cuda.graph``
    capture, which is what ``torch.compile(mode="reduce-overhead")`` uses, and forces a
    cross-stream dependency on every call.
    """

    @wraps(function)
    def wrapper(*args, **kwargs):
        device = next(
            argument.device for argument in args if isinstance(argument, torch.Tensor)
        )
        with _scoped_warp_stream(device):
            return function(*args, **kwargs)

    return wrapper


def _wp(tensor, dtype):
    """View a Torch tensor as a Warp array without copying."""
    return wp.from_torch(tensor, dtype=dtype, return_ctype=True)


def _dtypes(reference: torch.Tensor):
    """Warp scalar, vector and matrix dtypes matching a Torch tensor."""
    return (
        get_wp_dtype(reference.dtype),
        get_wp_vec_dtype(reference.dtype),
        get_wp_mat_dtype(reference.dtype),
    )


@torch_custom_op(
    "nvalchemiops::fourier_dftd3_prologue",
    mutates_args=("coord_num", "c6", "dc6_dcn"),
)
@_on_torch_stream
def _fd3_prologue_op(
    positions: torch.Tensor,
    numbers: torch.Tensor,
    species_index: torch.Tensor,
    cartesian_shifts: torch.Tensor,
    rcov: torch.Tensor,
    cnref: torch.Tensor,
    v_q: torch.Tensor,
    r_cut: float,
    coord_num: torch.Tensor,
    c6: torch.Tensor,
    dc6_dcn: torch.Tensor,
    neighbor_list: torch.Tensor | None = None,
    neighbor_ptr: torch.Tensor | None = None,
    neighbor_matrix: torch.Tensor | None = None,
    fill_value: int | None = None,
    device: str | None = None,
) -> None:
    """Internal op for the real-space passes: coordination numbers and coefficients."""
    coord_num.zero_()
    c6.zero_()
    dc6_dcn.zero_()
    if positions.size(0) == 0:
        return
    if device is None:
        device = str(positions.device)
    wp_dtype, vec_dtype, _ = _dtypes(positions)

    if neighbor_matrix is not None:
        fd3_coordination_numbers_matrix(
            _wp(positions.detach(), vec_dtype),
            _wp(numbers, wp.int32),
            _wp(neighbor_matrix, wp.int32),
            _wp(cartesian_shifts, vec_dtype),
            _wp(rcov, wp_dtype),
            r_cut,
            _wp(coord_num, wp_dtype),
            wp_dtype,
            device,
            fill_value,
        )
    else:
        fd3_coordination_numbers(
            _wp(positions.detach(), vec_dtype),
            _wp(numbers, wp.int32),
            _wp(neighbor_list, wp.int32),
            _wp(neighbor_ptr, wp.int32),
            _wp(cartesian_shifts, vec_dtype),
            _wp(rcov, wp_dtype),
            r_cut,
            _wp(coord_num, wp_dtype),
            wp_dtype,
            device,
        )

    fd3_coefficients(
        _wp(coord_num, wp_dtype),
        _wp(species_index, wp.int32),
        _wp(cnref, wp_dtype),
        _wp(v_q, wp_dtype),
        _wp(c6, wp_dtype),
        _wp(dc6_dcn, wp_dtype),
        wp_dtype,
        device,
    )


@torch_custom_op("nvalchemiops::fourier_dftd3_spread", mutates_args=("mesh",))
@_on_torch_stream
def _fd3_spread_op(
    positions: torch.Tensor,
    c6: torch.Tensor,
    group_idx: torch.Tensor,
    cell_inv_t: torch.Tensor,
    spline_order: int,
    rank: int,
    mesh: torch.Tensor,
    device: str | None = None,
) -> None:
    """Internal op spreading the separable coefficients onto the mesh."""
    mesh.zero_()
    if positions.size(0) == 0:
        return
    if device is None:
        device = str(positions.device)
    wp_dtype, vec_dtype, mat_dtype = _dtypes(positions)
    fd3_spread(
        _wp(positions.detach(), vec_dtype),
        _wp(c6, wp_dtype),
        _wp(group_idx, wp.int32),
        _wp(cell_inv_t, mat_dtype),
        spline_order,
        rank,
        _wp(mesh, wp_dtype),
        wp_dtype,
        device,
    )


@torch_custom_op(
    "nvalchemiops::fourier_dftd3_kspace",
    mutates_args=("energy", "cotangent", "virial"),
)
@_on_torch_stream
def _fd3_kspace_op(
    mesh_fft: torch.Tensor,
    k_matrix: torch.Tensor,
    moduli_x: torch.Tensor,
    moduli_y: torch.Tensor,
    moduli_z: torch.Tensor,
    volumes: torch.Tensor,
    sqrt_q: torch.Tensor,
    eigs: torch.Tensor,
    s6: float,
    s8: float,
    a1: float,
    a2: float,
    mesh_nx: int,
    mesh_ny: int,
    mesh_nz: int,
    n_species: int,
    rank: int,
    energy: torch.Tensor,
    cotangent: torch.Tensor,
    virial: torch.Tensor,
    compute_virial: bool = False,
    device: str | None = None,
) -> None:
    """Internal op contracting the transformed mesh against the dispersion kernel."""
    energy.zero_()
    cotangent.zero_()
    virial.zero_()
    if device is None:
        device = str(mesh_fft.device)
    wp_dtype, _, mat_dtype = _dtypes(volumes)
    pair_dtype = wp.vec2f if wp_dtype == wp.float32 else wp.vec2d
    fd3_kspace(
        _wp(mesh_fft, pair_dtype),
        _wp(k_matrix, mat_dtype),
        _wp(moduli_x, wp_dtype),
        _wp(moduli_y, wp_dtype),
        _wp(moduli_z, wp_dtype),
        _wp(volumes, wp_dtype),
        _wp(sqrt_q, wp_dtype),
        _wp(eigs, wp_dtype),
        s6,
        s8,
        a1,
        a2,
        (mesh_nx, mesh_ny, mesh_nz),
        n_species,
        rank,
        _wp(energy, wp_dtype),
        _wp(cotangent, pair_dtype),
        _wp(virial, mat_dtype),
        wp_dtype,
        device,
        compute_virial,
    )


@torch_custom_op(
    "nvalchemiops::fourier_dftd3_epilogue",
    mutates_args=("d_energy_d_c6", "d_energy_d_cn", "energy", "forces", "virial"),
)
@_on_torch_stream
def _fd3_epilogue_op(
    potential: torch.Tensor,
    positions: torch.Tensor,
    numbers: torch.Tensor,
    species_index: torch.Tensor,
    group_idx: torch.Tensor,
    batch_idx: torch.Tensor,
    cell_inv_t: torch.Tensor,
    c6: torch.Tensor,
    dc6_dcn: torch.Tensor,
    cartesian_shifts: torch.Tensor,
    rcov: torch.Tensor,
    sqrt_q: torch.Tensor,
    eigs: torch.Tensor,
    spline_order: int,
    rank: int,
    s6: float,
    s8: float,
    a1: float,
    a2: float,
    r_cut: float,
    d_energy_d_c6: torch.Tensor,
    d_energy_d_cn: torch.Tensor,
    energy: torch.Tensor,
    forces: torch.Tensor,
    virial: torch.Tensor,
    neighbor_list: torch.Tensor | None = None,
    neighbor_ptr: torch.Tensor | None = None,
    neighbor_matrix: torch.Tensor | None = None,
    fill_value: int | None = None,
    compute_virial: bool = False,
    device: str | None = None,
) -> None:
    """Internal op for gather, self-energy and the coordination chain rule.

    The self-energy is folded into ``d_energy_d_c6`` before the chain rule contracts it,
    because it is quadratic in the coefficients and so reaches the forces through the
    coordination numbers. Applying it afterwards as a scalar would drop that contribution.
    """
    d_energy_d_c6.zero_()
    d_energy_d_cn.zero_()
    forces.zero_()
    if positions.size(0) == 0:
        return
    if device is None:
        device = str(positions.device)
    wp_dtype, vec_dtype, mat_dtype = _dtypes(positions)

    fd3_gather_and_force(
        _wp(potential, wp_dtype),
        _wp(positions.detach(), vec_dtype),
        _wp(c6, wp_dtype),
        _wp(group_idx, wp.int32),
        _wp(cell_inv_t, mat_dtype),
        spline_order,
        rank,
        _wp(d_energy_d_c6, wp_dtype),
        _wp(forces, vec_dtype),
        wp_dtype,
        device,
    )

    fd3_self_energy(
        _wp(c6, wp_dtype),
        _wp(species_index, wp.int32),
        _wp(batch_idx, wp.int32),
        _wp(sqrt_q, wp_dtype),
        _wp(eigs, wp_dtype),
        s6,
        s8,
        a1,
        a2,
        _wp(energy, wp_dtype),
        _wp(d_energy_d_c6, wp_dtype),
        wp_dtype,
        device,
    )

    common = (
        _wp(d_energy_d_c6, wp_dtype),
        _wp(dc6_dcn, wp_dtype),
        _wp(positions.detach(), vec_dtype),
        _wp(numbers, wp.int32),
    )
    tail = (
        _wp(rcov, wp_dtype),
        r_cut,
        _wp(batch_idx, wp.int32),
        _wp(d_energy_d_cn, wp_dtype),
        _wp(forces, vec_dtype),
        _wp(virial, mat_dtype),
        wp_dtype,
        device,
        compute_virial,
    )
    if neighbor_matrix is not None:
        fd3_cn_chain_matrix(
            *common,
            _wp(neighbor_matrix, wp.int32),
            _wp(cartesian_shifts, vec_dtype),
            *tail,
            fill_value,
        )
    else:
        fd3_cn_chain(
            *common,
            _wp(neighbor_list, wp.int32),
            _wp(neighbor_ptr, wp.int32),
            _wp(cartesian_shifts, vec_dtype),
            *tail,
        )


def _bspline_moduli(miller, mesh_size, spline_order, exact, dtype, device):
    """B-spline attenuation for one mesh axis.

    Two conventions exist. ``sinc(m/N)**p`` is the continuous transform of the spline, which
    is what the electrostatics PME path in this package uses. The exact discrete alternative
    is the magnitude of the DFT of the spline coefficients, which is what the interpolation
    on a finite mesh actually applies; it differs at large ``m`` and matters more for forces
    than for energies, because the force involves the gradient of the interpolation.

    Parameters
    ----------
    miller : torch.Tensor
        Signed frequency indices for the axis.
    mesh_size : int
        Number of mesh points along the axis.
    spline_order : int
        B-spline order.
    exact : bool
        Whether to use the discrete form.

    Returns
    -------
    torch.Tensor
        Attenuation per frequency, same shape as ``miller``.
    """
    if not exact:
        return torch.special.sinc(miller / mesh_size) ** spline_order

    # Cardinal B-spline weights at integer offsets, zero-padded to the mesh length.
    coefficients = torch.zeros(mesh_size, dtype=dtype, device=device)
    nodes = torch.arange(spline_order, dtype=dtype, device=device)
    weights = _cardinal_bspline(nodes + 1.0, spline_order)
    coefficients[:spline_order] = weights
    modulus = torch.fft.fft(coefficients).abs()
    if mesh_size % 2 == 0:
        # The Nyquist bin can vanish, which would divide by zero during deconvolution.
        half = mesh_size // 2
        modulus[half] = 0.5 * (modulus[half - 1] + modulus[half + 1])
    index = miller.round().long() % mesh_size
    return modulus[index]


def _cardinal_bspline(u, order):
    """Cardinal B-spline of the given order, by the Cox-de Boor recursion.

    ``M_1`` is the indicator of ``[0, 1)`` and
    ``M_n(u) = (u * M_{n-1}(u) + (n - u) * M_{n-1}(u - 1)) / (n - 1)``.
    """
    if order == 1:
        return torch.where(
            (u >= 0.0) & (u < 1.0), torch.ones_like(u), torch.zeros_like(u)
        )
    lower = _cardinal_bspline(u, order - 1)
    shifted = _cardinal_bspline(u - 1.0, order - 1)
    return (u * lower + (float(order) - u) * shifted) / float(order - 1)


@dataclass
class FourierD3Setup:
    """Cell- and mesh-derived quantities that do not change from step to step.

    Building these costs a matrix inversion per system, which is negligible once but wasteful
    every step, and `torch.linalg.inv` cannot be recorded into a CUDA graph. Precomputing them
    is therefore both an optimisation and what makes
    ``torch.compile(mode="reduce-overhead")`` usable.

    Reuse is only valid while the cell and the mesh are unchanged. Under constant-volume
    dynamics that is the whole trajectory; under variable-cell dynamics it is one step, so
    rebuild or omit it there.

    Attributes
    ----------
    cell_inv_grouped : torch.Tensor, shape (B * n_species, 3, 3)
        Transpose of the inverse cell, repeat-interleaved across species channels.
    volumes : torch.Tensor, shape (B,)
        Cell volume per system.
    k_matrix : torch.Tensor, shape (B, 3, 3)
        ``2 * pi * inverse(cell)`` per system.
    moduli_x, moduli_y, moduli_z : torch.Tensor
        B-spline attenuation per mesh axis.
    mesh_dimensions : tuple[int, int, int]
        The mesh these were built for.
    spline_order : int
        The spline order these were built for.
    """

    cell_inv_grouped: torch.Tensor
    volumes: torch.Tensor
    k_matrix: torch.Tensor
    moduli_x: torch.Tensor
    moduli_y: torch.Tensor
    moduli_z: torch.Tensor
    mesh_dimensions: tuple[int, int, int]
    spline_order: int

    @classmethod
    def build(
        cls,
        cell: torch.Tensor,
        n_species: int,
        mesh_dimensions: tuple[int, int, int],
        spline_order: int = 4,
        exact_moduli: bool = True,
    ) -> FourierD3Setup:
        """Derive the reusable quantities from a cell and a mesh.

        Parameters
        ----------
        cell : torch.Tensor, shape (3, 3), (1, 3, 3) or (B, 3, 3)
            Lattice vectors as rows.
        n_species : int
            Number of species channels, from ``FourierD3Parameters.n_species``.
        mesh_dimensions : tuple[int, int, int]
            Mesh size.
        spline_order : int, default=4
            B-spline order.
        exact_moduli : bool, default=True
            Whether to use the discrete B-spline modulus.

        Returns
        -------
        FourierD3Setup
        """
        cells = cell.reshape(-1, 3, 3)
        dtype, device = cells.dtype, cells.device
        mesh_nx, mesh_ny, mesh_nz = (int(n) for n in mesh_dimensions)
        cell_inv_t = torch.linalg.inv(cells).transpose(-1, -2).contiguous()
        millers = (
            torch.fft.fftfreq(mesh_nx, d=1.0 / mesh_nx, dtype=dtype, device=device),
            torch.fft.fftfreq(mesh_ny, d=1.0 / mesh_ny, dtype=dtype, device=device),
            torch.fft.rfftfreq(mesh_nz, d=1.0 / mesh_nz, dtype=dtype, device=device),
        )
        moduli = [
            _bspline_moduli(m, n, spline_order, exact_moduli, dtype, device)
            for m, n in zip(millers, (mesh_nx, mesh_ny, mesh_nz), strict=True)
        ]
        return cls(
            cell_inv_grouped=cell_inv_t.repeat_interleave(
                n_species, dim=0
            ).contiguous(),
            volumes=torch.abs(torch.linalg.det(cells)).contiguous(),
            k_matrix=(2.0 * torch.pi * torch.linalg.inv(cells)).contiguous(),
            moduli_x=moduli[0],
            moduli_y=moduli[1],
            moduli_z=moduli[2],
            mesh_dimensions=(mesh_nx, mesh_ny, mesh_nz),
            spline_order=spline_order,
        )


def _resolve_mesh(mesh_dimensions, mesh_spacing, cells):
    """Settle the mesh size, requiring exactly one of the two ways of asking for it.

    Unlike PME there is no accuracy-based estimator to fall back on, so leaving both unset is
    an error rather than a guess. When a spacing is given the largest cell in the batch sets
    the size, since one mesh serves every system and sizing from the first would under-resolve
    the rest.
    """
    if (mesh_dimensions is None) == (mesh_spacing is None):
        raise ValueError(
            "Provide exactly one of mesh_dimensions or mesh_spacing. There is no "
            "accuracy-based default for FourierD3."
        )
    if mesh_dimensions is not None:
        if len(mesh_dimensions) != 3 or any(int(n) < 1 for n in mesh_dimensions):
            raise ValueError(
                f"mesh_dimensions must be three positive integers, got {mesh_dimensions}."
            )
        return tuple(int(n) for n in mesh_dimensions)
    if mesh_spacing <= 0.0:
        raise ValueError(f"mesh_spacing must be positive, got {mesh_spacing}.")
    lengths = torch.linalg.norm(cells, dim=-1).max(dim=0).values
    return tuple(
        max(1, int(torch.ceil(length / mesh_spacing).item())) for length in lengths
    )


def _validate_neighbours(
    neighbor_matrix, neighbor_matrix_shifts, neighbor_list, neighbor_ptr, unit_shifts
):
    """Check that exactly one neighbour format arrived, with its matching shifts."""
    matrix_given = neighbor_matrix is not None
    list_given = neighbor_list is not None
    if matrix_given and list_given:
        raise ValueError(
            "Cannot provide both neighbor_matrix and neighbor_list. "
            "Please provide only one neighbor representation format."
        )
    if not matrix_given and not list_given:
        raise ValueError("Must provide either neighbor_matrix or neighbor_list.")
    if matrix_given:
        if unit_shifts is not None:
            raise ValueError(
                "unit_shifts is for neighbor_list format. "
                "Use neighbor_matrix_shifts for neighbor_matrix format."
            )
        if neighbor_matrix_shifts is None:
            raise ValueError(
                "neighbor_matrix_shifts is required: FourierD3 is periodic, so every "
                "neighbour needs its lattice image."
            )
        return
    if neighbor_matrix_shifts is not None:
        raise ValueError(
            "neighbor_matrix_shifts is for neighbor_matrix format. "
            "Use unit_shifts for neighbor_list format."
        )
    if neighbor_ptr is None:
        raise ValueError("neighbor_ptr is required alongside neighbor_list.")
    if unit_shifts is None:
        raise ValueError(
            "unit_shifts is required: FourierD3 is periodic, so every neighbour needs "
            "its lattice image."
        )


def _pairing_residual(sources, targets, shifts):
    """How far a neighbour list is from holding both directions of every pair.

    FourierD3 accumulates each atom's coordination number, and the chain rule from it, out
    of that atom's own row only; the reverse edge is walked by the other atom's block. Both
    orientations therefore have to be present, which is what the neighbour builders produce
    unless asked for ``half_fill=True``.

    In a full directed list every edge is cancelled by its reverse, so ``source - target``
    and the image shifts each sum to exactly zero. A half-filled list generally breaks both.
    The residual is therefore sound as a rejection -- a valid list can never produce a
    non-zero one -- without being complete.
    """
    balance = (sources.to(torch.int64) - targets.to(torch.int64)).sum().abs()
    drift = shifts.to(torch.int64).flatten(end_dim=-2).sum(dim=0).abs().sum()
    return balance + drift


def fourier_dftd3(
    positions: torch.Tensor,
    numbers: torch.Tensor,
    a1: float,
    a2: float,
    s8: float,
    *,
    fd3_params: FourierD3Parameters,
    cell: torch.Tensor,
    r_cut: float,
    mesh_dimensions: tuple[int, int, int] | None = None,
    mesh_spacing: float | None = None,
    neighbor_matrix: torch.Tensor | None = None,
    neighbor_matrix_shifts: torch.Tensor | None = None,
    neighbor_list: torch.Tensor | None = None,
    neighbor_ptr: torch.Tensor | None = None,
    unit_shifts: torch.Tensor | None = None,
    fill_value: int | None = None,
    s6: float = 1.0,
    spline_order: int = 4,
    batch_idx: torch.Tensor | None = None,
    compute_virial: bool = False,
    num_systems: int | None = None,
    exact_moduli: bool = True,
    setup: FourierD3Setup | None = None,
    device: str | None = None,
) -> tuple[torch.Tensor, ...]:
    r"""Evaluate the DFT-D3(BJ) dispersion correction by particle-mesh summation.

    The dispersion sum itself carries no real-space cutoff. The only cutoff is ``r_cut``, on
    the coordination-number neighbour list, which a machine-learned force field already builds
    for its own descriptors.

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3)
        Atomic positions.
    numbers : torch.Tensor, shape (N,)
        Atomic numbers. Zero marks a padding atom.
    a1, a2, s8 : float
        Becke-Johnson damping parameters for the exchange-correlation functional in use.
    fd3_params : FourierD3Parameters
        Separable coefficients covering every species present.
    cell : torch.Tensor, shape (3, 3), (1, 3, 3) or (B, 3, 3)
        Lattice vectors as rows. Required: FourierD3 is periodic.
    r_cut : float
        Coordination-number cutoff, in the same length unit as ``positions``. **No default**,
        because the DFT-D3 tables are conventionally atomic units and a value meant as 6
        Angstrom would silently act as 6 Bohr. This must equal the radius the neighbour list
        was built with: the counting function is constructed to reach zero exactly there, and
        a mismatch reintroduces the truncation discontinuity it exists to remove.
    mesh_dimensions : tuple[int, int, int], optional
        Mesh size. Exactly one of this and ``mesh_spacing`` must be given.
    mesh_spacing : float, optional
        Target spacing, in the same unit as ``cell``. Sized from the largest cell in a batch.
        Reads cell lengths into Python integers, so pass explicit ``mesh_dimensions`` when
        tracing.
    neighbor_matrix, neighbor_matrix_shifts : torch.Tensor, optional
        Dense padded neighbour indices and their lattice images.
    neighbor_list, neighbor_ptr, unit_shifts : torch.Tensor, optional
        CSR neighbour list and its lattice images. Exactly one format must be supplied.

        Whichever format is used must hold **both directions of every pair**, which is what
        the neighbour builders produce by default. FourierD3 accumulates each atom's
        coordination number from its own row alone, so a list built with ``half_fill=True``
        loses half of every atom's coordination and yields wrong energies and
        non-conservative forces. Such a list is rejected rather than used.
    fill_value : int, optional
        Padding sentinel for the dense format. Defaults to the atom count.
    s6 : float, default=1.0
        Sixth-order scaling; unity for every common parametrisation.
    spline_order : int, default=4
        B-spline interpolation order, from 2 to 6. Accuracy at a fixed mesh improves with
        order: measured against a converged reference, orders 2 to 5 land roughly three
        orders of magnitude apart each way, so raising the order buys more than refining the
        mesh does. Order 3 is noticeably noisier than its neighbours; prefer an even order
        unless you have measured otherwise.
    batch_idx : torch.Tensor, shape (N,), optional
        System index per atom. Atoms must be grouped by system.
    compute_virial : bool, default=False
        Whether to return the virial.
    num_systems : int, optional
        Number of systems, inferred from ``cell`` when omitted.
    exact_moduli : bool, default=True
        Use the discrete B-spline modulus rather than ``sinc(m/N)**p``. The discrete form is
        what interpolation on a finite mesh actually applies; ``sinc(m/N)**p`` is its
        continuous approximation, which the electrostatics PME path in this package uses.
        Measured against an independent implementation of this method, the discrete form
        agrees to machine precision while the continuous one leaves a force discrepancy
        around 1e-5 at a 48-cubed mesh. Set to False only to reproduce the PME convention.
    device : str, optional
        Warp device string. Inferred from ``positions`` when omitted.

    Returns
    -------
    energy : torch.Tensor, shape (num_systems,)
    forces : torch.Tensor, shape (N, 3)
    virial : torch.Tensor, shape (num_systems, 3, 3)
        Returned only when ``compute_virial`` is set. This is ``dE/d(strain)``; divide by the
        cell volume and negate for the stress, matching the convention of
        :func:`~nvalchemiops.torch.interactions.dispersion.dftd3`.

    Notes
    -----
    Energies are reduced with atomic adds, whose summation order varies between launches, so
    repeated identical calls can differ in the last bit.

    Examples
    --------
    >>> energy, forces = fourier_dftd3(
    ...     positions, numbers, a1=0.4289, a2=4.4407, s8=0.7875,
    ...     fd3_params=params, cell=cell, r_cut=11.34,
    ...     mesh_dimensions=(32, 32, 32),
    ...     neighbor_list=pairs, neighbor_ptr=pointer, unit_shifts=shifts,
    ... )
    """
    _validate_neighbours(
        neighbor_matrix,
        neighbor_matrix_shifts,
        neighbor_list,
        neighbor_ptr,
        unit_shifts,
    )
    if cell is None:
        raise ValueError("cell is required: FourierD3 evaluates a periodic sum.")
    if spline_order < 2 or spline_order > 6:
        raise ValueError(f"spline_order must be between 2 and 6, got {spline_order}.")

    positions = positions if positions.is_floating_point() else positions.double()
    cells = cell.reshape(-1, 3, 3).to(dtype=positions.dtype, device=positions.device)
    n_atoms = positions.size(0)
    if num_systems is None:
        num_systems = cells.size(0)
    if batch_idx is None:
        batch_idx = torch.zeros(n_atoms, dtype=torch.int32, device=positions.device)
    batch_idx = batch_idx.to(dtype=torch.int32)

    params = fd3_params.to(device=positions.device, dtype=positions.dtype)
    species_index = params.species_map[numbers.long()].to(torch.int32)
    # Whether the parameters cover the system is a property of the setup, not of the step.
    # Reading the answer back forces a device synchronisation, which breaks a compile graph
    # and is illegal outright during CUDA graph capture, so the check is skipped in both.
    # Atomic number zero marks a padding atom, which the kernels are built to skip; only a
    # real element that the decomposition does not cover is an error.
    uncovered = (species_index < 0) & (numbers != 0)
    if not torch.compiler.is_compiling() and not _capturing() and bool(uncovered.any()):
        missing = torch.unique(numbers[uncovered]).tolist()
        raise ValueError(
            f"Atomic numbers {missing} are not covered by fd3_params. Rebuild the "
            f"decomposition with every species present in the system."
        )

    # A half-filled neighbour list gives silently wrong coordination numbers, and so wrong
    # energies and non-conservative forces. Reading the residual back synchronises, so this
    # is skipped while compiling and during graph capture, exactly as the species check is.
    if not torch.compiler.is_compiling() and not _capturing():
        if neighbor_matrix is not None:
            limit = n_atoms if fill_value is None else fill_value
            valid = neighbor_matrix < limit
            rows = torch.arange(
                neighbor_matrix.shape[0], device=neighbor_matrix.device
            ).unsqueeze(1)
            residual = _pairing_residual(
                rows.expand_as(neighbor_matrix)[valid],
                neighbor_matrix[valid],
                neighbor_matrix_shifts[valid],
            )
        else:
            residual = _pairing_residual(
                neighbor_list[0], neighbor_list[1], unit_shifts
            )
        if bool(residual != 0):
            raise ValueError(
                "The neighbour list does not hold both directions of every pair. "
                "FourierD3 builds each atom's coordination number from its own row, so a "
                "half-filled list omits contributions and yields wrong energies and "
                "non-conservative forces. Rebuild it with half_fill=False."
            )

    if setup is not None:
        mesh_nx, mesh_ny, mesh_nz = setup.mesh_dimensions
        spline_order = setup.spline_order
    else:
        mesh_nx, mesh_ny, mesh_nz = _resolve_mesh(mesh_dimensions, mesh_spacing, cells)
    n_species, rank = params.n_species, params.rank
    n_channels = n_species * rank

    # One mesh slab per (system, species, rank); the composite index routes each atom to its
    # own slab so the spread cost scales with the rank rather than the slab count.
    # Padding atoms keep a negative group so that every kernel's guard fires. Folding the
    # system index in first would make a padding atom in system 1 or later land on a valid
    # slab belonging to an earlier system, where the guard cannot see it.
    group_idx = torch.where(
        species_index < 0,
        torch.full_like(species_index, -1, dtype=torch.long),
        batch_idx.long() * n_species + species_index.long(),
    ).to(torch.int32)
    if setup is None:
        setup = FourierD3Setup.build(
            cells, n_species, (mesh_nx, mesh_ny, mesh_nz), spline_order, exact_moduli
        )
    cell_inv_grouped = setup.cell_inv_grouped

    if neighbor_matrix is not None:
        shifts = neighbor_matrix_shifts.to(positions.dtype)
        if cells.shape[0] == 1:
            cartesian_shifts = (shifts @ cells[0]).contiguous()
        else:
            # A row of the matrix holds one atom's neighbours, so the whole row shifts by
            # that atom's own lattice. Using a single cell here would place the periodic
            # images of every system after the first on the wrong lattice.
            cartesian_shifts = (shifts @ cells[batch_idx.long()]).contiguous()
    else:
        shifts = unit_shifts.to(positions.dtype)
        if cells.shape[0] == 1:
            # Every edge shares one cell, so this is a single small matmul. Taking the
            # general path here would gather a 3x3 cell per edge, which for a large
            # neighbour list is both a batched matrix-vector product and tens of megabytes
            # of materialised copies.
            cartesian_shifts = (shifts @ cells[0]).contiguous()
        else:
            edge_system = batch_idx[neighbor_list[0].long()].long()
            cartesian_shifts = (
                (shifts.unsqueeze(1) @ cells[edge_system]).squeeze(1).contiguous()
            )

    empty = dict(dtype=positions.dtype, device=positions.device)
    coord_num = torch.zeros(n_atoms, **empty)
    c6 = torch.zeros(n_atoms, rank, **empty)
    dc6_dcn = torch.zeros(n_atoms, rank, **empty)
    idx_j = (
        neighbor_list[1].contiguous().to(torch.int32)
        if neighbor_list is not None
        else None
    )

    _fd3_prologue_op(
        positions,
        numbers.to(torch.int32),
        species_index,
        cartesian_shifts,
        params.rcov,
        params.cnref,
        params.v_q,
        r_cut,
        coord_num,
        c6,
        dc6_dcn,
        idx_j,
        neighbor_ptr.to(torch.int32) if neighbor_ptr is not None else None,
        neighbor_matrix.to(torch.int32) if neighbor_matrix is not None else None,
        fill_value,
        device,
    )

    mesh = torch.zeros(num_systems * n_channels, mesh_nx, mesh_ny, mesh_nz, **empty)
    _fd3_spread_op(
        positions, c6, group_idx, cell_inv_grouped, spline_order, rank, mesh, device
    )

    mesh_fft = torch.fft.rfftn(mesh, dim=(-3, -2, -1), norm="backward")
    mesh_fft_pairs = torch.view_as_real(mesh_fft.resolve_conj()).contiguous()

    moduli = (setup.moduli_x, setup.moduli_y, setup.moduli_z)
    volumes = setup.volumes
    k_matrix = setup.k_matrix

    energy = torch.zeros(num_systems, **empty)
    virial = torch.zeros(num_systems, 3, 3, **empty)
    cotangent = torch.zeros_like(mesh_fft_pairs)
    _fd3_kspace_op(
        mesh_fft_pairs,
        k_matrix,
        moduli[0],
        moduli[1],
        moduli[2],
        volumes,
        params.sqrt_q,
        params.eigs,
        s6,
        s8,
        a1,
        a2,
        mesh_nx,
        mesh_ny,
        mesh_nz,
        n_species,
        rank,
        energy,
        cotangent,
        virial,
        compute_virial,
        device,
    )

    # The unnormalised inverse transform is the adjoint of the forward one, which is what
    # makes the gathered result the derivative of the energy rather than its inverse.
    potential = torch.fft.irfftn(
        torch.view_as_complex(cotangent),
        s=(mesh_nx, mesh_ny, mesh_nz),
        dim=(-3, -2, -1),
        norm="forward",
    ).contiguous()

    d_energy_d_c6 = torch.zeros(n_atoms, rank, **empty)
    d_energy_d_cn = torch.zeros(n_atoms, **empty)
    forces = torch.zeros(n_atoms, 3, **empty)
    _fd3_epilogue_op(
        potential,
        positions,
        numbers.to(torch.int32),
        species_index,
        group_idx,
        batch_idx,
        cell_inv_grouped,
        c6,
        dc6_dcn,
        cartesian_shifts,
        params.rcov,
        params.sqrt_q,
        params.eigs,
        spline_order,
        rank,
        s6,
        s8,
        a1,
        a2,
        r_cut,
        d_energy_d_c6,
        d_energy_d_cn,
        energy,
        forces,
        virial,
        idx_j,
        neighbor_ptr.to(torch.int32) if neighbor_ptr is not None else None,
        neighbor_matrix.to(torch.int32) if neighbor_matrix is not None else None,
        fill_value,
        compute_virial,
        device,
    )

    if compute_virial:
        return energy, forces, virial
    return energy, forces
