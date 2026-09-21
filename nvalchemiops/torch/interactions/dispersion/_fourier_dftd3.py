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

Periodic particle-mesh DFT-D3(BJ), with no pair cutoff on the dispersion sum. See
:mod:`nvalchemiops.interactions.dispersion._fourier_dftd3` for the method and pass structure.

This layer supplies the two Fourier transforms, which Warp cannot perform on a full mesh, and
drives the Warp launchers around them -- the same division as the electrostatics PME path,
rather than the all-Warp real-space
:func:`~nvalchemiops.torch.interactions.dispersion.dftd3`.

Units
-----
``positions``, ``cell``, ``rcov``, ``cutoff`` and ``mesh_spacing`` must share one length unit.
D3 parameters are conventionally atomic units, so a cutoff quoted in Angstrom must be
converted; ``cutoff`` has no default for that reason.
"""

from __future__ import annotations

from collections.abc import Sequence
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
    _check_mesh_supports_stencil,
    _resolve_mesh,
    check_neighbour_format,
    fd3_cn_chain,
    fd3_cn_chain_matrix,
    fd3_coefficients,
    fd3_coordination_numbers,
    fd3_coordination_numbers_matrix,
    fd3_gather_and_force,
    fd3_kspace,
    fd3_self_energy,
    fd3_spread,
    resolve_rank_slots,
)
from nvalchemiops.torch import torch_custom_op
from nvalchemiops.torch.autograd import warp_from_torch, warp_stream_from_torch
from nvalchemiops.torch.interactions.electrostatics.pme import (
    compute_bspline_moduli_1d,
)
from nvalchemiops.torch.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

__all__ = [
    "FourierD3Parameters",
    "FourierD3Setup",
    "fourier_dftd3",
]


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
    cn_ref : torch.Tensor, shape (n_species, n_ref)
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
    cn_ref: torch.Tensor
    v_q: torch.Tensor
    eigs: torch.Tensor
    species_map: torch.Tensor
    max_relative_error: float

    def __post_init__(self):
        """Validate shapes and device consistency."""
        tensors = {
            "rcov": self.rcov,
            "sqrt_q": self.sqrt_q,
            "cn_ref": self.cn_ref,
            "v_q": self.v_q,
            "eigs": self.eigs,
            "species_map": self.species_map,
        }
        for name, tensor in tensors.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)}.")
        # An integral eigs truncates the decomposition to whole numbers, and a floating
        # species_map indexes as something else again; both used to run and return a
        # plausible energy.
        for name in ("rcov", "sqrt_q", "cn_ref", "v_q", "eigs"):
            dtype = tensors[name].dtype
            if dtype not in (torch.float32, torch.float64):
                raise TypeError(f"{name} must be float32 or float64, got {dtype}.")
        if self.species_map.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                f"species_map must be int32 or int64, got {self.species_map.dtype}."
            )
        # The kernels read these as flat tables. A stray leading axis is not caught by the
        # pairwise shape checks below, and reaches the device as an out-of-bounds read.
        for name in ("rcov", "sqrt_q", "eigs", "species_map"):
            if tensors[name].ndim != 1:
                raise ValueError(
                    f"{name} must be 1D, got shape {tuple(tensors[name].shape)}."
                )
        devices = {tensor.device for tensor in tensors.values()}
        if len(devices) > 1:
            raise ValueError(
                f"All FourierD3Parameters tensors must share one device, got {devices}."
            )
        if self.cn_ref.ndim != 2:
            raise ValueError(
                f"cn_ref must be 2D, got shape {tuple(self.cn_ref.shape)}."
            )
        if self.v_q.ndim != 3:
            raise ValueError(f"v_q must be 3D, got shape {tuple(self.v_q.shape)}.")
        if self.v_q.shape[:2] != self.cn_ref.shape:
            raise ValueError(
                f"v_q and cn_ref disagree on (n_species, n_ref): "
                f"{tuple(self.v_q.shape[:2])} against {tuple(self.cn_ref.shape)}."
            )
        if self.eigs.shape[0] != self.v_q.shape[2]:
            raise ValueError(
                f"eigs has rank {self.eigs.shape[0]} but v_q has rank {self.v_q.shape[2]}."
            )
        if self.rcov.shape != self.species_map.shape:
            raise ValueError(
                f"rcov and species_map are both indexed by atomic number, so they must "
                f"have the same length, got {self.rcov.shape[0]} and "
                f"{self.species_map.shape[0]}."
            )
        if self.sqrt_q.shape[0] != self.cn_ref.shape[0]:
            raise ValueError(
                f"sqrt_q covers {self.sqrt_q.shape[0]} species but cn_ref covers "
                f"{self.cn_ref.shape[0]}."
            )

    @property
    def rank(self) -> int:
        """Number of retained rank slots, and so of mesh channels per species."""
        return int(self.eigs.shape[0])

    @property
    def n_species(self) -> int:
        """Number of species channels."""
        return int(self.cn_ref.shape[0])

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
            cn_ref=self.cn_ref.to(device=device, dtype=dtype),
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
            cn_ref=as_tensor(decomposition.cn_ref),
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


def _warp_view(tensor, dtype):
    """View a Torch tensor as a Warp array without copying.

    ``requires_grad=False`` is explicit: these kernels run with ``enable_backward=False``
    and forces are an output, so the lightweight ctype view is always the right one.
    """
    return warp_from_torch(tensor, dtype, requires_grad=False)


def _on_torch_stream(function):
    """Run a Warp-launching op on PyTorch's current CUDA stream.

    Warp otherwise launches on a stream of its own, which prevents ``torch.cuda.graph``
    capture -- what ``torch.compile(mode="reduce-overhead")`` uses -- and forces a
    cross-stream dependency on every call. ``sync_enter=False`` because an entry
    synchronisation is illegal mid-capture, and ordering is already guaranteed by both
    sides using the same stream.
    """

    @wraps(function)
    def wrapper(*args, **kwargs):
        with warp_stream_from_torch(*args, sync_enter=False):
            return function(*args, **kwargs)

    return wrapper


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
    covalent_radii: torch.Tensor,
    cnref: torch.Tensor,
    v_q: torch.Tensor,
    cutoff: float,
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
            _warp_view(positions.detach(), vec_dtype),
            _warp_view(numbers, wp.int32),
            _warp_view(neighbor_matrix, wp.int32),
            _warp_view(cartesian_shifts, vec_dtype),
            _warp_view(covalent_radii, wp_dtype),
            cutoff,
            _warp_view(coord_num, wp_dtype),
            wp_dtype,
            device,
            fill_value,
        )
    else:
        fd3_coordination_numbers(
            _warp_view(positions.detach(), vec_dtype),
            _warp_view(numbers, wp.int32),
            _warp_view(neighbor_list, wp.int32),
            _warp_view(neighbor_ptr, wp.int32),
            _warp_view(cartesian_shifts, vec_dtype),
            _warp_view(covalent_radii, wp_dtype),
            cutoff,
            _warp_view(coord_num, wp_dtype),
            wp_dtype,
            device,
        )

    fd3_coefficients(
        _warp_view(coord_num, wp_dtype),
        _warp_view(species_index, wp.int32),
        _warp_view(cnref, wp_dtype),
        _warp_view(v_q, wp_dtype),
        _warp_view(c6, wp_dtype),
        _warp_view(dc6_dcn, wp_dtype),
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
        _warp_view(positions.detach(), vec_dtype),
        _warp_view(c6, wp_dtype),
        _warp_view(group_idx, wp.int32),
        _warp_view(cell_inv_t, mat_dtype),
        spline_order,
        rank,
        _warp_view(mesh, wp_dtype),
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
        _warp_view(mesh_fft, pair_dtype),
        _warp_view(k_matrix, mat_dtype),
        _warp_view(moduli_x, wp_dtype),
        _warp_view(moduli_y, wp_dtype),
        _warp_view(moduli_z, wp_dtype),
        _warp_view(volumes, wp_dtype),
        _warp_view(sqrt_q, wp_dtype),
        _warp_view(eigs, wp_dtype),
        s6,
        s8,
        a1,
        a2,
        (mesh_nx, mesh_ny, mesh_nz),
        n_species,
        rank,
        _warp_view(energy, wp_dtype),
        _warp_view(cotangent, pair_dtype),
        _warp_view(virial, mat_dtype),
        wp_dtype,
        device,
        compute_virial,
    )


@torch_custom_op(
    "nvalchemiops::fourier_dftd3_gather",
    mutates_args=("d_energy_d_c6", "forces"),
)
@_on_torch_stream
def _fd3_gather_op(
    potential: torch.Tensor,
    positions: torch.Tensor,
    group_idx: torch.Tensor,
    cell_inv_t: torch.Tensor,
    c6: torch.Tensor,
    spline_order: int,
    rank: int,
    d_energy_d_c6: torch.Tensor,
    forces: torch.Tensor,
    device: str | None = None,
) -> None:
    """Internal op for the mesh gather: the coefficient derivative and the direct force.

    This is the only stage that touches the mesh, so it is the only one that runs per rank
    chunk. Both outputs are accumulated into rather than overwritten, because a chunk
    contributes a slice of ``d_energy_d_c6`` and a share of the force.
    """
    if positions.size(0) == 0:
        return
    if device is None:
        device = str(positions.device)
    wp_dtype, vec_dtype, mat_dtype = _dtypes(positions)

    fd3_gather_and_force(
        _warp_view(potential, wp_dtype),
        _warp_view(positions.detach(), vec_dtype),
        _warp_view(c6, wp_dtype),
        _warp_view(group_idx, wp.int32),
        _warp_view(cell_inv_t, mat_dtype),
        spline_order,
        rank,
        _warp_view(d_energy_d_c6, wp_dtype),
        _warp_view(forces, vec_dtype),
        wp_dtype,
        device,
    )


@torch_custom_op(
    "nvalchemiops::fourier_dftd3_finalise",
    mutates_args=("d_energy_d_c6", "d_energy_d_cn", "energy", "forces", "virial"),
)
@_on_torch_stream
def _fd3_finalise_op(
    positions: torch.Tensor,
    numbers: torch.Tensor,
    species_index: torch.Tensor,
    batch_idx: torch.Tensor,
    c6: torch.Tensor,
    dc6_dcn: torch.Tensor,
    cartesian_shifts: torch.Tensor,
    covalent_radii: torch.Tensor,
    sqrt_q: torch.Tensor,
    eigs: torch.Tensor,
    s6: float,
    s8: float,
    a1: float,
    a2: float,
    cutoff: float,
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
    """Internal op for the self-energy and the coordination chain rule.

    Neither stage touches the mesh and both need every rank slot at once, so this runs once
    after the rank chunks rather than inside them. The chain rule walks the whole neighbour
    list, which is why running it per chunk would be the expensive mistake.

    The self-energy is folded into ``d_energy_d_c6`` before the chain rule contracts it,
    because it is quadratic in the coefficients and so reaches the forces through the
    coordination numbers. Applying it afterwards as a scalar would drop that contribution.
    """
    d_energy_d_cn.zero_()
    if positions.size(0) == 0:
        return
    if device is None:
        device = str(positions.device)
    wp_dtype, vec_dtype, mat_dtype = _dtypes(positions)

    fd3_self_energy(
        _warp_view(c6, wp_dtype),
        _warp_view(species_index, wp.int32),
        _warp_view(batch_idx, wp.int32),
        _warp_view(sqrt_q, wp_dtype),
        _warp_view(eigs, wp_dtype),
        s6,
        s8,
        a1,
        a2,
        _warp_view(energy, wp_dtype),
        _warp_view(d_energy_d_c6, wp_dtype),
        wp_dtype,
        device,
    )

    common = (
        _warp_view(d_energy_d_c6, wp_dtype),
        _warp_view(dc6_dcn, wp_dtype),
        _warp_view(positions.detach(), vec_dtype),
        _warp_view(numbers, wp.int32),
    )
    tail = (
        _warp_view(covalent_radii, wp_dtype),
        cutoff,
        _warp_view(batch_idx, wp.int32),
        _warp_view(d_energy_d_cn, wp_dtype),
        _warp_view(forces, vec_dtype),
        _warp_view(virial, mat_dtype),
        wp_dtype,
        device,
        compute_virial,
    )
    if neighbor_matrix is not None:
        fd3_cn_chain_matrix(
            *common,
            _warp_view(neighbor_matrix, wp.int32),
            _warp_view(cartesian_shifts, vec_dtype),
            *tail,
            fill_value,
        )
    else:
        fd3_cn_chain(
            *common,
            _warp_view(neighbor_list, wp.int32),
            _warp_view(neighbor_ptr, wp.int32),
            _warp_view(cartesian_shifts, vec_dtype),
            *tail,
        )


@dataclass
class FourierD3Setup:
    """Cell- and mesh-derived quantities that do not change from step to step.

    An optional performance cache: it saves a matrix inversion and the spline moduli per call.
    ``fourier_dftd3`` works without it, including under ``torch.compile`` and CUDA graph
    capture.

    **Keeping it consistent with the cell is the caller's responsibility**, as with the PME
    and multipole caches. The derived quantities stand in for the cell everywhere except the
    Cartesian image shifts, which still come from the cell passed to the call, so a stale
    setup mixes two cells in one evaluation with nothing raised. Rebuild it whenever the cell
    changes. :meth:`validate_for` checks only metadata -- batch size, species count, dtype,
    device and mesh -- which is cheap and works under compilation; it does not compare cell
    values.

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

        Returns
        -------
        FourierD3Setup
        """
        # Detached up front: nothing derived below is ever differentiated -- the kernels run
        # with ``enable_backward=False`` and forces are explicit outputs -- so keeping the
        # graph would pin it for the lifetime of a setup that is deliberately long-lived.
        cells = cell.detach().reshape(-1, 3, 3)
        dtype, device = cells.dtype, cells.device
        _check_spline_order(spline_order, "FourierD3Setup.build")
        mesh_nx, mesh_ny, mesh_nz = _check_mesh_supports_stencil(
            tuple(int(n) for n in mesh_dimensions),
            spline_order,
            "FourierD3Setup.build",
        )
        # ``inv_ex`` over ``inv``: it reports singularity in a status code rather than
        # raising, so it does not read the result back and does not synchronise. Same
        # choice as the spline and PME paths.
        cell_inv = torch.linalg.inv_ex(cells)[0]
        cell_inv_t = cell_inv.transpose(-1, -2).contiguous()
        millers = (
            torch.fft.fftfreq(mesh_nx, d=1.0 / mesh_nx, dtype=dtype, device=device),
            torch.fft.fftfreq(mesh_ny, d=1.0 / mesh_ny, dtype=dtype, device=device),
            torch.fft.rfftfreq(mesh_nz, d=1.0 / mesh_nz, dtype=dtype, device=device),
        )
        moduli = [
            compute_bspline_moduli_1d(m, n, spline_order)
            for m, n in zip(millers, (mesh_nx, mesh_ny, mesh_nz), strict=True)
        ]
        return cls(
            cell_inv_grouped=cell_inv_t.repeat_interleave(
                n_species, dim=0
            ).contiguous(),
            volumes=torch.abs(torch.linalg.det(cells)).contiguous(),
            k_matrix=(2.0 * torch.pi * cell_inv).contiguous(),
            moduli_x=moduli[0],
            moduli_y=moduli[1],
            moduli_z=moduli[2],
            mesh_dimensions=(mesh_nx, mesh_ny, mesh_nz),
            spline_order=spline_order,
        )

    def validate_for(self, cells, n_species, mesh_dimensions, mesh_spacing=None):
        """Refuse to be used with inputs it was not built for.

        The derived quantities here stand in for the cell everywhere except the Cartesian
        image shifts, which are still taken from the cell passed to the call. A setup that
        does not match therefore does not produce a stale answer so much as an incoherent
        one, mixing two cells in a single evaluation.

        Metadata only -- batch size, species count, dtype, device and mesh. Cell *values* are
        not compared: that would read device memory, which graph capture forbids, and keeping
        the setup consistent with the cell is the caller's contract.
        """
        if self.volumes.shape[0] != cells.shape[0]:
            raise ValueError(
                f"setup was built for {self.volumes.shape[0]} system(s) but the call "
                f"passes {cells.shape[0]}. Rebuild it for this batch."
            )
        expected = cells.shape[0] * n_species
        if self.cell_inv_grouped.shape[0] != expected:
            raise ValueError(
                f"setup carries {self.cell_inv_grouped.shape[0]} mesh channels but this "
                f"call needs {expected} ({cells.shape[0]} system(s) x {n_species} "
                f"species). Rebuild it for these parameters."
            )
        if self.volumes.dtype != cells.dtype or self.volumes.device != cells.device:
            raise ValueError(
                f"setup is {self.volumes.dtype} on {self.volumes.device} but the call is "
                f"{cells.dtype} on {cells.device}. Rebuild it for this precision "
                f"and device."
            )
        if (
            mesh_dimensions is not None
            and tuple(mesh_dimensions) != self.mesh_dimensions
        ):
            raise ValueError(
                f"setup was built for mesh {self.mesh_dimensions} but the call asks for "
                f"{tuple(mesh_dimensions)}. Pass one or the other, not both."
            )
        if mesh_spacing is not None:
            raise ValueError(
                f"setup already fixes the mesh at {self.mesh_dimensions}, so "
                f"mesh_spacing={mesh_spacing} cannot apply. Resolving a spacing means "
                "reading the cell lengths off the device, which is what a precomputed "
                "setup exists to avoid. Either drop mesh_spacing, or resolve it to mesh "
                "dimensions yourself -- ceil(length / spacing) per axis, rounded up to a "
                "size whose only prime factors are 2, 3, 5 and 7 -- and pass those to "
                "FourierD3Setup.build, which takes mesh_dimensions only."
            )


def _check_spline_order(spline_order, origin):
    """Reject an interpolation order the kernels are not built for.

    Orders outside 2 to 6 are not merely inaccurate: order 1 returns an energy around 1e19,
    order 0 recurses until the stack gives out, and a negative order fails inside
    ``torch.arange``. A setup carries its order through to the call and overrides whatever
    was passed there, so both entry points have to apply this or the check is bypassable.
    """
    if spline_order < 2 or spline_order > 6:
        raise ValueError(
            f"spline_order must be between 2 and 6, got {spline_order} (from {origin})."
        )


def fourier_dftd3(
    positions: torch.Tensor,
    numbers: torch.Tensor,
    a1: float,
    a2: float,
    s8: float,
    *,
    fourier_d3_params: FourierD3Parameters,
    cell: torch.Tensor,
    cutoff: float,
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
    rank_chunk_size: int | None = None,
    setup: FourierD3Setup | None = None,
    device: str | None = None,
) -> tuple[torch.Tensor, ...]:
    r"""Evaluate the DFT-D3(BJ) dispersion correction by particle-mesh summation.

    The dispersion sum itself is untruncated. ``cutoff`` applies only to the
    coordination-number neighbour list, which a machine-learned force field already builds
    for its own descriptors.

    Parameters
    ----------
    positions : torch.Tensor, shape (N, 3)
        Atomic positions.
    numbers : torch.Tensor, shape (N,)
        Atomic numbers. Zero marks a padding atom.
    a1, a2, s8 : float
        Becke-Johnson damping parameters for the exchange-correlation functional in use.
    fourier_d3_params : FourierD3Parameters
        Separable coefficients covering every species present. Coverage is a caller
        precondition and is not checked; an uncovered element is dropped from the sum.
    cell : torch.Tensor, shape (3, 3), (1, 3, 3) or (B, 3, 3)
        Lattice vectors as rows. Required: FourierD3 is periodic.
    cutoff : float
        Coordination-number cutoff, in the same length unit as ``positions``. **No default**,
        because the DFT-D3 tables are conventionally atomic units and a value meant as 6
        Angstrom would silently act as 6 Bohr. This must equal the radius the neighbour list
        was built with: the counting function is constructed to reach zero exactly there, and
        a mismatch reintroduces the truncation discontinuity it exists to remove.
    mesh_dimensions : tuple[int, int, int], optional
        Mesh size. Exactly one of this and ``mesh_spacing`` must be given, unless a ``setup``
        is passed -- it already carries the mesh it was built for, and then both are
        optional.
    mesh_spacing : float, optional
        Target spacing, in the same unit as ``cell``. Sized from the largest cell in a batch,
        then rounded **up** to a size whose only prime factors are 2, 3, 5 and 7, so the
        transform stays on cuFFT's radix kernels rather than falling back to Bluestein's
        algorithm. The resulting mesh is therefore never coarser than the spacing asked for,
        and may be a little finer. An explicit ``mesh_dimensions`` is used exactly as given.
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
        non-conservative forces. This is not checked: detecting it costs a reduction over
        the whole list on every call, and the builders' default already satisfies it.
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
    setup : FourierD3Setup, optional
        Cell- and mesh-derived quantities from :meth:`FourierD3Setup.build`, reused across
        steps. A performance cache only: it saves a matrix inversion and a set of spline
        moduli per call, and the call works without it under eager, ``torch.compile`` and
        CUDA graph capture alike. Batch size, species count, precision, device and mesh are
        checked; **keeping it consistent with the cell is yours to do**, since comparing cell
        values would read device memory. It also supplies the mesh
        and the spline order, so ``mesh_dimensions`` and ``mesh_spacing`` may be omitted and
        the usual "exactly one of them" rule does not apply. A ``mesh_dimensions`` that
        disagrees with the setup raises, and so does any ``mesh_spacing``, rather than either
        being silently ignored. ``spline_order`` is the one exception: it has a default, so a
        value passed alongside a setup cannot be told apart from the default and the setup's
        own order is used.
    compute_virial : bool, default=False
        Whether to return the virial.
    num_systems : int, optional
        Number of systems, inferred from ``cell`` when omitted.
    rank_chunk_size : int, optional
        Number of rank slots to hold on the mesh at once. ``None``, the default, processes
        all ``rank`` slots in a single pass, which is fastest. The mesh and its transforms
        dominate the workspace and scale as ``num_systems * n_species * rank * nx * ny * nz``,
        so a system with many species or a large retained rank can exceed device memory on a
        fine mesh. Setting this to ``k`` caps the resident slots at ``k`` and reduces that
        allocation by roughly ``rank / k``, at the cost of one extra spread, forward and
        inverse transform, and gather per chunk. Only those reciprocal stages are chunked;
        the self-energy and the coordination chain rule need every slot at once and run once
        afterwards. The per-atom ``(N, rank)`` coefficient arrays stay at full width
        throughout. The result is unchanged to round-off: every reciprocal stage is a sum over
        slots with no coupling between them. Must be a host-side Python integer, since it
        determines the number of kernel launches.
    device : str, optional
        Warp device string. Inferred from ``positions`` when omitted.

    Returns
    -------
    energy : torch.Tensor, shape (num_systems,)
    forces : torch.Tensor, shape (N, 3)
    virial : torch.Tensor, shape (num_systems, 3, 3)
        Returned only when ``compute_virial`` is set. Follows the repository convention in
        ``docs/userguide/about/conventions.md``: the **negative** derivative of the energy
        with respect to the affine displacement, :math:`W = -\partial E/\partial u`, with
        deformation applied as :math:`R' = R(I + u)` and :math:`C' = C(I + u)`. The
        tensile-positive Cauchy stress is :math:`\sigma = -W/V`. Matches
        :func:`~nvalchemiops.torch.interactions.dispersion.dftd3`.

    Notes
    -----
    The returned ``energy`` is **not differentiable**. The kernels are launched with
    ``enable_backward=False`` and no ``torch.library.register_autograd`` rule is registered,
    so it comes back with ``requires_grad=False`` and ``grad_fn=None``, detached from
    ``positions``. ``torch.autograd.grad`` on it raises rather than reproducing ``forces``.
    Use the returned ``forces`` and ``virial``, which are analytic derivatives of the same
    energy. This matches :func:`~nvalchemiops.torch.interactions.dispersion.dftd3`, and it is
    what keeps the pipeline capturable into a CUDA graph.

    Energies are reduced with atomic adds, whose summation order varies between launches, so
    repeated identical calls can differ in the last bit.

    Examples
    --------
    >>> energy, forces = fourier_dftd3(
    ...     positions, numbers, a1=0.4289, a2=4.4407, s8=0.7875,
    ...     fourier_d3_params=params, cell=cell, cutoff=11.34,
    ...     mesh_dimensions=(32, 32, 32),
    ...     neighbor_list=pairs, neighbor_ptr=pointer, unit_shifts=shifts,
    ... )
    """
    check_neighbour_format(
        neighbor_matrix,
        neighbor_matrix_shifts,
        neighbor_list,
        neighbor_ptr,
        unit_shifts,
    )
    if cell is None:
        raise ValueError("cell is required: FourierD3 evaluates a periodic sum.")
    _check_spline_order(spline_order, "fourier_dftd3")

    positions = positions if positions.is_floating_point() else positions.double()
    cells = cell.reshape(-1, 3, 3).to(dtype=positions.dtype, device=positions.device)
    n_atoms = positions.size(0)
    if num_systems is None:
        num_systems = cells.size(0)
    elif num_systems != cells.size(0):
        raise ValueError(
            f"num_systems is {num_systems} but cell holds {cells.size(0)} system(s). "
            "The energy and virial are shaped from num_systems while the mesh is built "
            "from the cells, so a mismatch silently pads the result with zeros."
        )
    if batch_idx is None:
        batch_idx = torch.zeros(n_atoms, dtype=torch.int32, device=positions.device)
    elif batch_idx.numel() != n_atoms:
        raise ValueError(
            f"batch_idx has {batch_idx.numel()} entries but there are {n_atoms} atoms."
        )
    batch_idx = batch_idx.to(dtype=torch.int32)
    # Out of range, an atom lands on a mesh slab belonging to no system and its contribution
    # disappears without trace -- a whole-system batch_idx of 5 returns exactly zero energy.
    batch_guard = None
    if n_atoms > 0:
        if not torch.compiler.is_compiling() and not _capturing():
            lowest = int(batch_idx.min())
            highest = int(batch_idx.max())
            if lowest < 0 or highest >= num_systems:
                raise ValueError(
                    f"batch_idx values span [{lowest}, {highest}], outside "
                    f"[0, {num_systems}) for {num_systems} system(s)."
                )
        else:
            # Reading the bounds back would synchronise, which graph capture forbids, and
            # ``torch.compile(mode="reduce-overhead")`` is a documented path, so skipping
            # would leave this reachable in ordinary use. Handled on device instead, in two
            # parts. The clamp keeps the kernels in bounds: an index past the last system
            # otherwise indexes the grouped cell and the mesh out of range, which is an
            # illegal access rather than a wrong number. The guard then poisons the result,
            # so the clamp cannot quietly turn bad input into a plausible answer.
            in_range = (batch_idx >= 0) & (batch_idx < num_systems)
            batch_guard = torch.where(
                in_range.all(),
                torch.ones((), dtype=positions.dtype, device=positions.device),
                torch.full(
                    (), float("nan"), dtype=positions.dtype, device=positions.device
                ),
            )
            batch_idx = batch_idx.clamp(0, num_systems - 1)

    params = fourier_d3_params.to(device=positions.device, dtype=positions.dtype)
    # Species coverage is a caller precondition, as in ``dftd3``: every element present must
    # be in the decomposition. Checking it here would read device memory on every step.
    species_index = params.species_map[numbers.long()].to(torch.int32)

    if setup is not None:
        setup.validate_for(cells, params.n_species, mesh_dimensions, mesh_spacing)
        mesh_nx, mesh_ny, mesh_nz = setup.mesh_dimensions
        spline_order = setup.spline_order
    else:
        # Reading the lengths is framework-specific and synchronises, so it is done here
        # and only on the route that needs them.
        lengths = (
            None
            if mesh_spacing is None
            else torch.linalg.norm(cells, dim=-1).max(dim=0).values.tolist()
        )
        mesh_nx, mesh_ny, mesh_nz = _resolve_mesh(
            mesh_dimensions, mesh_spacing, lengths, spline_order
        )
    n_species, rank = params.n_species, params.rank

    if n_atoms == 0:
        # Nothing to spread, so mesh, transforms and reciprocal sum are all zero. Placed
        # after every argument check, not before: an empty batch must reject a bad mesh just
        # as a populated one does, or the mistake hides until a batch has atoms.
        empty = dict(dtype=positions.dtype, device=positions.device)
        energy = torch.zeros(num_systems, **empty)
        forces = torch.zeros(0, 3, **empty)
        if compute_virial:
            return energy, forces, torch.zeros(num_systems, 3, 3, **empty)
        return energy, forces

    # One mesh slab per (system, species, rank), so spread cost scales with rank, not slab
    # count. Padding keeps a negative group so every guard fires; folding the system index in
    # first would land padding on an earlier system's valid slab.
    group_idx = torch.where(
        species_index < 0,
        torch.full_like(species_index, -1, dtype=torch.long),
        batch_idx.long() * n_species + species_index.long(),
    ).to(torch.int32)
    if setup is None:
        setup = FourierD3Setup.build(
            cells, n_species, (mesh_nx, mesh_ny, mesh_nz), spline_order
        )
    cell_inv_grouped = setup.cell_inv_grouped

    if neighbor_matrix is not None:
        shifts = neighbor_matrix_shifts.to(positions.dtype)
        if cells.shape[0] == 1:
            cartesian_shifts = (shifts @ cells[0]).contiguous()
        else:
            # A row holds one atom's neighbours, so it shifts by that atom's own lattice.
            # One shared cell would misplace images for every system after the first.
            cartesian_shifts = (shifts @ cells[batch_idx.long()]).contiguous()
    else:
        shifts = unit_shifts.to(positions.dtype)
        if cells.shape[0] == 1:
            # One cell for every edge, so a single small matmul. The general path would
            # gather a 3x3 per edge -- tens of megabytes on a large list.
            cartesian_shifts = (shifts @ cells[0]).contiguous()
        else:
            edge_system = batch_idx[neighbor_list[0].long()].long()
            cartesian_shifts = (
                (shifts.unsqueeze(1) @ cells[edge_system]).squeeze(1).contiguous()
            )

    empty = dict(dtype=positions.dtype, device=positions.device)
    # These are cleared by the op that fills them, so zeroing here would write each buffer
    # twice. Anything *accumulated* across ops below keeps torch.zeros.
    coord_num = torch.empty(n_atoms, **empty)
    c6 = torch.empty(n_atoms, rank, **empty)
    dc6_dcn = torch.empty(n_atoms, rank, **empty)
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
        params.cn_ref,
        params.v_q,
        cutoff,
        coord_num,
        c6,
        dc6_dcn,
        idx_j,
        neighbor_ptr.to(torch.int32) if neighbor_ptr is not None else None,
        neighbor_matrix.to(torch.int32) if neighbor_matrix is not None else None,
        fill_value,
        device,
    )

    energy = torch.zeros(num_systems, **empty)
    virial = torch.zeros(num_systems, 3, 3, **empty)
    forces = torch.zeros(n_atoms, 3, **empty)

    moduli = (setup.moduli_x, setup.moduli_y, setup.moduli_z)
    volumes = setup.volumes
    k_matrix = setup.k_matrix

    # Slots do not couple, so a chunk can go through spread, transform, contraction and
    # gather alone and be added in. The mesh dominates allocation and scales with resident
    # slots: this trades extra passes for peak memory.
    # ``d_energy_d_c6`` stays full width -- the chain rule below needs every slot at once.
    # It is per-atom, not per-mesh-point, so it does not undermine the bound.
    d_energy_d_c6 = torch.zeros(n_atoms, rank, **empty)
    d_energy_d_cn = torch.empty(n_atoms, **empty)

    # A non-positive size yields no chunks at all, so the loop below never runs and the
    # result stays at its zero initialisation. Checked rather than left to ``range``, which
    # rejects a step of zero but silently produces nothing for a negative one.
    slots = resolve_rank_slots(rank_chunk_size, rank)
    single_pass = slots >= rank
    for slot_start in range(0, rank, slots):
        slot_count = min(slots, rank - slot_start)
        # The k-space op clears energy and virial rather than accumulating, so multiple
        # chunks reduce into scratch. The gather accumulates, so forces pass through.
        if single_pass:
            chunk_energy, chunk_virial = energy, virial
        else:
            chunk_energy = torch.empty(num_systems, **empty)
            chunk_virial = torch.empty(num_systems, 3, 3, **empty)

        c6_chunk = c6[:, slot_start : slot_start + slot_count].contiguous()
        eigs_chunk = params.eigs[slot_start : slot_start + slot_count].contiguous()

        mesh = torch.empty(
            num_systems * n_species * slot_count, mesh_nx, mesh_ny, mesh_nz, **empty
        )
        _fd3_spread_op(
            positions,
            c6_chunk,
            group_idx,
            cell_inv_grouped,
            spline_order,
            slot_count,
            mesh,
            device,
        )

        mesh_fft = torch.fft.rfftn(mesh, dim=(-3, -2, -1), norm="backward")
        mesh_fft_pairs = torch.view_as_real(mesh_fft.resolve_conj()).contiguous()
        del mesh, mesh_fft

        cotangent = torch.empty_like(mesh_fft_pairs)
        _fd3_kspace_op(
            mesh_fft_pairs,
            k_matrix,
            moduli[0],
            moduli[1],
            moduli[2],
            volumes,
            params.sqrt_q,
            eigs_chunk,
            s6,
            s8,
            a1,
            a2,
            mesh_nx,
            mesh_ny,
            mesh_nz,
            n_species,
            slot_count,
            chunk_energy,
            cotangent,
            chunk_virial,
            compute_virial,
            device,
        )
        del mesh_fft_pairs

        # The unnormalised inverse transform is the adjoint of the forward one, which is what
        # makes the gathered result the derivative of the energy rather than its inverse.
        potential = torch.fft.irfftn(
            torch.view_as_complex(cotangent),
            s=(mesh_nx, mesh_ny, mesh_nz),
            dim=(-3, -2, -1),
            norm="forward",
        ).contiguous()
        del cotangent

        # The only stage that reads the mesh, hence the only one in the loop. Its derivative
        # fills this chunk's columns; its force goes straight into the running total.
        d_energy_d_c6_chunk = torch.zeros(n_atoms, slot_count, **empty)
        _fd3_gather_op(
            potential,
            positions,
            group_idx,
            cell_inv_grouped,
            c6_chunk,
            spline_order,
            slot_count,
            d_energy_d_c6_chunk,
            forces,
            device,
        )
        del potential
        d_energy_d_c6[:, slot_start : slot_start + slot_count] = d_energy_d_c6_chunk

        if not single_pass:
            energy += chunk_energy
            virial += chunk_virial

    # Both need every slot at once and never touch the mesh, so they run once. The chain
    # rule walks the whole neighbour list -- per chunk, that is the expensive mistake.
    _fd3_finalise_op(
        positions,
        numbers.to(torch.int32),
        species_index,
        batch_idx,
        c6,
        dc6_dcn,
        cartesian_shifts,
        params.rcov,
        params.sqrt_q,
        params.eigs,
        s6,
        s8,
        a1,
        a2,
        cutoff,
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

    if batch_guard is not None:
        # 1.0 unless a traced batch_idx pointed outside [0, num_systems); see above.
        energy = energy * batch_guard
        forces = forces * batch_guard
        virial = virial * batch_guard
    if compute_virial:
        return energy, forces, virial
    return energy, forces
