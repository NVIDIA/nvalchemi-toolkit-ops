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

from __future__ import annotations

from dataclasses import dataclass

import torch
import warp as wp

from nvalchemiops.interactions.dispersion._dftd3 import (
    dftd3 as wp_dftd3,
)
from nvalchemiops.interactions.dispersion._dftd3 import (
    dftd3_matrix as wp_dftd3_matrix,
)
from nvalchemiops.interactions.dispersion._dftd3 import (
    dftd3_matrix_pbc as wp_dftd3_matrix_pbc,
)
from nvalchemiops.interactions.dispersion._dftd3 import (
    dftd3_pbc as wp_dftd3_pbc,
)
from nvalchemiops.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype

__all__ = [
    "D3Parameters",
    "dftd3",
]


@dataclass
class D3Parameters:
    """
    DFT-D3 reference parameters for dispersion correction calculations.

    This dataclass encapsulates all element-specific parameters required for
    DFT-D3 dispersion corrections. The main purpose for this structure is to
    provide validation, ensuring the correct shapes, dtypes, and keys are
    present and complete. These parameters are used by :func:`dftd3`.

    Parameters
    ----------
    rcov : torch.Tensor
        Covalent radii [max_Z+1] as float32 or float64. Units should be consistent
        with position coordinates. Index 0 is reserved for
        padding; valid atomic numbers are 1 to max_Z.
    r4r2 : torch.Tensor
        <r⁴>/<r²> expectation values [max_Z+1] as float32 or float64.
        Dimensionless ratio used for computing C8 coefficients from C6 values.
    c6ab : torch.Tensor
        C6 reference coefficients [max_Z+1, max_Z+1, interp_mesh, interp_mesh]
        as float32 or float64. Units are energy x distance^6. Indexed by atomic numbers and coordination number reference indices.
    cn_ref : torch.Tensor
        Coordination number reference grid [max_Z+1, max_Z+1, interp_mesh, interp_mesh]
        as float32 or float64. Dimensionless CN values for Gaussian interpolation.
    interp_mesh : int, optional
        Size of the coordination number interpolation mesh. Default: 5
        (standard DFT-D3 uses a 5x5 grid)

    Raises
    ------
    ValueError
        If parameter shapes are inconsistent or invalid
    TypeError
        If parameters are not torch.Tensor or have invalid dtypes

    Notes
    -----
    - Parameters should use consistent units matching your coordinate system.
      Standard D3 parameters from the Grimme group use atomic units (Bohr for
      distances, Hartree x Bohr^6 for C6 coefficients).
    - Index 0 in all arrays is reserved for padding atoms (atomic number 0)
    - Valid atomic numbers range from 1 to max_z
    - The standard DFT-D3 implementation supports elements 1-94 (H to Pu)
    - Parameters can be float32 or float64; they will be converted to float32
      during computation for efficiency

    Examples
    --------
    Create parameters from individual tensors:

    >>> params = D3Parameters(
    ...     rcov=torch.rand(95),  # 94 elements + padding
    ...     r4r2=torch.rand(95),
    ...     c6ab=torch.rand(95, 95, 5, 5),
    ...     cn_ref=torch.rand(95, 95, 5, 5),
    ... )

    Create from a dictionary (e.g., loaded from file):

    >>> state_dict = torch.load("dftd3_parameters.pt")
    >>> params = D3Parameters(
    ...     rcov=state_dict["rcov"],
    ...     r4r2=state_dict["r4r2"],
    ...     c6ab=state_dict["c6ab"],
    ...     cn_ref=state_dict["cn_ref"],
    ... )
    """

    rcov: torch.Tensor
    r4r2: torch.Tensor
    c6ab: torch.Tensor
    cn_ref: torch.Tensor
    interp_mesh: int = 5

    def __post_init__(self) -> None:
        """Validate parameter shapes, dtypes, and physical constraints."""
        # Type validation
        for name, tensor in [
            ("rcov", self.rcov),
            ("r4r2", self.r4r2),
            ("c6ab", self.c6ab),
            ("cn_ref", self.cn_ref),
        ]:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"Parameter '{name}' must be a torch.Tensor, got {type(tensor)}"
                )
            if tensor.dtype not in (torch.float32, torch.float64):
                raise TypeError(
                    f"Parameter '{name}' must be float32 or float64, got {tensor.dtype}"
                )

        # Shape validation
        if self.rcov.ndim != 1:
            raise ValueError(
                f"rcov must be 1D tensor [max_Z+1], got shape {self.rcov.shape}"
            )

        max_z = self.rcov.size(0) - 1
        if max_z < 1:
            raise ValueError(
                f"rcov must have at least 2 elements (padding + 1 element), got {self.rcov.size(0)}"
            )

        if self.r4r2.shape != (max_z + 1,):
            raise ValueError(
                f"r4r2 must have shape [{max_z + 1}] to match rcov, got {self.r4r2.shape}"
            )

        expected_c6_shape = (max_z + 1, max_z + 1, self.interp_mesh, self.interp_mesh)
        if self.c6ab.shape != expected_c6_shape:
            raise ValueError(
                f"c6ab must have shape {expected_c6_shape}, got {self.c6ab.shape}"
            )

        expected_cn_shape = (max_z + 1, max_z + 1, self.interp_mesh, self.interp_mesh)
        if self.cn_ref.shape != expected_cn_shape:
            raise ValueError(
                f"cn_ref must have shape {expected_cn_shape}, got {self.cn_ref.shape}"
            )

        # Device consistency validation
        devices = [
            self.rcov.device,
            self.r4r2.device,
            self.c6ab.device,
            self.cn_ref.device,
        ]
        if len({str(d) for d in devices}) > 1:
            raise ValueError(
                f"All parameters must be on the same device. "
                f"Got devices: rcov={self.rcov.device}, r4r2={self.r4r2.device}, "
                f"c6ab={self.c6ab.device}, cn_ref={self.cn_ref.device}"
            )

    @property
    def max_z(self) -> int:
        """Maximum atomic number supported by these parameters."""
        return self.rcov.size(0) - 1

    @property
    def device(self) -> torch.device:
        """Device where parameters are stored."""
        return self.rcov.device

    def to(
        self,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> D3Parameters:
        """
        Move all parameters to the specified device and/or convert to specified dtype.

        Parameters
        ----------
        device : str or torch.device or None, optional
            Target device (e.g., 'cpu', 'cuda', 'cuda:0'). If None, keeps current device.
        dtype : torch.dtype or None, optional
            Target dtype (e.g., torch.float32, torch.float64). If None, keeps current dtype.

        Returns
        -------
        D3Parameters
            New instance with parameters on the target device and/or dtype

        Examples
        --------
        Move to GPU:

        >>> params_gpu = params.to(device='cuda')

        Convert to float32:

        >>> params_f32 = params.to(dtype=torch.float32)

        Move to GPU and convert to float32:

        >>> params_gpu_f32 = params.to(device='cuda', dtype=torch.float32)
        """
        return D3Parameters(
            rcov=self.rcov.to(device=device, dtype=dtype),
            r4r2=self.r4r2.to(device=device, dtype=dtype),
            c6ab=self.c6ab.to(device=device, dtype=dtype),
            cn_ref=self.cn_ref.to(device=device, dtype=dtype),
            interp_mesh=self.interp_mesh,
        )


# ==============================================================================
# PyTorch Wrapper
# ==============================================================================


@torch.library.custom_op(
    "nvalchemiops::dftd3_matrix",
    mutates_args=("energy", "forces", "coord_num", "virial"),
)
def _dftd3_matrix_op(
    positions: torch.Tensor,
    numbers: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    covalent_radii: torch.Tensor,
    r4r2: torch.Tensor,
    c6_reference: torch.Tensor,
    coord_num_ref: torch.Tensor,
    a1: float,
    a2: float,
    s8: float,
    energy: torch.Tensor,
    forces: torch.Tensor,
    coord_num: torch.Tensor,
    virial: torch.Tensor,
    k1: float = 16.0,
    k3: float = -4.0,
    s6: float = 1.0,
    s5_smoothing_on: float = 1e10,
    s5_smoothing_off: float = 1e10,
    fill_value: int | None = None,
    batch_idx: torch.Tensor | None = None,
    device: str | None = None,
) -> None:
    """Internal custom op for DFT-D3(BJ) dispersion energy and forces
    computation (non-PBC, neighbor matrix format).

    This is a low-level custom operator that performs DFT-D3(BJ) dispersion
    calculations using Warp kernels for non-periodic systems with neighbor matrix format.
    Output tensors must be pre-allocated by the caller and are modified in-place.
    For most use cases, prefer the higher-level :func:`dftd3` wrapper function
    instead of calling this method directly.

    This function is torch.compile compatible.

    Parameters
    ----------
    positions : torch.Tensor, shape (num_atoms, 3)
        Atomic coordinates as float32 or float64, in consistent distance units
        (conventionally Bohr)
    numbers : torch.Tensor, shape (num_atoms), dtype=int32
        Atomic numbers
    neighbor_matrix : torch.Tensor, shape (num_atoms, max_neighbors), dtype=int32
        Neighbor indices. See module docstring for format details.
        Padding entries have values >= fill_value.
    covalent_radii : torch.Tensor, shape (max_Z+1), dtype=float32
        Covalent radii indexed by atomic number, in same units as positions
    r4r2 : torch.Tensor, shape (max_Z+1), dtype=float32
        <r⁴>/<r²> expectation values for C8 computation (dimensionless)
    c6_reference : torch.Tensor, shape (max_Z+1, max_Z+1, 5, 5), dtype=float32
        C6 reference values in energy x distance^6 units
    coord_num_ref : torch.Tensor, shape (max_Z+1, max_Z+1, 5, 5), dtype=float32
        CN reference grid (dimensionless)
    a1 : float
        Becke-Johnson damping parameter 1 (functional-dependent, dimensionless)
    a2 : float
        Becke-Johnson damping parameter 2 (functional-dependent), in same units as positions
    s8 : float
        C8 term scaling factor (functional-dependent, dimensionless)
    energy : torch.Tensor, shape (num_systems,), dtype=float32
        OUTPUT: Total dispersion energy. Must be pre-allocated. Units are energy
        (Hartree when using standard D3 parameters).
    forces : torch.Tensor, shape (num_atoms, 3), dtype=float32
        OUTPUT: Atomic forces. Must be pre-allocated. Units are energy/distance
        (Hartree/Bohr when using standard D3 parameters).
    coord_num : torch.Tensor, shape (num_atoms,), dtype=float32
        OUTPUT: Coordination numbers (dimensionless). Must be pre-allocated.
    virial : torch.Tensor, shape (num_systems, 3, 3), dtype=float32
        OUTPUT: Virial tensor (remains zeros for non-PBC). Must be pre-allocated.
    k1 : float, optional
        CN counting function steepness parameter, in inverse distance units
        (typically 16.0 1/Bohr for atomic units)
    k3 : float, optional
        CN interpolation Gaussian width parameter (typically -4.0, dimensionless)
    s6 : float, optional
        C6 term scaling factor (typically 1.0, dimensionless)
    s5_smoothing_on : float, optional
        Distance where S5 switching begins, in same units as positions. Default: 1e10
    s5_smoothing_off : float, optional
        Distance where S5 switching completes, in same units as positions. Default: 1e10
    fill_value : int | None, optional
        Value indicating padding in neighbor_matrix. If None, defaults to num_atoms.
    batch_idx : torch.Tensor, shape (num_atoms,), dtype=int32, optional
        Batch indices. If None, all atoms are in a single system (batch 0).
    device : str, optional
        Warp device string (e.g., 'cuda:0', 'cpu'). If None, inferred from positions.

    Returns
    -------
    None

    Modifies input tensors in-place: energy, forces, coord_num, virial (remains zeros)

    Notes
    -----
    - All input tensors should use consistent units. Standard D3 parameters use
      atomic units (Bohr for distances, Hartree for energy).
    - Float32 or float64 precision for positions; outputs always float32
    - Padding atoms indicated by numbers[i] == 0
    - **Two-body only**: Computes pairwise C6 and C8 dispersion terms; three-body
      Axilrod-Teller-Muto (ATM/C9) terms are not included
    - For PBC calculations, use :func:`_dftd3_matrix_pbc_op` instead

    See Also
    --------
    :func:`dftd3` : Higher-level wrapper that handles allocation
    :func:`_dftd3_matrix_pbc_op` : PBC variant with neighbor matrix format
    """
    # Ensure all parameters are on correct device/dtype
    covalent_radii = covalent_radii.to(device=positions.device, dtype=torch.float32)
    r4r2 = r4r2.to(device=positions.device, dtype=torch.float32)
    c6_reference = c6_reference.to(device=positions.device, dtype=torch.float32)
    coord_num_ref = coord_num_ref.to(device=positions.device, dtype=torch.float32)

    # Get shapes
    num_atoms = positions.size(0)

    # Set fill_value if not provided
    if fill_value is None:
        fill_value = num_atoms

    # Handle empty case
    if num_atoms == 0:
        return

    # Infer device from positions if not provided
    if device is None:
        device = str(positions.device)

    # Zero output tensors
    energy.zero_()
    forces.zero_()
    coord_num.zero_()
    virial.zero_()

    # Detect dtype and set appropriate Warp types
    wp_dtype = get_wp_dtype(positions.dtype)
    vec_dtype = get_wp_vec_dtype(positions.dtype)

    # Create batch indices if not provided (single system)
    if batch_idx is None:
        batch_idx = torch.zeros(num_atoms, dtype=torch.int32, device=positions.device)

    # Convert PyTorch tensors to Warp arrays (detach positions)
    positions_wp = wp.from_torch(positions.detach(), dtype=vec_dtype, return_ctype=True)
    numbers_wp = wp.from_torch(numbers, dtype=wp.int32, return_ctype=True)
    neighbor_matrix_wp = wp.from_torch(
        neighbor_matrix, dtype=wp.int32, return_ctype=True
    )
    batch_idx_wp = wp.from_torch(batch_idx, dtype=wp.int32, return_ctype=True)

    # Convert parameter tensors to Warp arrays (ensure float32)
    covalent_radii_wp = wp.from_torch(
        covalent_radii.to(dtype=torch.float32, device=positions.device),
        dtype=wp.float32,
        return_ctype=True,
    )
    r4r2_wp = wp.from_torch(
        r4r2.to(dtype=torch.float32, device=positions.device),
        dtype=wp.float32,
        return_ctype=True,
    )
    c6_reference_wp = wp.from_torch(
        c6_reference.to(dtype=torch.float32, device=positions.device),
        dtype=wp.float32,
        return_ctype=True,
    )
    coord_num_ref_wp = wp.from_torch(
        coord_num_ref.to(dtype=torch.float32, device=positions.device),
        dtype=wp.float32,
        return_ctype=True,
    )

    # Convert pre-allocated output arrays to Warp
    coord_num_wp = wp.from_torch(coord_num, dtype=wp.float32, return_ctype=True)
    forces_wp = wp.from_torch(forces, dtype=wp.vec3f, return_ctype=True)
    energy_wp = wp.from_torch(energy, dtype=wp.float32, return_ctype=True)
    virial_wp = wp.from_torch(virial, dtype=wp.mat33f, return_ctype=True)

    # Call non-PBC warp launcher
    wp_dftd3_matrix(
        positions=positions_wp,
        numbers=numbers_wp,
        neighbor_matrix=neighbor_matrix_wp,
        covalent_radii=covalent_radii_wp,
        r4r2=r4r2_wp,
        c6_reference=c6_reference_wp,
        coord_num_ref=coord_num_ref_wp,
        a1=a1,
        a2=a2,
        s8=s8,
        coord_num=coord_num_wp,
        forces=forces_wp,
        energy=energy_wp,
        virial=virial_wp,
        wp_dtype=wp_dtype,
        device=device,
        k1=k1,
        k3=k3,
        s6=s6,
        s5_smoothing_on=s5_smoothing_on,
        s5_smoothing_off=s5_smoothing_off,
        fill_value=fill_value,
        batch_idx=batch_idx_wp,
    )


@torch.library.custom_op(
    "nvalchemiops::dftd3_matrix_pbc",
    mutates_args=("energy", "forces", "coord_num", "virial"),
)
def _dftd3_matrix_pbc_op(
    positions: torch.Tensor,
    numbers: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    cell: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    covalent_radii: torch.Tensor,
    r4r2: torch.Tensor,
    c6_reference: torch.Tensor,
    coord_num_ref: torch.Tensor,
    a1: float,
    a2: float,
    s8: float,
    energy: torch.Tensor,
    forces: torch.Tensor,
    coord_num: torch.Tensor,
    virial: torch.Tensor,
    k1: float = 16.0,
    k3: float = -4.0,
    s6: float = 1.0,
    s5_smoothing_on: float = 1e10,
    s5_smoothing_off: float = 1e10,
    fill_value: int | None = None,
    batch_idx: torch.Tensor | None = None,
    compute_virial: bool = False,
    device: str | None = None,
) -> None:
    """Internal custom op for DFT-D3(BJ) dispersion energy and forces computation (PBC, neighbor matrix format).

    This is a low-level custom operator that performs DFT-D3(BJ) dispersion
    calculations using Warp kernels for periodic systems with neighbor matrix format.
    Output tensors must be pre-allocated by the caller and are modified in-place.
    For most use cases, prefer the higher-level :func:`dftd3` wrapper function
    instead of calling this method directly.

    This function is torch.compile compatible.

    Parameters
    ----------
    positions : torch.Tensor, shape (num_atoms, 3)
        Atomic coordinates as float32 or float64, in consistent distance units
        (conventionally Bohr)
    numbers : torch.Tensor, shape (num_atoms), dtype=int32
        Atomic numbers
    neighbor_matrix : torch.Tensor, shape (num_atoms, max_neighbors), dtype=int32
        Neighbor indices. See module docstring for format details.
        Padding entries have values >= fill_value.
    cell : torch.Tensor, shape (num_systems, 3, 3), dtype=float32 or float64
        Unit cell lattice vectors for PBC, in same dtype and units as positions.
    neighbor_matrix_shifts : torch.Tensor, shape (num_atoms, max_neighbors, 3), dtype=int32
        Integer unit cell shifts for PBC.
    covalent_radii : torch.Tensor, shape (max_Z+1), dtype=float32
        Covalent radii indexed by atomic number, in same units as positions
    r4r2 : torch.Tensor, shape (max_Z+1), dtype=float32
        <r⁴>/<r²> expectation values for C8 computation (dimensionless)
    c6_reference : torch.Tensor, shape (max_Z+1, max_Z+1, 5, 5), dtype=float32
        C6 reference values in energy x distance^6 units
    coord_num_ref : torch.Tensor, shape (max_Z+1, max_Z+1, 5, 5), dtype=float32
        CN reference grid (dimensionless)
    a1 : float
        Becke-Johnson damping parameter 1 (functional-dependent, dimensionless)
    a2 : float
        Becke-Johnson damping parameter 2 (functional-dependent), in same units as positions
    s8 : float
        C8 term scaling factor (functional-dependent, dimensionless)
    energy : torch.Tensor, shape (num_systems,), dtype=float32
        OUTPUT: Total dispersion energy. Must be pre-allocated. Units are energy
        (Hartree when using standard D3 parameters).
    forces : torch.Tensor, shape (num_atoms, 3), dtype=float32
        OUTPUT: Atomic forces. Must be pre-allocated. Units are energy/distance
        (Hartree/Bohr when using standard D3 parameters).
    coord_num : torch.Tensor, shape (num_atoms,), dtype=float32
        OUTPUT: Coordination numbers (dimensionless). Must be pre-allocated.
    virial : torch.Tensor, shape (num_systems, 3, 3), dtype=float32
        OUTPUT: Virial tensor. Must be pre-allocated. Units are energy
        (Hartree when using standard D3 parameters).
    k1 : float, optional
        CN counting function steepness parameter, in inverse distance units
        (typically 16.0 1/Bohr for atomic units)
    k3 : float, optional
        CN interpolation Gaussian width parameter (typically -4.0, dimensionless)
    s6 : float, optional
        C6 term scaling factor (typically 1.0, dimensionless)
    s5_smoothing_on : float, optional
        Distance where S5 switching begins, in same units as positions. Default: 1e10
    s5_smoothing_off : float, optional
        Distance where S5 switching completes, in same units as positions. Default: 1e10
    fill_value : int | None, optional
        Value indicating padding in neighbor_matrix. If None, defaults to num_atoms.
    batch_idx : torch.Tensor, shape (num_atoms,), dtype=int32, optional
        Batch indices. If None, all atoms are in a single system (batch 0).
    compute_virial : bool, optional
        If True, compute virial tensor. Default: False
    device : str, optional
        Warp device string (e.g., 'cuda:0', 'cpu'). If None, inferred from positions.

    Returns
    -------
    None

    Modifies input tensors in-place: energy, forces, coord_num, virial (if compute_virial=True)

    Notes
    -----
    - All input tensors should use consistent units. Standard D3 parameters use
      atomic units (Bohr for distances, Hartree for energy).
    - Float32 or float64 precision for positions and cell; outputs always float32
    - Padding atoms indicated by numbers[i] == 0
    - **Two-body only**: Computes pairwise C6 and C8 dispersion terms; three-body
      Axilrod-Teller-Muto (ATM/C9) terms are not included
    - Bulk stress tensor can be obtained by dividing virial by system volume.
    - For non-PBC calculations, use :func:`_dftd3_matrix_op` instead

    See Also
    --------
    :func:`dftd3` : Higher-level wrapper that handles allocation
    :func:`_dftd3_matrix_op` : Non-PBC variant with neighbor matrix format
    """
    # Ensure all parameters are on correct device/dtype
    covalent_radii = covalent_radii.to(device=positions.device, dtype=torch.float32)
    r4r2 = r4r2.to(device=positions.device, dtype=torch.float32)
    c6_reference = c6_reference.to(device=positions.device, dtype=torch.float32)
    coord_num_ref = coord_num_ref.to(device=positions.device, dtype=torch.float32)

    # Get shapes
    num_atoms = positions.size(0)

    # Set fill_value if not provided
    if fill_value is None:
        fill_value = num_atoms

    # Handle empty case
    if num_atoms == 0:
        return

    # Infer device from positions if not provided
    if device is None:
        device = str(positions.device)

    # Zero output tensors
    energy.zero_()
    forces.zero_()
    coord_num.zero_()
    virial.zero_()

    # Detect dtype and set appropriate Warp types
    wp_dtype = get_wp_dtype(positions.dtype)
    vec_dtype = get_wp_vec_dtype(positions.dtype)
    mat_dtype = get_wp_mat_dtype(positions.dtype)

    # Create batch indices if not provided (single system)
    if batch_idx is None:
        batch_idx = torch.zeros(num_atoms, dtype=torch.int32, device=positions.device)

    # Convert PyTorch tensors to Warp arrays (detach positions)
    positions_wp = wp.from_torch(positions.detach(), dtype=vec_dtype, return_ctype=True)
    numbers_wp = wp.from_torch(numbers, dtype=wp.int32, return_ctype=True)
    neighbor_matrix_wp = wp.from_torch(
        neighbor_matrix, dtype=wp.int32, return_ctype=True
    )
    batch_idx_wp = wp.from_torch(batch_idx, dtype=wp.int32, return_ctype=True)

    # Convert parameter tensors to Warp arrays (ensure float32)
    covalent_radii_wp = wp.from_torch(
        covalent_radii.to(dtype=torch.float32, device=positions.device),
        dtype=wp.float32,
        return_ctype=True,
    )
    r4r2_wp = wp.from_torch(
        r4r2.to(dtype=torch.float32, device=positions.device),
        dtype=wp.float32,
        return_ctype=True,
    )
    c6_reference_wp = wp.from_torch(
        c6_reference.to(dtype=torch.float32, device=positions.device),
        dtype=wp.float32,
        return_ctype=True,
    )
    coord_num_ref_wp = wp.from_torch(
        coord_num_ref.to(dtype=torch.float32, device=positions.device),
        dtype=wp.float32,
        return_ctype=True,
    )

    # Convert cell and neighbor_matrix_shifts to warp for PBC
    cell_wp = wp.from_torch(
        cell.detach().to(dtype=positions.dtype, device=positions.device),
        dtype=mat_dtype,
        return_ctype=True,
    )
    neighbor_matrix_shifts_wp = wp.from_torch(
        neighbor_matrix_shifts.to(dtype=torch.int32, device=positions.device),
        dtype=wp.vec3i,
        return_ctype=True,
    )

    # Convert pre-allocated output arrays to Warp
    coord_num_wp = wp.from_torch(coord_num, dtype=wp.float32, return_ctype=True)
    forces_wp = wp.from_torch(forces, dtype=wp.vec3f, return_ctype=True)
    energy_wp = wp.from_torch(energy, dtype=wp.float32, return_ctype=True)
    virial_wp = wp.from_torch(virial, dtype=wp.mat33f, return_ctype=True)

    # Call PBC warp launcher
    wp_dftd3_matrix_pbc(
        positions=positions_wp,
        numbers=numbers_wp,
        neighbor_matrix=neighbor_matrix_wp,
        cell=cell_wp,
        neighbor_matrix_shifts=neighbor_matrix_shifts_wp,
        covalent_radii=covalent_radii_wp,
        r4r2=r4r2_wp,
        c6_reference=c6_reference_wp,
        coord_num_ref=coord_num_ref_wp,
        a1=a1,
        a2=a2,
        s8=s8,
        coord_num=coord_num_wp,
        forces=forces_wp,
        energy=energy_wp,
        virial=virial_wp,
        wp_dtype=wp_dtype,
        device=device,
        k1=k1,
        k3=k3,
        s6=s6,
        s5_smoothing_on=s5_smoothing_on,
        s5_smoothing_off=s5_smoothing_off,
        fill_value=fill_value,
        batch_idx=batch_idx_wp,
        compute_virial=compute_virial,
    )


@torch.library.custom_op(
    "nvalchemiops::dftd3",
    mutates_args=("energy", "forces", "coord_num", "virial"),
)
def _dftd3_op(
    positions: torch.Tensor,
    numbers: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    covalent_radii: torch.Tensor,
    r4r2: torch.Tensor,
    c6_reference: torch.Tensor,
    coord_num_ref: torch.Tensor,
    a1: float,
    a2: float,
    s8: float,
    energy: torch.Tensor,
    forces: torch.Tensor,
    coord_num: torch.Tensor,
    virial: torch.Tensor,
    k1: float = 16.0,
    k3: float = -4.0,
    s6: float = 1.0,
    s5_smoothing_on: float = 1e10,
    s5_smoothing_off: float = 1e10,
    batch_idx: torch.Tensor | None = None,
    device: str | None = None,
) -> None:
    """Internal custom op for DFT-D3(BJ) using CSR neighbor list format (non-PBC).

    This is a low-level custom operator that performs DFT-D3(BJ) dispersion
    calculations using CSR (Compressed Sparse Row) neighbor list format with
    idx_j (destination indices) and neighbor_ptr (row pointers) for non-periodic
    systems. Output tensors must be pre-allocated by the caller and are modified
    in-place. For most use cases, prefer the higher-level :func:`dftd3` wrapper
    function instead of calling this method directly.

    This function is torch.compile compatible.

    Parameters
    ----------
    positions : torch.Tensor, shape (num_atoms, 3)
        Atomic coordinates as float32 or float64
    numbers : torch.Tensor, shape (num_atoms), dtype=int32
        Atomic numbers
    idx_j : torch.Tensor, shape (num_edges,), dtype=int32
        Destination atom indices (flattened neighbor list in CSR format)
    neighbor_ptr : torch.Tensor, shape (num_atoms+1,), dtype=int32
        CSR row pointers where neighbor_ptr[i]:neighbor_ptr[i+1] gives neighbors of atom i
    covalent_radii : torch.Tensor, shape (max_Z+1), dtype=float32
        Covalent radii indexed by atomic number
    r4r2 : torch.Tensor, shape (max_Z+1), dtype=float32
        <r⁴>/<r²> expectation values
    c6_reference : torch.Tensor, shape (max_Z+1, max_Z+1, 5, 5), dtype=float32
        C6 reference values
    coord_num_ref : torch.Tensor, shape (max_Z+1, max_Z+1, 5, 5), dtype=float32
        CN reference grid
    a1 : float
        Becke-Johnson damping parameter 1
    a2 : float
        Becke-Johnson damping parameter 2
    s8 : float
        C8 term scaling factor
    energy : torch.Tensor, shape (num_systems,), dtype=float32
        OUTPUT: Total dispersion energy
    forces : torch.Tensor, shape (num_atoms, 3), dtype=float32
        OUTPUT: Atomic forces
    coord_num : torch.Tensor, shape (num_atoms,), dtype=float32
        OUTPUT: Coordination numbers
    virial : torch.Tensor, shape (num_systems, 3, 3), dtype=float32
        OUTPUT: Virial tensor (remains zeros for non-PBC). Must be pre-allocated.
    k1 : float, optional
        CN counting function steepness parameter
    k3 : float, optional
        CN interpolation Gaussian width parameter
    s6 : float, optional
        C6 term scaling factor
    s5_smoothing_on : float, optional
        Distance where S5 switching begins
    s5_smoothing_off : float, optional
        Distance where S5 switching completes
    batch_idx : torch.Tensor, shape (num_atoms,), dtype=int32, optional
        Batch indices
    device : str, optional
        Warp device string

    Returns
    -------
    None

    Modifies input tensors in-place: energy, forces, coord_num, virial (remains zeros)

    Notes
    -----
    - All input tensors should use consistent units. Standard D3 parameters use
      atomic units (Bohr for distances, Hartree for energy).
    - Float32 or float64 precision for positions; outputs always float32
    - Padding atoms indicated by numbers[i] == 0
    - **Two-body only**: Computes pairwise C6 and C8 dispersion terms; three-body
      Axilrod-Teller-Muto (ATM/C9) terms are not included
    - For PBC calculations, use :func:`_dftd3_pbc_op` instead

    See Also
    --------
    :func:`dftd3` : Higher-level wrapper that handles allocation
    :func:`_dftd3_pbc_op` : PBC variant with CSR neighbor list format

    """
    # Ensure all parameters are on correct device/dtype
    covalent_radii = covalent_radii.to(device=positions.device, dtype=torch.float32)
    r4r2 = r4r2.to(device=positions.device, dtype=torch.float32)
    c6_reference = c6_reference.to(device=positions.device, dtype=torch.float32)
    coord_num_ref = coord_num_ref.to(device=positions.device, dtype=torch.float32)

    # Get shapes
    num_atoms = positions.size(0)
    num_edges = idx_j.size(0)

    # Handle empty case
    if num_atoms == 0 or num_edges == 0:
        return

    # Infer device from positions if not provided
    if device is None:
        device = str(positions.device)

    # Zero output tensors
    energy.zero_()
    forces.zero_()
    coord_num.zero_()
    virial.zero_()

    # Detect dtype and set appropriate Warp types
    wp_dtype = get_wp_dtype(positions.dtype)
    vec_dtype = get_wp_vec_dtype(positions.dtype)

    # Create batch indices if not provided (single system)
    if batch_idx is None:
        batch_idx = torch.zeros(num_atoms, dtype=torch.int32, device=positions.device)

    # Convert PyTorch tensors to Warp arrays
    positions_wp = wp.from_torch(positions.detach(), dtype=vec_dtype, return_ctype=True)
    numbers_wp = wp.from_torch(numbers, dtype=wp.int32, return_ctype=True)
    idx_j_wp = wp.from_torch(idx_j, dtype=wp.int32, return_ctype=True)
    neighbor_ptr_wp = wp.from_torch(neighbor_ptr, dtype=wp.int32, return_ctype=True)
    batch_idx_wp = wp.from_torch(batch_idx, dtype=wp.int32, return_ctype=True)

    # Convert parameter tensors to Warp arrays
    covalent_radii_wp = wp.from_torch(
        covalent_radii.to(dtype=torch.float32, device=positions.device),
        dtype=wp.float32,
        return_ctype=True,
    )
    r4r2_wp = wp.from_torch(
        r4r2.to(dtype=torch.float32, device=positions.device),
        dtype=wp.float32,
        return_ctype=True,
    )
    c6_reference_wp = wp.from_torch(
        c6_reference.to(dtype=torch.float32, device=positions.device),
        dtype=wp.float32,
        return_ctype=True,
    )
    coord_num_ref_wp = wp.from_torch(
        coord_num_ref.to(dtype=torch.float32, device=positions.device),
        dtype=wp.float32,
        return_ctype=True,
    )

    # Convert pre-allocated output arrays to Warp
    coord_num_wp = wp.from_torch(coord_num, dtype=wp.float32, return_ctype=True)
    forces_wp = wp.from_torch(forces, dtype=wp.vec3f, return_ctype=True)
    energy_wp = wp.from_torch(energy, dtype=wp.float32, return_ctype=True)
    virial_wp = wp.from_torch(virial, dtype=wp.mat33f, return_ctype=True)

    # Call non-PBC warp launcher
    wp_dftd3(
        positions=positions_wp,
        numbers=numbers_wp,
        idx_j=idx_j_wp,
        neighbor_ptr=neighbor_ptr_wp,
        covalent_radii=covalent_radii_wp,
        r4r2=r4r2_wp,
        c6_reference=c6_reference_wp,
        coord_num_ref=coord_num_ref_wp,
        a1=a1,
        a2=a2,
        s8=s8,
        coord_num=coord_num_wp,
        forces=forces_wp,
        energy=energy_wp,
        virial=virial_wp,
        wp_dtype=wp_dtype,
        device=device,
        k1=k1,
        k3=k3,
        s6=s6,
        s5_smoothing_on=s5_smoothing_on,
        s5_smoothing_off=s5_smoothing_off,
        batch_idx=batch_idx_wp,
    )


@torch.library.custom_op(
    "nvalchemiops::dftd3_pbc",
    mutates_args=("energy", "forces", "coord_num", "virial"),
)
def _dftd3_pbc_op(
    positions: torch.Tensor,
    numbers: torch.Tensor,
    idx_j: torch.Tensor,
    neighbor_ptr: torch.Tensor,
    cell: torch.Tensor,
    unit_shifts: torch.Tensor,
    covalent_radii: torch.Tensor,
    r4r2: torch.Tensor,
    c6_reference: torch.Tensor,
    coord_num_ref: torch.Tensor,
    a1: float,
    a2: float,
    s8: float,
    energy: torch.Tensor,
    forces: torch.Tensor,
    coord_num: torch.Tensor,
    virial: torch.Tensor,
    k1: float = 16.0,
    k3: float = -4.0,
    s6: float = 1.0,
    s5_smoothing_on: float = 1e10,
    s5_smoothing_off: float = 1e10,
    batch_idx: torch.Tensor | None = None,
    compute_virial: bool = False,
    device: str | None = None,
) -> None:
    """Internal custom op for DFT-D3(BJ) using CSR neighbor list format (PBC).

    This is a low-level custom operator that performs DFT-D3(BJ) dispersion
    calculations using CSR (Compressed Sparse Row) neighbor list format with
    idx_j (destination indices) and neighbor_ptr (row pointers) for periodic
    systems. Output tensors must be pre-allocated by the caller and are modified
    in-place. For most use cases, prefer the higher-level :func:`dftd3` wrapper
    function instead of calling this method directly.

    This function is torch.compile compatible.

    Parameters
    ----------
    positions : torch.Tensor, shape (num_atoms, 3)
        Atomic coordinates as float32 or float64
    numbers : torch.Tensor, shape (num_atoms), dtype=int32
        Atomic numbers
    idx_j : torch.Tensor, shape (num_edges,), dtype=int32
        Destination atom indices (flattened neighbor list in CSR format)
    neighbor_ptr : torch.Tensor, shape (num_atoms+1,), dtype=int32
        CSR row pointers where neighbor_ptr[i]:neighbor_ptr[i+1] gives neighbors of atom i
    cell : torch.Tensor, shape (num_systems, 3, 3), dtype=float32 or float64
        Unit cell lattice vectors for PBC, in same dtype and units as positions.
    unit_shifts : torch.Tensor, shape (num_edges, 3), dtype=int32
        Integer unit cell shifts for PBC
    covalent_radii : torch.Tensor, shape (max_Z+1), dtype=float32
        Covalent radii indexed by atomic number
    r4r2 : torch.Tensor, shape (max_Z+1), dtype=float32
        <r⁴>/<r²> expectation values
    c6_reference : torch.Tensor, shape (max_Z+1, max_Z+1, 5, 5), dtype=float32
        C6 reference values
    coord_num_ref : torch.Tensor, shape (max_Z+1, max_Z+1, 5, 5), dtype=float32
        CN reference grid
    a1 : float
        Becke-Johnson damping parameter 1
    a2 : float
        Becke-Johnson damping parameter 2
    s8 : float
        C8 term scaling factor
    energy : torch.Tensor, shape (num_systems,), dtype=float32
        OUTPUT: Total dispersion energy
    forces : torch.Tensor, shape (num_atoms, 3), dtype=float32
        OUTPUT: Atomic forces
    coord_num : torch.Tensor, shape (num_atoms,), dtype=float32
        OUTPUT: Coordination numbers
    virial : torch.Tensor, shape (num_systems, 3, 3), dtype=float32
        OUTPUT: Virial tensor. Must be pre-allocated. Units are energy
        (Hartree when using standard D3 parameters).
    k1 : float, optional
        CN counting function steepness parameter
    k3 : float, optional
        CN interpolation Gaussian width parameter
    s6 : float, optional
        C6 term scaling factor
    s5_smoothing_on : float, optional
        Distance where S5 switching begins
    s5_smoothing_off : float, optional
        Distance where S5 switching completes
    batch_idx : torch.Tensor, shape (num_atoms,), dtype=int32, optional
        Batch indices
    compute_virial : bool, optional
        If True, compute virial tensor. Default: False
    device : str, optional
        Warp device string

    Returns
    -------
    None

    Modifies input tensors in-place: energy, forces, coord_num, virial (if compute_virial=True)

    Notes
    -----
    - All input tensors should use consistent units. Standard D3 parameters use
      atomic units (Bohr for distances, Hartree for energy).
    - Float32 or float64 precision for positions and cell; outputs always float32
    - Padding atoms indicated by numbers[i] == 0
    - **Two-body only**: Computes pairwise C6 and C8 dispersion terms; three-body
      Axilrod-Teller-Muto (ATM/C9) terms are not included
    - Bulk stress tensor can be obtained by dividing virial by system volume.
    - For non-PBC calculations, use :func:`_dftd3_op` instead

    See Also
    --------
    :func:`dftd3` : Higher-level wrapper that handles allocation
    :func:`_dftd3_op` : Non-PBC variant with CSR neighbor list format

    """
    # Ensure all parameters are on correct device/dtype
    covalent_radii = covalent_radii.to(device=positions.device, dtype=torch.float32)
    r4r2 = r4r2.to(device=positions.device, dtype=torch.float32)
    c6_reference = c6_reference.to(device=positions.device, dtype=torch.float32)
    coord_num_ref = coord_num_ref.to(device=positions.device, dtype=torch.float32)

    # Get shapes
    num_atoms = positions.size(0)
    num_edges = idx_j.size(0)

    # Handle empty case
    if num_atoms == 0 or num_edges == 0:
        return

    # Infer device from positions if not provided
    if device is None:
        device = str(positions.device)

    # Zero output tensors
    energy.zero_()
    forces.zero_()
    coord_num.zero_()
    virial.zero_()

    # Detect dtype and set appropriate Warp types
    wp_dtype = get_wp_dtype(positions.dtype)
    vec_dtype = get_wp_vec_dtype(positions.dtype)
    mat_dtype = get_wp_mat_dtype(positions.dtype)

    # Create batch indices if not provided (single system)
    if batch_idx is None:
        batch_idx = torch.zeros(num_atoms, dtype=torch.int32, device=positions.device)

    # Convert PyTorch tensors to Warp arrays
    positions_wp = wp.from_torch(positions.detach(), dtype=vec_dtype, return_ctype=True)
    numbers_wp = wp.from_torch(numbers, dtype=wp.int32, return_ctype=True)
    idx_j_wp = wp.from_torch(idx_j, dtype=wp.int32, return_ctype=True)
    neighbor_ptr_wp = wp.from_torch(neighbor_ptr, dtype=wp.int32, return_ctype=True)
    batch_idx_wp = wp.from_torch(batch_idx, dtype=wp.int32, return_ctype=True)

    # Convert parameter tensors to Warp arrays
    covalent_radii_wp = wp.from_torch(
        covalent_radii.to(dtype=torch.float32, device=positions.device),
        dtype=wp.float32,
        return_ctype=True,
    )
    r4r2_wp = wp.from_torch(
        r4r2.to(dtype=torch.float32, device=positions.device),
        dtype=wp.float32,
        return_ctype=True,
    )
    c6_reference_wp = wp.from_torch(
        c6_reference.to(dtype=torch.float32, device=positions.device),
        dtype=wp.float32,
        return_ctype=True,
    )
    coord_num_ref_wp = wp.from_torch(
        coord_num_ref.to(dtype=torch.float32, device=positions.device),
        dtype=wp.float32,
        return_ctype=True,
    )

    # Convert cell and unit_shifts to warp for PBC
    cell_wp = wp.from_torch(
        cell.detach().to(dtype=positions.dtype, device=positions.device),
        dtype=mat_dtype,
        return_ctype=True,
    )
    unit_shifts_wp = wp.from_torch(
        unit_shifts.to(dtype=torch.int32, device=positions.device),
        dtype=wp.vec3i,
        return_ctype=True,
    )

    # Convert pre-allocated output arrays to Warp
    coord_num_wp = wp.from_torch(coord_num, dtype=wp.float32, return_ctype=True)
    forces_wp = wp.from_torch(forces, dtype=wp.vec3f, return_ctype=True)
    energy_wp = wp.from_torch(energy, dtype=wp.float32, return_ctype=True)
    virial_wp = wp.from_torch(virial, dtype=wp.mat33f, return_ctype=True)

    # Call PBC warp launcher
    wp_dftd3_pbc(
        positions=positions_wp,
        numbers=numbers_wp,
        idx_j=idx_j_wp,
        neighbor_ptr=neighbor_ptr_wp,
        cell=cell_wp,
        unit_shifts=unit_shifts_wp,
        covalent_radii=covalent_radii_wp,
        r4r2=r4r2_wp,
        c6_reference=c6_reference_wp,
        coord_num_ref=coord_num_ref_wp,
        a1=a1,
        a2=a2,
        s8=s8,
        coord_num=coord_num_wp,
        forces=forces_wp,
        energy=energy_wp,
        virial=virial_wp,
        wp_dtype=wp_dtype,
        device=device,
        k1=k1,
        k3=k3,
        s6=s6,
        s5_smoothing_on=s5_smoothing_on,
        s5_smoothing_off=s5_smoothing_off,
        batch_idx=batch_idx_wp,
        compute_virial=compute_virial,
    )


def dftd3(
    positions: torch.Tensor,
    numbers: torch.Tensor,
    a1: float,
    a2: float,
    s8: float,
    k1: float = 16.0,
    k3: float = -4.0,
    s6: float = 1.0,
    s5_smoothing_on: float = 1e10,
    s5_smoothing_off: float = 1e10,
    fill_value: int | None = None,
    d3_params: D3Parameters | dict[str, torch.Tensor] | None = None,
    covalent_radii: torch.Tensor | None = None,
    r4r2: torch.Tensor | None = None,
    c6_reference: torch.Tensor | None = None,
    coord_num_ref: torch.Tensor | None = None,
    batch_idx: torch.Tensor | None = None,
    cell: torch.Tensor | None = None,
    neighbor_matrix: torch.Tensor | None = None,
    neighbor_matrix_shifts: torch.Tensor | None = None,
    neighbor_list: torch.Tensor | None = None,
    neighbor_ptr: torch.Tensor | None = None,
    unit_shifts: torch.Tensor | None = None,
    compute_virial: bool = False,
    num_systems: int | None = None,
    device: str | None = None,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
):
    """
    Compute DFT-D3(BJ) dispersion energy and forces using Warp
    with optional periodic boundary condition support and smoothing function.

    **DFT-D3 parameters must be explicitly provided** using one of three methods:

    1. **D3Parameters dataclass**: Supply a :class:`D3Parameters` instance (recommended).
       Individual parameters can override dataclass values if both are provided.

    2. **Explicit parameters**: Supply all four parameters individually:
       ``covalent_radii``, ``r4r2``, ``c6_reference``, and ``coord_num_ref``.

    3. **Dictionary**: Provide a ``d3_params`` dictionary with keys:
       ``"rcov"``, ``"r4r2"``, ``"c6ab"``, and ``"cn_ref"``.
       Individual parameters can override dictionary values if both are provided.

    See ``examples/interactions/utils.py`` for parameter generation utilities.

    This wrapper can be launched by either supplying a neighbor matrix or a
    neighbor list, both of which can be generated by the :func:`nvalchemiops.neighborlist.neighbor_list` function where the latter can be returned by setting the `return_neighbor_list` parameter to True.

    Parameters
    ----------
    positions : torch.Tensor
        Atomic coordinates [num_atoms, 3] as float32 or float64, in consistent distance
        units (conventionally Bohr when using standard D3 parameters)
    numbers : torch.Tensor
        Atomic numbers [num_atoms] as int32
    a1 : float
        Becke-Johnson damping parameter 1 (functional-dependent, dimensionless)
    a2 : float
        Becke-Johnson damping parameter 2 (functional-dependent), in same units as positions
    s8 : float
        C8 term scaling factor (functional-dependent, dimensionless)
    k1 : float, optional
        CN counting function steepness parameter, in inverse distance units
        (typically 16.0 1/Bohr for atomic units)
    k3 : float, optional
        CN interpolation Gaussian width parameter (typically -4.0, dimensionless)
    s6 : float, optional
        C6 term scaling factor (typically 1.0, dimensionless)
    s5_smoothing_on : float, optional
        Distance where S5 switching begins, in same units as positions. Set greater or
        equal to s5_smoothing_off to disable smoothing. Default: 1e10
    s5_smoothing_off : float, optional
        Distance where S5 switching completes, in same units as positions.
        Default: 1e10 (effectively no cutoff)
    fill_value : int | None, optional
        Value indicating padding in neighbor_matrix. If None, defaults to num_atoms.
        Entries with neighbor_matrix[i, k] >= fill_value are treated as padding. Default: None
    d3_params : D3Parameters | dict[str, torch.Tensor] | None, optional
        DFT-D3 parameters provided as either:
        - :class:`D3Parameters` dataclass instance (recommended)
        - Dictionary with keys: "rcov", "r4r2", "c6ab", "cn_ref"
        Individual parameters below can override values from d3_params.
    covalent_radii : torch.Tensor | None, optional
        Covalent radii [max_Z+1] as float32, indexed by atomic number, in same units
        as positions. If provided, overrides the value in d3_params.
    r4r2 : torch.Tensor | None, optional
        <r4>/<r2> expectation values [max_Z+1] as float32 for C8 computation (dimensionless).
        If provided, overrides the value in d3_params.
    c6_reference : torch.Tensor | None, optional
        C6 reference values [max_Z+1, max_Z+1, 5, 5] as float32 in energy × distance^6 units.
        If provided, overrides the value in d3_params.
    coord_num_ref : torch.Tensor | None, optional
        CN reference grid [max_Z+1, max_Z+1, 5, 5] as float32 (dimensionless).
        If provided, overrides the value in d3_params.
    batch_idx : torch.Tensor or None, optional
        Batch indices [num_atoms] as int32. If None, all atoms are assumed
        to be in a single system (batch 0). For batched calculations, atoms with
        the same batch index belong to the same system. Default: None
    cell : torch.Tensor or None, optional, as float32 or float64
        Unit cell lattice vectors [num_systems, 3, 3] for PBC, in same dtype and units as positions.
        Convention: cell[s, i, :] is i-th lattice vector for system s.
        If None, non-periodic calculation. Default: None
    neighbor_matrix : torch.Tensor | None, optional
        Neighbor indices [num_atoms, max_neighbors] as int32. See module docstring for
        details on the format. Padding entries have values >= fill_value.
        Mutually exclusive with neighbor_list. Default: None
    neighbor_matrix_shifts : torch.Tensor or None, optional
        Integer unit cell shifts [num_atoms, max_neighbors, 3] as int32 for PBC with
        neighbor_matrix format. If None, non-periodic calculation. If provided along
        with cell, Cartesian shifts are computed. Mutually exclusive with unit_shifts.
        Default: None
    neighbor_list : torch.Tensor or None, optional
        Neighbor pairs [2, num_pairs] as int32 in COO format, where row 0 contains
        source atom indices and row 1 contains target atom indices. Alternative to
        neighbor_matrix for sparse neighbor representations. Mutually exclusive with
        neighbor_matrix. Must be used together with `neighbor_ptr` (both are returned
        by the neighbor list API when `return_neighbor_list=True`).
        Default: None
    neighbor_ptr : torch.Tensor or None, optional
        CSR row pointers [num_atoms+1] as int32. Required when using `neighbor_list`.
        Indicates that `neighbor_list[1, :]` contains destination atoms in CSR
        format where
        `neighbor_ptr[i]:neighbor_ptr[i+1]` gives the range of neighbors for atom i.
        Returned by the neighbor list API when `return_neighbor_list=True`.
        Default: None
    unit_shifts : torch.Tensor or None, optional
        Integer unit cell shifts [num_pairs, 3] as int32 for PBC with neighbor_list
        format. If None, non-periodic calculation. If provided along with cell,
        Cartesian shifts are computed. Mutually exclusive with neighbor_matrix_shifts.
        Default: None
    compute_virial : bool, optional
        If True, allocate and compute virial tensor. Ignored if virial
        parameter is provided. Default: False
    num_systems : int, optional
        Number of systems in batch. In none provided, inferred from cell
        or from batch_idx (introcudes CUDA synchronization overhead). Default: None
    device : str or None, optional
        Warp device string (e.g., 'cuda:0', 'cpu'). If None, inferred from
        positions tensor. Default: None

    Returns
    -------
    energy : torch.Tensor
        Total dispersion energy [num_systems] as float32. Units are energy
        (Hartree when using standard D3 parameters).
    forces : torch.Tensor
        Atomic forces [num_atoms, 3] as float32. Units are energy/distance
        (Hartree/Bohr when using standard D3 parameters).
    coord_num : torch.Tensor
        Coordination numbers [num_atoms] as float32 (dimensionless)
    virial : torch.Tensor, optional
        Virial tensor [num_systems, 3, 3] as float32.
        Units are energy (Hartree when using standard D3 parameters). Only returned
        if compute_virial=True.

    Notes
    -----
    - **Unit consistency**: All inputs must use consistent units. Standard D3 parameters
      from the Grimme group use atomic units (Bohr for distances, Hartree for energy),
      so using atomic units throughout is recommended and conventional.
    - Float32 or float64 precision for positions and cell; outputs always float32
    - **Neighbor formats**: Supports both neighbor_matrix (dense) and neighbor_list (sparse COO)
      formats. Choose neighbor_list for sparse systems or when memory efficiency is important.
    - Padding atoms indicated by numbers[i] == 0
    - Requires symmetric neighbor representation (each pair appears twice)
    - **Two-body only**: Computes pairwise C6 and C8 dispersion terms; three-body
      Axilrod-Teller-Muto (ATM/C9) terms are not included
    - Virial computation requires periodic boundary conditions.
    - Bulk stress tensor can be obtained by dividing virial by system volume.

    **Neighbor Format Selection**:

    - Use neighbor_matrix for dense systems or when max_neighbors is small
    - Use neighbor_list for sparse systems, large cutoffs, or memory-constrained scenarios
    - Both formats produce identical results and support PBC

    **PBC Handling**:

    - Matrix format: Provide cell and neighbor_matrix_shifts
    - List format: Provide cell and unit_shifts
    - Non-periodic: Omit both cell and shift parameters

    See Also
    --------
    :class:`D3Parameters` : Dataclass for organizing DFT-D3 reference parameters
    :func:`_dftd3_matrix_op` : Internal custom operator for neighbor matrix format (non-PBC)
    :func:`_dftd3_matrix_pbc_op` : Internal custom operator for neighbor matrix format (PBC)
    :func:`_dftd3_op` : Internal custom operator for neighbor list format (non-PBC)
    :func:`_dftd3_pbc_op` : Internal custom operator for neighbor list format (PBC)
    """
    # Validate neighbor format inputs
    matrix_provided = neighbor_matrix is not None
    list_provided = neighbor_list is not None

    if matrix_provided and list_provided:
        raise ValueError(
            "Cannot provide both neighbor_matrix and neighbor_list. "
            "Please provide only one neighbor representation format."
        )
    if not matrix_provided and not list_provided:
        raise ValueError("Must provide either neighbor_matrix or neighbor_list.")

    # Validate PBC shift inputs match neighbor format
    if matrix_provided and unit_shifts is not None:
        raise ValueError(
            "unit_shifts is for neighbor_list format. "
            "Use neighbor_matrix_shifts for neighbor_matrix format."
        )
    if list_provided and neighbor_matrix_shifts is not None:
        raise ValueError(
            "neighbor_matrix_shifts is for neighbor_matrix format. "
            "Use unit_shifts for neighbor_list format."
        )

    # Validate neighbor_ptr is provided when using neighbor_list format
    if list_provided and neighbor_ptr is None:
        raise ValueError(
            "neighbor_ptr must be provided when using neighbor_list format. "
            "Obtain it from the neighbor list API by setting return_neighbor_list=True."
        )

    # Validate functional parameters
    if a1 is None or a2 is None or s8 is None:
        raise ValueError(
            "Functional parameters a1, a2, and s8 must be provided. "
            "These are functional-dependent parameters required for DFT-D3(BJ) calculations."
        )

    # Validate virial computation requires PBC
    if compute_virial:
        if cell is None:
            raise ValueError(
                "Virial computation requires periodic boundary conditions. "
                "Please provide unit cell parameters (cell) and shifts "
                "(neighbor_matrix_shifts or unit_shifts) when compute_virial=True "
                "or when passing a virial tensor."
            )
        if matrix_provided and neighbor_matrix_shifts is None:
            raise ValueError(
                "Virial computation requires periodic boundary conditions. "
                "Please provide neighbor_matrix_shifts along with cell when using "
                "neighbor_matrix format and compute_virial=True or passing a virial tensor."
            )
        if list_provided and unit_shifts is None:
            raise ValueError(
                "Virial computation requires periodic boundary conditions. "
                "Please provide unit_shifts along with cell when using "
                "neighbor_list format and compute_virial=True or passing a virial tensor."
            )

    # Determine how parameters are being supplied
    # Case 1: All individual parameters provided explicitly
    if all(
        param is not None
        for param in [covalent_radii, r4r2, c6_reference, coord_num_ref]
    ):
        # Use explicit parameters directly (already assigned)
        pass
    # Case 2: d3_params provided (D3Parameters or dictionary, with optional overrides)
    elif d3_params is not None:
        # Convert D3Parameters to dictionary for consistent access
        if isinstance(d3_params, D3Parameters):
            d3_params = d3_params.__dict__
        # these are written to throw KeyError if the keys are not present
        if covalent_radii is None:
            covalent_radii = d3_params["rcov"]
        if r4r2 is None:
            r4r2 = d3_params["r4r2"]
        if c6_reference is None:
            c6_reference = d3_params["c6ab"]
        if coord_num_ref is None:
            coord_num_ref = d3_params["cn_ref"]
    # Case 3: No parameters provided - raise error
    else:
        raise RuntimeError(
            "DFT-D3 parameters must be explicitly provided. "
            "Either supply all individual parameters (covalent_radii, r4r2, "
            "c6_reference, coord_num_ref), provide a D3Parameters instance, "
            "or provide a d3_params dictionary. See the function docstring for details."
        )

    # Get shapes
    num_atoms = positions.size(0)

    # Handle empty case
    if num_atoms == 0:
        if batch_idx is None or (
            isinstance(batch_idx, torch.Tensor) and batch_idx.numel() == 0
        ):
            num_systems = 1
        else:
            num_systems = int(batch_idx.max().item()) + 1

        empty_energy = torch.zeros(
            num_systems, dtype=torch.float32, device=positions.device
        )
        empty_forces = torch.zeros((0, 3), dtype=torch.float32, device=positions.device)
        empty_cn = torch.zeros((0,), dtype=torch.float32, device=positions.device)

        # Handle virial for empty case if compute_virial is True
        if compute_virial:
            empty_virial = torch.zeros(
                (0, 3, 3), dtype=torch.float32, device=positions.device
            )
            return empty_energy, empty_forces, empty_cn, empty_virial
        else:
            return empty_energy, empty_forces, empty_cn

    # Determine number of systems for energy allocation
    if num_systems is None:
        if batch_idx is None:
            num_systems = 1
        elif cell is not None:
            num_systems = cell.size(0)
        else:
            num_systems = int(batch_idx.max().item()) + 1

    # Allocate output tensors
    energy = torch.zeros(num_systems, dtype=torch.float32, device=positions.device)
    forces = torch.zeros((num_atoms, 3), dtype=torch.float32, device=positions.device)
    coord_num = torch.zeros(num_atoms, dtype=torch.float32, device=positions.device)
    if compute_virial:
        virial = torch.zeros(
            (num_systems, 3, 3), dtype=torch.float32, device=positions.device
        )
    else:
        virial = torch.zeros((0, 3, 3), dtype=torch.float32, device=positions.device)

    # Dispatch to appropriate implementation based on neighbor format and PBC
    if neighbor_matrix is not None:
        # Matrix format - dispatch based on PBC
        if cell is not None and neighbor_matrix_shifts is not None:
            # PBC variant
            _dftd3_matrix_pbc_op(
                positions=positions,
                numbers=numbers,
                neighbor_matrix=neighbor_matrix,
                cell=cell,
                neighbor_matrix_shifts=neighbor_matrix_shifts,
                covalent_radii=covalent_radii,
                r4r2=r4r2,
                c6_reference=c6_reference,
                coord_num_ref=coord_num_ref,
                a1=a1,
                a2=a2,
                s8=s8,
                energy=energy,
                forces=forces,
                coord_num=coord_num,
                virial=virial,
                k1=k1,
                k3=k3,
                s6=s6,
                s5_smoothing_on=s5_smoothing_on,
                s5_smoothing_off=s5_smoothing_off,
                fill_value=fill_value,
                batch_idx=batch_idx,
                compute_virial=compute_virial,
                device=device,
            )
        else:
            # Non-PBC variant
            _dftd3_matrix_op(
                positions=positions,
                numbers=numbers,
                neighbor_matrix=neighbor_matrix,
                covalent_radii=covalent_radii,
                r4r2=r4r2,
                c6_reference=c6_reference,
                coord_num_ref=coord_num_ref,
                a1=a1,
                a2=a2,
                s8=s8,
                energy=energy,
                forces=forces,
                coord_num=coord_num,
                virial=virial,
                k1=k1,
                k3=k3,
                s6=s6,
                s5_smoothing_on=s5_smoothing_on,
                s5_smoothing_off=s5_smoothing_off,
                fill_value=fill_value,
                batch_idx=batch_idx,
                device=device,
            )
    else:
        # List format - use CSR format from neighbor list API
        # neighbor_list: [2, num_pairs] in COO format where row 1 is idx_j (destination atoms)
        # neighbor_ptr: [num_atoms+1] CSR row pointers (required, from neighbor list API)

        # Extract idx_j from neighbor_list (row 1 contains destination atoms)
        idx_j_csr = neighbor_list[1]

        # Dispatch based on PBC
        if cell is not None and unit_shifts is not None:
            # PBC variant
            _dftd3_pbc_op(
                positions=positions,
                numbers=numbers,
                idx_j=idx_j_csr,
                neighbor_ptr=neighbor_ptr,
                cell=cell,
                unit_shifts=unit_shifts,
                covalent_radii=covalent_radii,
                r4r2=r4r2,
                c6_reference=c6_reference,
                coord_num_ref=coord_num_ref,
                a1=a1,
                a2=a2,
                s8=s8,
                energy=energy,
                forces=forces,
                coord_num=coord_num,
                virial=virial,
                k1=k1,
                k3=k3,
                s6=s6,
                s5_smoothing_on=s5_smoothing_on,
                s5_smoothing_off=s5_smoothing_off,
                batch_idx=batch_idx,
                compute_virial=compute_virial,
                device=device,
            )
        else:
            # Non-PBC variant
            _dftd3_op(
                positions=positions,
                numbers=numbers,
                idx_j=idx_j_csr,
                neighbor_ptr=neighbor_ptr,
                covalent_radii=covalent_radii,
                r4r2=r4r2,
                c6_reference=c6_reference,
                coord_num_ref=coord_num_ref,
                a1=a1,
                a2=a2,
                s8=s8,
                energy=energy,
                forces=forces,
                coord_num=coord_num,
                virial=virial,
                k1=k1,
                k3=k3,
                s6=s6,
                s5_smoothing_on=s5_smoothing_on,
                s5_smoothing_off=s5_smoothing_off,
                batch_idx=batch_idx,
                device=device,
            )

    if compute_virial:
        return energy, forces, coord_num, virial
    else:
        return energy, forces, coord_num
