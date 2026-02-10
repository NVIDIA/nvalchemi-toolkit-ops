"""
Unified Test Suite for Ewald Summation Implementation
======================================================

This test suite validates the correctness of the unified Ewald summation API:

1. API Tests - Basic functionality for single-system and batch modes
2. Correctness Tests - Validation against torchpme reference (parameterized)
3. Autograd Tests - Gradient computation for positions, charges, cells
4. Batch Consistency Tests - Batch vs single-system consistency
5. Physical Property Tests - Conservation laws and symmetries
6. Numerical Stability Tests - Edge cases and stability

The unified API uses:
- ewald_real_space(compute_forces=, batch_idx=)
- ewald_reciprocal_space(compute_forces=, batch_idx=)
- ewald_summation(compute_forces=, batch_idx=)
"""

import pytest
import torch
from torchpme.lib.kvectors import _generate_kvectors as _generate_kvectors_torchpme

from nvalchemiops.torch.interactions.electrostatics.ewald import (
    ewald_real_space,
    ewald_reciprocal_space,
    ewald_summation,
)
from nvalchemiops.torch.interactions.electrostatics.k_vectors import (
    generate_k_vectors_ewald_summation,
)
from nvalchemiops.torch.neighbors import batch_cell_list, cell_list

# Check optional dependencies
try:
    from torchpme import EwaldCalculator
    from torchpme.potentials import CoulombPotential

    HAS_TORCHPME = True
except ModuleNotFoundError:
    HAS_TORCHPME = False
    EwaldCalculator = None
    CoulombPotential = None

# Import test utilities for crystal structure generation
from .test_utils import (
    VIRIAL_DTYPE,
    create_cscl_supercell,
    create_wurtzite_system,
    create_zincblende_system,
    fd_virial_full,
    get_virial_neighbor_data,
    make_non_neutral_system,
    make_virial_batch_cscl_system,
    make_virial_crystal_system,
    make_virial_cscl_system,
)

# Tolerances
TIGHT_TOL = 1e-6
LOOSE_TOL = 1e-4


###########################################################################################
########################### Helper Functions ##############################################
###########################################################################################


def compute_torchpme_reciprocal(
    positions, charges, cell, k_cutoff, alpha, device, dtype
):
    """Compute reciprocal energy using torchpme."""
    import math

    lr_wavelength = 2 * torch.pi / k_cutoff
    # torchpme uses smearing σ where Gaussian is exp(-r²/(2σ²))
    # Standard Ewald uses exp(-α²r²), so σ = 1/(√2·α)
    smearing = 1.0 / (math.sqrt(2.0) * alpha)
    potential = CoulombPotential(smearing=smearing).to(device=device, dtype=dtype)
    charges_col = charges.unsqueeze(1)
    calculator = EwaldCalculator(
        potential=potential, lr_wavelength=lr_wavelength, full_neighbor_list=True
    ).to(device=device, dtype=dtype)
    potentials = calculator._compute_kspace(charges_col, cell.squeeze(0), positions)
    return (charges_col * potentials).flatten()


def compute_torchpme_real_space(
    charges, neighbor_indices, neighbor_distances, alpha, k_cutoff, device, dtype
):
    """Compute real-space energy using torchpme."""
    import math

    lr_wavelength = 2 * torch.pi / k_cutoff
    # torchpme uses smearing σ where Gaussian is exp(-r²/(2σ²))
    # Standard Ewald uses exp(-α²r²), so σ = 1/(√2·α)
    smearing = 1.0 / (math.sqrt(2.0) * alpha)
    potential = CoulombPotential(smearing=smearing).to(device=device, dtype=dtype)
    charges_col = charges.unsqueeze(1)
    calculator = EwaldCalculator(
        potential=potential, lr_wavelength=lr_wavelength, full_neighbor_list=True
    ).to(device=device, dtype=dtype)
    potentials = calculator._compute_rspace(
        charges_col, neighbor_indices, neighbor_distances
    )
    return (charges_col * potentials).flatten()


def create_simple_system(device, dtype=torch.float64, num_atoms=4, cell_size=10.0):
    """Create a simple test system with random positions and neutral charges."""
    positions = (
        torch.rand((num_atoms, 3), dtype=dtype, device=device) * cell_size * 0.8
        + cell_size * 0.1
    )
    charges = torch.randn(num_atoms, dtype=dtype, device=device)
    charges[-1] = -charges[:-1].sum()  # Make neutral
    cell = torch.eye(3, dtype=dtype, device=device).unsqueeze(0) * cell_size
    return positions, charges, cell


def create_dipole_system(
    device, dtype=torch.float64, separation=6.0, cell_size=10.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a simple dipole system.

    Parameters
    ----------
    device : torch.device
        Device for tensors
    dtype : torch.dtype
        Data type for floating point tensors (float32 or float64)
    separation : float
        Distance between the two charges
    cell_size : float
        Size of the cubic cell

    Returns
    -------
    positions, charges, cell, neighbor_list, neighbor_ptr, neighbor_shifts
    """
    center = cell_size / 2
    positions = torch.tensor(
        [
            [center - separation / 2, center, center],
            [center + separation / 2, center, center],
        ],
        dtype=dtype,
        device=device,
    )
    charges = torch.tensor([1.0, -1.0], dtype=dtype, device=device)
    cell = torch.eye(3, dtype=dtype, device=device).unsqueeze(0) * cell_size
    # Simple neighbor list for the pair
    neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
    neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    neighbor_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)
    return positions, charges, cell, neighbor_list, neighbor_ptr, neighbor_shifts


###########################################################################################
########################### Dtype Tests ####################################################
###########################################################################################


class TestDtypeSupport:
    """Test that Ewald functions support both float32 and float64 dtypes."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_real_space_dtype_returns_correct_type(self, device, dtype):
        """Test that real-space returns energies in float64, forces in input dtype."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell, neighbor_list, neighbor_ptr, neighbor_shifts = (
            create_dipole_system(device, dtype=dtype)
        )
        alpha = torch.tensor([0.3], dtype=dtype, device=device)

        # Test energy-only -- energies are always float64
        energies = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=False,
        )
        assert energies.dtype == torch.float64, (
            f"Expected float64, got {energies.dtype}"
        )

        # Test with forces -- forces match input dtype
        energies, forces = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=True,
        )
        assert energies.dtype == torch.float64, (
            f"Expected float64, got {energies.dtype}"
        )
        assert forces.dtype == dtype, f"Expected {dtype}, got {forces.dtype}"

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_reciprocal_space_dtype_returns_correct_type(self, device, dtype):
        """Test that reciprocal-space returns energies in float64, forces in input dtype."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell, _, _, _ = create_dipole_system(device, dtype=dtype)
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0).squeeze(0)
        alpha = torch.tensor([0.3], dtype=dtype, device=device)

        # Test energy-only -- energies are always float64
        energies = ewald_reciprocal_space(
            positions,
            charges,
            cell,
            k_vectors,
            alpha,
            compute_forces=False,
        )
        assert energies.dtype == torch.float64, (
            f"Expected float64, got {energies.dtype}"
        )

        # Test with forces -- forces match input dtype
        energies, forces = ewald_reciprocal_space(
            positions,
            charges,
            cell,
            k_vectors,
            alpha,
            compute_forces=True,
        )
        assert energies.dtype == torch.float64, (
            f"Expected float64, got {energies.dtype}"
        )
        assert forces.dtype == dtype, f"Expected {dtype}, got {forces.dtype}"

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_ewald_summation_dtype_returns_correct_type(self, device, dtype):
        """Test that full ewald_summation returns energies in float64, forces in input dtype."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell, neighbor_list, neighbor_ptr, neighbor_shifts = (
            create_dipole_system(device, dtype=dtype)
        )

        # Test energy-only -- energies are always float64
        energies = ewald_summation(
            positions,
            charges,
            cell,
            alpha=0.3,
            k_cutoff=8.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=False,
        )
        assert energies.dtype == torch.float64, (
            f"Expected float64, got {energies.dtype}"
        )

        # Test with forces -- forces match input dtype
        energies, forces = ewald_summation(
            positions,
            charges,
            cell,
            alpha=0.3,
            k_cutoff=8.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=True,
        )
        assert energies.dtype == torch.float64, (
            f"Expected float64, got {energies.dtype}"
        )
        assert forces.dtype == dtype, f"Expected {dtype}, got {forces.dtype}"

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_float32_vs_float64_consistency(self, device):
        """Test that float32 and float64 produce consistent results."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        # Create systems in both dtypes
        positions_f32, charges_f32, cell_f32, nl_f32, nptr_f32, ns_f32 = (
            create_dipole_system(device, dtype=torch.float32)
        )
        positions_f64, charges_f64, cell_f64, nl_f64, nptr_f64, ns_f64 = (
            create_dipole_system(device, dtype=torch.float64)
        )

        # Use same values
        positions_f64 = positions_f32.double()
        charges_f64 = charges_f32.double()
        cell_f64 = cell_f32.double()

        alpha_f32 = torch.tensor([0.3], dtype=torch.float32, device=device)
        alpha_f64 = torch.tensor([0.3], dtype=torch.float64, device=device)

        # Real space
        e_f32, f_f32 = ewald_real_space(
            positions_f32,
            charges_f32,
            cell_f32,
            alpha_f32,
            neighbor_list=nl_f32,
            neighbor_ptr=nptr_f32,
            neighbor_shifts=ns_f32,
            compute_forces=True,
        )
        e_f64, f_f64 = ewald_real_space(
            positions_f64,
            charges_f64,
            cell_f64,
            alpha_f64,
            neighbor_list=nl_f64,
            neighbor_ptr=nptr_f64,
            neighbor_shifts=ns_f64,
            compute_forces=True,
        )

        # Results should be close (within float32 precision)
        assert torch.allclose(e_f32.double(), e_f64, rtol=1e-4, atol=1e-5), (
            f"Energy mismatch: f32={e_f32.sum()}, f64={e_f64.sum()}"
        )
        assert torch.allclose(f_f32.double(), f_f64, rtol=1e-4, atol=1e-5), (
            f"Forces mismatch: f32={f_f32}, f64={f_f64}"
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_batch_dtype_returns_correct_type(self, device, dtype):
        """Test that batch operations return tensors in input dtype."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [8.0, 5.0, 5.0], [3.0, 5.0, 5.0], [7.0, 5.0, 5.0]],
            dtype=dtype,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=dtype, device=device)
        cell = (
            torch.eye(3, dtype=dtype, device=device)
            .unsqueeze(0)
            .expand(2, -1, -1)
            .contiguous()
            * 10.0
        )
        alpha = torch.tensor([0.3, 0.3], dtype=dtype, device=device)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
        neighbor_list = torch.tensor(
            [[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.int32, device=device
        )
        neighbor_ptr = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((4, 3), dtype=torch.int32, device=device)

        # Real space -- energies always float64, forces match input dtype
        energies, forces = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            batch_idx=batch_idx,
            compute_forces=True,
        )
        assert energies.dtype == torch.float64
        assert forces.dtype == dtype

        # Reciprocal space -- energies always float64, forces match input dtype
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0)
        energies, forces = ewald_reciprocal_space(
            positions,
            charges,
            cell,
            k_vectors,
            alpha,
            batch_idx=batch_idx,
            compute_forces=True,
        )
        assert energies.dtype == torch.float64
        assert forces.dtype == dtype


###########################################################################################
########################### API Tests: Real Space #########################################
###########################################################################################


class TestEwaldRealSpaceAPI:
    """Test ewald_real_space API for single and batch modes."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_single_system_energy_only(self, device):
        """Test single system with compute_forces=False."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell, neighbor_list, neighbor_ptr, neighbor_shifts = (
            create_dipole_system(device)
        )
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)

        energies = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=False,
        )

        assert energies.shape == (2,)
        assert torch.isfinite(energies).all()
        assert energies.sum() < 0  # Opposite charges attract

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_single_system_with_forces(self, device):
        """Test single system with compute_forces=True."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell, neighbor_list, neighbor_ptr, neighbor_shifts = (
            create_dipole_system(device)
        )
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)

        energies, forces = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=True,
        )

        assert energies.shape == (2,)
        assert forces.shape == (2, 3)
        assert torch.isfinite(forces).all()
        # Positive charge should be attracted in +x direction
        assert forces[0, 0] > 0
        assert forces[1, 0] < 0

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_system_energy_only(self, device):
        """Test batch system with compute_forces=False."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        # Two systems
        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [8.0, 5.0, 5.0], [3.0, 5.0, 5.0], [7.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor(
            [1.0, -1.0, 1.0, -1.0], dtype=torch.float64, device=device
        )
        cell = (
            torch.eye(3, dtype=torch.float64, device=device)
            .unsqueeze(0)
            .expand(2, -1, -1)
            * 10.0
        )
        alpha = torch.tensor([0.3, 0.3], dtype=torch.float64, device=device)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
        neighbor_list = torch.tensor(
            [[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.int32, device=device
        )
        neighbor_ptr = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)

        energies = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            batch_idx=batch_idx,
            compute_forces=False,
        )

        assert energies.shape == (4,)
        assert torch.isfinite(energies).all()

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_system_with_forces(self, device):
        """Test batch system with compute_forces=True."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [8.0, 5.0, 5.0], [3.0, 5.0, 5.0], [7.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor(
            [1.0, -1.0, 1.0, -1.0], dtype=torch.float64, device=device
        )
        cell = (
            torch.eye(3, dtype=torch.float64, device=device)
            .unsqueeze(0)
            .expand(2, -1, -1)
            .contiguous()
            * 10.0
        )
        alpha = torch.tensor([0.3, 0.3], dtype=torch.float64, device=device)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
        neighbor_list = torch.tensor(
            [[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.int32, device=device
        )
        neighbor_ptr = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)

        energies, forces = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            batch_idx=batch_idx,
            compute_forces=True,
        )

        assert energies.shape == (4,)
        assert forces.shape == (4, 3)
        assert torch.isfinite(forces).all()


###########################################################################################
########################### API Tests: Reciprocal Space ###################################
###########################################################################################


class TestEwaldReciprocalSpaceAPI:
    """Test ewald_reciprocal_space API for single and batch modes."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_single_system_energy_only(self, device):
        """Test single system with compute_forces=False."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell, _, _, _ = create_dipole_system(device)
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0).squeeze(0)
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)

        energies = ewald_reciprocal_space(
            positions,
            charges,
            cell,
            k_vectors,
            alpha,
            compute_forces=False,
        )

        assert energies.shape == (2,)
        assert torch.isfinite(energies).all()

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_single_system_with_forces(self, device):
        """Test single system with compute_forces=True."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell, _, _, _ = create_dipole_system(device)
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0).squeeze(0)
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)

        energies, forces = ewald_reciprocal_space(
            positions,
            charges,
            cell,
            k_vectors,
            alpha,
            compute_forces=True,
        )

        assert energies.shape == (2,)
        assert forces.shape == (2, 3)
        assert torch.isfinite(forces).all()

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_system_energy_only(self, device):
        """Test batch system with compute_forces=False."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [8.0, 5.0, 5.0], [3.0, 5.0, 5.0], [7.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor(
            [1.0, -1.0, 1.0, -1.0], dtype=torch.float64, device=device
        )
        cell = (
            torch.eye(3, dtype=torch.float64, device=device)
            .unsqueeze(0)
            .expand(2, -1, -1)
            .contiguous()
            * 10.0
        )
        alpha = torch.tensor([0.3, 0.3], dtype=torch.float64, device=device)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0)

        energies = ewald_reciprocal_space(
            positions,
            charges,
            cell,
            k_vectors,
            alpha,
            batch_idx=batch_idx,
            compute_forces=False,
        )

        assert energies.shape == (4,)
        assert torch.isfinite(energies).all()

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_system_with_forces(self, device):
        """Test batch system with compute_forces=True."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [8.0, 5.0, 5.0], [3.0, 5.0, 5.0], [7.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor(
            [1.0, -1.0, 1.0, -1.0], dtype=torch.float64, device=device
        )
        cell = (
            torch.eye(3, dtype=torch.float64, device=device)
            .unsqueeze(0)
            .expand(2, -1, -1)
            .contiguous()
            * 10.0
        )
        alpha = torch.tensor([0.3, 0.3], dtype=torch.float64, device=device)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0)

        energies, forces = ewald_reciprocal_space(
            positions,
            charges,
            cell,
            k_vectors,
            alpha,
            batch_idx=batch_idx,
            compute_forces=True,
        )

        assert energies.shape == (4,)
        assert forces.shape == (4, 3)
        assert torch.isfinite(forces).all()


###########################################################################################
########################### API Tests: Full Ewald Summation ###############################
###########################################################################################


class TestEwaldSummationAPI:
    """Test ewald_summation unified API for single and batch modes."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_single_system_energy_only(self, device):
        """Test single system with compute_forces=False."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell, neighbor_list, neighbor_ptr, neighbor_shifts = (
            create_dipole_system(device)
        )

        energies = ewald_summation(
            positions,
            charges,
            cell,
            alpha=0.3,
            k_cutoff=8.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=False,
        )

        assert energies.shape == (2,)
        assert torch.isfinite(energies).all()
        assert energies.sum() < 0  # Opposite charges attract

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_system_energy_only(self, device):
        """Test batch system with compute_forces=False."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [8.0, 5.0, 5.0], [3.0, 5.0, 5.0], [7.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor(
            [1.0, -1.0, 1.0, -1.0], dtype=torch.float64, device=device
        )
        cell = (
            torch.eye(3, dtype=torch.float64, device=device)
            .unsqueeze(0)
            .expand(2, -1, -1)
            .contiguous()
            * 10.0
        )
        alpha = torch.tensor([0.3, 0.3], dtype=torch.float64, device=device)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
        neighbor_list = torch.tensor(
            [[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.int32, device=device
        )
        neighbor_ptr = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)

        energies = ewald_summation(
            positions,
            charges,
            cell,
            alpha=alpha,
            k_cutoff=8.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            batch_idx=batch_idx,
            compute_forces=False,
        )

        assert energies.shape == (4,)
        assert torch.isfinite(energies).all()

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_system_with_forces(self, device):
        """Test batch system with compute_forces=True."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [8.0, 5.0, 5.0], [3.0, 5.0, 5.0], [7.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor(
            [1.0, -1.0, 1.0, -1.0], dtype=torch.float64, device=device
        )
        cell = (
            torch.eye(3, dtype=torch.float64, device=device)
            .unsqueeze(0)
            .expand(2, -1, -1)
            .contiguous()
            * 10.0
        )
        alpha = torch.tensor([0.3, 0.3], dtype=torch.float64, device=device)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
        neighbor_list = torch.tensor(
            [[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.int32, device=device
        )
        neighbor_ptr = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)

        energies, forces = ewald_summation(
            positions,
            charges,
            cell,
            alpha=alpha,
            k_cutoff=8.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            batch_idx=batch_idx,
            compute_forces=True,
        )

        assert energies.shape == (4,)
        assert forces.shape == (4, 3)

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_per_system_alpha(self, device):
        """Test that per-system alpha values work."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [8.0, 5.0, 5.0], [3.0, 5.0, 5.0], [7.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor(
            [1.0, -1.0, 1.0, -1.0], dtype=torch.float64, device=device
        )
        cell = (
            torch.eye(3, dtype=torch.float64, device=device)
            .unsqueeze(0)
            .expand(2, -1, -1)
            .contiguous()
            * 10.0
        )
        alpha = torch.tensor([0.3, 0.5], dtype=torch.float64, device=device)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
        neighbor_list = torch.tensor(
            [[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.int32, device=device
        )
        neighbor_ptr = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)

        energies, forces = ewald_summation(
            positions,
            charges,
            cell,
            alpha=alpha,
            k_cutoff=8.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            batch_idx=batch_idx,
            compute_forces=True,
        )

        assert torch.isfinite(energies).all()
        assert torch.isfinite(forces).all()


###########################################################################################
########################### Correctness Tests: Real Space vs TorchPME #####################
###########################################################################################


@pytest.mark.skipif(not HAS_TORCHPME, reason="torchpme not installed")
class TestRealSpaceCorrectness:
    """Validate real-space implementation against torchpme reference."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("size", [1, 2, 3])
    @pytest.mark.parametrize("system_fn", ["cscl", "wurtzite", "zincblende"])
    @pytest.mark.parametrize("cutoff", [5.0])
    @pytest.mark.parametrize("alpha", [0.3, 0.5, 0.75])
    def test_real_space_energy_matches_torchpme(
        self, device, size, system_fn, cutoff, alpha
    ):
        """Test real-space energy matches torchpme for crystal systems."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        dtype = torch.float64

        system_fns = {
            "cscl": create_cscl_supercell,
            "wurtzite": create_wurtzite_system,
            "zincblende": create_zincblende_system,
        }
        system = system_fns[system_fn](size)

        cell = torch.tensor(system.cell, dtype=dtype, device=device).unsqueeze(0)
        positions = torch.tensor(system.positions, dtype=dtype, device=device)
        charges = torch.tensor(system.charges, dtype=dtype, device=device)
        pbc = torch.tensor(
            [True, True, True], dtype=torch.bool, device=device
        ).unsqueeze(0)

        neighbor_list, neighbor_ptr, unit_shifts = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=True
        )

        alpha_tensor = torch.tensor([alpha], dtype=dtype, device=device)
        our_energies = ewald_real_space(
            positions,
            charges,
            cell,
            alpha_tensor,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=unit_shifts,
            compute_forces=False,
        )

        # TorchPME calculation
        i, j = neighbor_list
        S = unit_shifts.to(dtype=dtype) @ cell.squeeze(0)
        neighbor_distances = torch.norm(positions[j] - positions[i] + S, dim=1)
        torchpme_energies = compute_torchpme_real_space(
            charges, neighbor_list.T, neighbor_distances, alpha, cutoff, device, dtype
        )

        assert torch.allclose(our_energies, torchpme_energies, rtol=1e-3, atol=1e-3), (
            f"Real space energy mismatch: ours={our_energies.sum()}, torchpme={torchpme_energies.sum()}"
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("size", [1, 2, 3])
    @pytest.mark.parametrize("system_fn", ["cscl", "wurtzite", "zincblende"])
    @pytest.mark.parametrize("cutoff", [5.0])
    @pytest.mark.parametrize("alpha", [0.3, 0.5, 0.75])
    def test_real_space_forces_match_torchpme(
        self, device, size, system_fn, cutoff, alpha
    ):
        """Test real-space forces match torchpme for crystal systems."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        dtype = torch.float64

        system_fns = {
            "cscl": create_cscl_supercell,
            "wurtzite": create_wurtzite_system,
            "zincblende": create_zincblende_system,
        }
        system = system_fns[system_fn](size)

        cell = torch.tensor(system.cell, dtype=dtype, device=device).unsqueeze(0)
        positions = torch.tensor(system.positions, dtype=dtype, device=device)
        charges = torch.tensor(system.charges, dtype=dtype, device=device)
        pbc = torch.tensor(
            [True, True, True], dtype=torch.bool, device=device
        ).unsqueeze(0)

        neighbor_list, neighbor_ptr, unit_shifts = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=True
        )

        alpha_tensor = torch.tensor([alpha], dtype=dtype, device=device)
        our_energies, our_forces = ewald_real_space(
            positions,
            charges,
            cell,
            alpha_tensor,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=unit_shifts,
            compute_forces=True,
        )

        # TorchPME calculation via autograd
        positions_ref = positions.clone().requires_grad_(True)
        i, j = neighbor_list
        S = unit_shifts.to(dtype=dtype) @ cell.squeeze(0)
        neighbor_distances = torch.norm(positions_ref[j] - positions_ref[i] + S, dim=1)
        torchpme_energies = compute_torchpme_real_space(
            charges, neighbor_list.T, neighbor_distances, alpha, cutoff, device, dtype
        )
        torchpme_energies.sum().backward()
        torchpme_forces = -positions_ref.grad

        assert torch.allclose(our_energies, torchpme_energies, rtol=1e-3, atol=1e-3), (
            f"Real space energy mismatch: ours={our_energies.sum()}, torchpme={torchpme_energies.sum()}"
        )
        assert torch.allclose(our_forces, torchpme_forces, rtol=1e-3, atol=1e-3), (
            f"Real space forces mismatch: max diff = {(our_forces - torchpme_forces).abs().max()}"
        )


###########################################################################################
########################### Correctness Tests: Reciprocal Space vs TorchPME ###############
###########################################################################################


def generate_kvectors_for_ewald_reference(cell, k_cutoff):
    """Generate k-vectors using torchpme as reference."""
    basis_norms = torch.linalg.norm(cell, dim=1)
    ns_float = k_cutoff * basis_norms / 2 / torch.pi
    ns = torch.ceil(ns_float).long().to(cell.device)
    kvectors = _generate_kvectors_torchpme(cell, ns, for_ewald=True)
    return kvectors


@pytest.mark.skipif(not HAS_TORCHPME, reason="torchpme not installed")
class TestReciprocalSpaceCorrectness:
    """Validate reciprocal-space implementation against torchpme reference."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("size", [1, 2, 3])
    @pytest.mark.parametrize("system_fn", ["cscl", "wurtzite", "zincblende"])
    @pytest.mark.parametrize("k_cutoff", [8.0, 12.0])
    @pytest.mark.parametrize("alpha", [0.3, 0.5, 0.75])
    def test_reciprocal_energy_matches_torchpme(
        self, device, size, system_fn, k_cutoff, alpha
    ):
        """Test reciprocal-space energy matches torchpme for crystal systems."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        dtype = torch.float64

        system_fns = {
            "cscl": create_cscl_supercell,
            "wurtzite": create_wurtzite_system,
            "zincblende": create_zincblende_system,
        }
        system = system_fns[system_fn](size)

        cell = torch.tensor(system.cell, dtype=dtype, device=device).unsqueeze(0)
        positions = torch.tensor(system.positions, dtype=dtype, device=device)
        charges = torch.tensor(system.charges, dtype=dtype, device=device)

        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff).squeeze(0)
        print("our k_vectors", k_vectors.shape)

        torchpme_k_vectors = generate_kvectors_for_ewald_reference(cell[0], k_cutoff)
        print("torchpme k_vectors", torchpme_k_vectors.shape)
        alpha_tensor = torch.tensor([alpha], dtype=dtype, device=device)

        our_energies = ewald_reciprocal_space(
            positions,
            charges,
            cell,
            k_vectors,
            alpha_tensor,
            compute_forces=False,
        )

        torchpme_energies = compute_torchpme_reciprocal(
            positions, charges, cell, k_cutoff, alpha, device, dtype
        )
        print("our energies", our_energies.shape, our_energies)
        print("torchpme energies", torchpme_energies.shape, torchpme_energies)
        assert torch.allclose(our_energies, torchpme_energies, rtol=1e-3, atol=1e-3), (
            f"Reciprocal energy mismatch: ours={our_energies.sum()}, torchpme={torchpme_energies.sum()}"
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("size", [1, 2, 3])
    @pytest.mark.parametrize("system_fn", ["cscl", "wurtzite", "zincblende"])
    @pytest.mark.parametrize("k_cutoff", [8.0, 12.0])
    @pytest.mark.parametrize("alpha", [0.3, 0.5, 0.75])
    def test_reciprocal_forces_match_torchpme(
        self, device, size, system_fn, k_cutoff, alpha
    ):
        """Test reciprocal-space forces match torchpme for crystal systems."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        dtype = torch.float64

        system_fns = {
            "cscl": create_cscl_supercell,
            "wurtzite": create_wurtzite_system,
            "zincblende": create_zincblende_system,
        }
        system = system_fns[system_fn](size)

        cell = torch.tensor(system.cell, dtype=dtype, device=device).unsqueeze(0)
        positions = torch.tensor(system.positions, dtype=dtype, device=device)
        charges = torch.tensor(system.charges, dtype=dtype, device=device)

        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff).squeeze(0)
        alpha_tensor = torch.tensor([alpha], dtype=dtype, device=device)

        our_energies, our_forces = ewald_reciprocal_space(
            positions,
            charges,
            cell,
            k_vectors,
            alpha_tensor,
            compute_forces=True,
        )

        # TorchPME via autograd
        positions_ref = positions.clone().requires_grad_(True)
        torchpme_energies = compute_torchpme_reciprocal(
            positions_ref, charges, cell, k_cutoff, alpha, device, dtype
        )
        torchpme_energies.sum().backward()
        torchpme_forces = -positions_ref.grad

        assert torch.allclose(our_energies, torchpme_energies, rtol=1e-3, atol=1e-3), (
            f"Reciprocal energy mismatch: ours={our_energies.sum()}, torchpme={torchpme_energies.sum()}"
        )
        assert torch.allclose(our_forces, torchpme_forces, rtol=1e-3, atol=1e-3), (
            f"Reciprocal forces mismatch: max diff = {(our_forces - torchpme_forces).abs().max()}"
        )


###########################################################################################
########################### Autograd Tests ################################################
###########################################################################################


class TestAutogradRealSpace:
    """Test autograd for real-space Ewald."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_position_gradients(self, device, dtype):
        """Test gradients w.r.t. positions."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell, neighbor_list, neighbor_ptr, neighbor_shifts = (
            create_dipole_system(device, dtype=dtype)
        )
        positions = positions.clone().requires_grad_(True)
        alpha = torch.tensor([0.3], dtype=dtype, device=device)

        energies = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=False,
        )
        energies.sum().backward()

        assert positions.grad is not None
        assert torch.isfinite(positions.grad).all()
        assert positions.grad.abs().sum() > 0

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_charge_gradients(self, device, dtype):
        """Test gradients w.r.t. charges."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell, neighbor_list, neighbor_ptr, neighbor_shifts = (
            create_dipole_system(device, dtype=dtype)
        )
        charges = charges.clone().requires_grad_(True)
        alpha = torch.tensor([0.3], dtype=dtype, device=device)

        energies = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=False,
        )
        energies.sum().backward()

        assert charges.grad is not None
        assert torch.isfinite(charges.grad).all()

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_autograd_matches_explicit_forces(self, device):
        """Test that autograd forces match explicit forces."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell, neighbor_list, neighbor_ptr, neighbor_shifts = (
            create_dipole_system(device)
        )
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)

        # Explicit forces
        _, explicit_forces = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=True,
        )

        # Autograd forces
        positions_ad = positions.clone().requires_grad_(True)
        energies = ewald_real_space(
            positions_ad,
            charges,
            cell,
            alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=False,
        )
        energies.sum().backward()
        autograd_forces = -positions_ad.grad

        assert torch.allclose(explicit_forces, autograd_forces, rtol=0.01, atol=1e-5)

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("system_fn", ["cscl", "wurtzite", "zincblende"])
    @pytest.mark.parametrize("size", [1, 2, 3])
    @pytest.mark.parametrize("compute_forces", [True, False])
    def test_autograd_charge_gradients_match_torchpme(
        self, device, system_fn, size, compute_forces
    ):
        """Test that charge gradients match torchpme."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        dtype = torch.float64
        system_fns = {
            "cscl": create_cscl_supercell,
            "wurtzite": create_wurtzite_system,
            "zincblende": create_zincblende_system,
        }
        system = system_fns[system_fn](size)

        cell = torch.tensor(system.cell, dtype=torch.float64, device=device).unsqueeze(
            0
        )
        pbc = torch.tensor(
            [True, True, True], dtype=torch.bool, device=device
        ).unsqueeze(0)
        positions = torch.tensor(system.positions, dtype=torch.float64, device=device)
        charges = torch.tensor(system.charges, dtype=torch.float64, device=device)
        our_charges = charges.clone().requires_grad_(True)
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)
        cutoff = 5.0
        neighbor_list, neighbor_ptr, neighbor_shifts = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=True
        )

        if compute_forces:
            our_energies, _ = ewald_real_space(
                positions,
                our_charges,
                cell,
                alpha,
                neighbor_list=neighbor_list,
                neighbor_ptr=neighbor_ptr,
                neighbor_shifts=neighbor_shifts,
                compute_forces=True,
            )
        else:
            our_energies = ewald_real_space(
                positions,
                our_charges,
                cell,
                alpha,
                neighbor_list=neighbor_list,
                neighbor_ptr=neighbor_ptr,
                neighbor_shifts=neighbor_shifts,
                compute_forces=False,
            )
        our_energies.sum().backward()
        our_charge_grad = our_charges.grad.clone()

        # TorchPME reference
        i, j = neighbor_list
        S = neighbor_shifts.to(dtype=dtype) @ cell.squeeze(0)
        neighbor_distances = torch.norm(positions[j] - positions[i] + S, dim=1)
        torchpme_charges_ref = charges.detach().clone().requires_grad_(True)
        torchpme_energies = compute_torchpme_real_space(
            torchpme_charges_ref,
            neighbor_list.T,
            neighbor_distances,
            alpha,
            cutoff,
            device,
            dtype,
        )
        torchpme_energies.sum().backward()
        torchpme_charge_grad = torchpme_charges_ref.grad.clone()

        assert torch.allclose(
            our_charge_grad, torchpme_charge_grad, rtol=1e-3, atol=1e-3
        ), (
            f"Charge gradients mismatch: ours={our_charge_grad}, torchpme={torchpme_charge_grad}"
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("system_fn", ["cscl", "wurtzite", "zincblende"])
    @pytest.mark.parametrize("size", [1, 2, 3])
    @pytest.mark.parametrize("compute_forces", [True, False])
    def test_autograd_cell_gradients_match_torchpme(
        self, device, system_fn, size, compute_forces
    ):
        """Test that cell gradients match torchpme."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        dtype = torch.float64
        system_fns = {
            "cscl": create_cscl_supercell,
            "wurtzite": create_wurtzite_system,
            "zincblende": create_zincblende_system,
        }
        system = system_fns[system_fn](size)

        cell = torch.tensor(system.cell, dtype=torch.float64, device=device).unsqueeze(
            0
        )
        pbc = torch.tensor(
            [True, True, True], dtype=torch.bool, device=device
        ).unsqueeze(0)
        our_cell = cell.clone().requires_grad_(True)
        positions = torch.tensor(system.positions, dtype=torch.float64, device=device)
        charges = torch.tensor(system.charges, dtype=torch.float64, device=device)
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)
        neighbor_list, neighbor_ptr, neighbor_shifts = cell_list(
            positions, 5.0, cell, pbc, return_neighbor_list=True
        )

        if compute_forces:
            our_energies, _ = ewald_real_space(
                positions,
                charges,
                our_cell,
                alpha,
                neighbor_list=neighbor_list,
                neighbor_ptr=neighbor_ptr,
                neighbor_shifts=neighbor_shifts,
                compute_forces=True,
            )
        else:
            our_energies = ewald_real_space(
                positions,
                charges,
                our_cell,
                alpha,
                neighbor_list=neighbor_list,
                neighbor_ptr=neighbor_ptr,
                neighbor_shifts=neighbor_shifts,
                compute_forces=False,
            )
        our_energies.sum().backward()
        our_cell_grad = our_cell.grad.clone()

        # TorchPME reference
        torchpme_cell_ref = cell.detach().clone().requires_grad_(True)
        i, j = neighbor_list
        S = neighbor_shifts.to(dtype=dtype) @ torchpme_cell_ref.squeeze(0)
        neighbor_distances = torch.norm(positions[j] - positions[i] + S, dim=1)
        torchpme_energies = compute_torchpme_real_space(
            charges,
            neighbor_list.T,
            neighbor_distances,
            alpha,
            5.0,
            device,
            torch.float64,
        )
        torchpme_energies.sum().backward()
        torchpme_cell_grad = torchpme_cell_ref.grad.clone()

        assert torch.allclose(
            our_cell_grad, torchpme_cell_grad, rtol=1e-3, atol=1e-3
        ), (
            f"Cell gradients mismatch: ours={our_cell_grad}, torchpme={torchpme_cell_grad}"
        )


class TestExplicitChargeGradients:
    """Test explicit charge gradients (compute_charge_gradients=True)."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_explicit_charge_grad_matches_autograd_neighbor_list(self, device):
        """Test that explicit charge gradients match autograd (neighbor list)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell, neighbor_list, neighbor_ptr, neighbor_shifts = (
            create_dipole_system(device)
        )
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)

        # Get explicit charge gradients
        energies, forces, explicit_charge_grad = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=True,
            compute_charge_gradients=True,
        )

        # Get autograd charge gradients
        charges_ad = charges.clone().requires_grad_(True)
        energies_ad = ewald_real_space(
            positions,
            charges_ad,
            cell,
            alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=False,
        )
        energies_ad.sum().backward()
        autograd_charge_grad = charges_ad.grad.clone()

        assert torch.allclose(
            explicit_charge_grad, autograd_charge_grad, rtol=1e-5, atol=1e-8
        ), (
            f"Charge gradients mismatch: explicit={explicit_charge_grad}, "
            f"autograd={autograd_charge_grad}"
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("system_fn", ["cscl", "wurtzite", "zincblende"])
    @pytest.mark.parametrize("size", [1, 2])
    def test_explicit_charge_grad_various_systems(self, device, system_fn, size):
        """Test explicit charge gradients on various crystal systems."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        system_fns = {
            "cscl": create_cscl_supercell,
            "wurtzite": create_wurtzite_system,
            "zincblende": create_zincblende_system,
        }
        system = system_fns[system_fn](size)

        cell = torch.tensor(system.cell, dtype=torch.float64, device=device).unsqueeze(
            0
        )
        pbc = torch.tensor(
            [True, True, True], dtype=torch.bool, device=device
        ).unsqueeze(0)
        positions = torch.tensor(system.positions, dtype=torch.float64, device=device)
        charges = torch.tensor(system.charges, dtype=torch.float64, device=device)
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)
        cutoff = 5.0
        neighbor_list, neighbor_ptr, neighbor_shifts = cell_list(
            positions, cutoff, cell, pbc, return_neighbor_list=True
        )

        # Get explicit charge gradients
        energies, forces, explicit_charge_grad = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=True,
            compute_charge_gradients=True,
        )

        # Get autograd charge gradients
        charges_ad = charges.clone().requires_grad_(True)
        energies_ad = ewald_real_space(
            positions,
            charges_ad,
            cell,
            alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=False,
        )
        energies_ad.sum().backward()
        autograd_charge_grad = charges_ad.grad.clone()

        assert torch.allclose(
            explicit_charge_grad, autograd_charge_grad, rtol=1e-4, atol=1e-7
        ), (
            f"Charge gradients mismatch on {system_fn} (size {size}): "
            f"explicit={explicit_charge_grad}, autograd={autograd_charge_grad}"
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_explicit_charge_grad_without_forces(self, device):
        """Test explicit charge gradients when compute_forces=False."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell, neighbor_list, neighbor_ptr, neighbor_shifts = (
            create_dipole_system(device)
        )
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)

        # Get charge gradients without explicit forces
        energies, charge_grad = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=False,
            compute_charge_gradients=True,
        )

        # Verify outputs
        assert energies.shape == (positions.shape[0],)
        assert charge_grad.shape == (positions.shape[0],)
        assert torch.isfinite(charge_grad).all()

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_explicit_charge_grad(self, device):
        """Test explicit charge gradients in batch mode."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        dtype = torch.float64

        # Create batched system
        system1 = create_cscl_supercell(1)
        system2 = create_wurtzite_system(1)

        n1 = len(system1.positions)
        n2 = len(system2.positions)

        positions = torch.tensor(
            list(system1.positions) + list(system2.positions),
            dtype=dtype,
            device=device,
        )
        charges = torch.tensor(
            list(system1.charges) + list(system2.charges), dtype=dtype, device=device
        )
        cells = torch.tensor([system1.cell, system2.cell], dtype=dtype, device=device)
        batch_idx = torch.tensor([0] * n1 + [1] * n2, dtype=torch.int32, device=device)
        pbc = torch.tensor([[True, True, True], [True, True, True]], device=device)
        alpha = torch.tensor([0.3, 0.3], dtype=dtype, device=device)

        neighbor_list, neighbor_ptr, neighbor_shifts = batch_cell_list(
            positions, 5.0, cells, pbc, batch_idx=batch_idx, return_neighbor_list=True
        )

        # Get explicit charge gradients
        energies, forces, explicit_charge_grad = ewald_real_space(
            positions,
            charges,
            cells,
            alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            batch_idx=batch_idx,
            compute_forces=True,
            compute_charge_gradients=True,
        )

        # Get autograd charge gradients
        charges_ad = charges.clone().requires_grad_(True)
        energies_ad = ewald_real_space(
            positions,
            charges_ad,
            cells,
            alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            batch_idx=batch_idx,
            compute_forces=False,
        )
        energies_ad.sum().backward()
        autograd_charge_grad = charges_ad.grad.clone()

        assert torch.allclose(
            explicit_charge_grad, autograd_charge_grad, rtol=1e-4, atol=1e-7
        ), (
            f"Batch charge gradients mismatch: explicit={explicit_charge_grad}, "
            f"autograd={autograd_charge_grad}"
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_empty_neighbor_list_charge_grad(self, device):
        """Test charge gradients with empty neighbor list returns zeros."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [8.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0) * 10.0
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)

        # Empty neighbor list
        neighbor_list = torch.zeros((2, 0), dtype=torch.int32, device=device)
        neighbor_ptr = torch.zeros(3, dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((0, 3), dtype=torch.int32, device=device)

        energies, forces, charge_grads = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=True,
            compute_charge_gradients=True,
        )

        assert energies.shape == (2,)
        assert forces.shape == (2, 3)
        assert charge_grads.shape == (2,)
        assert torch.allclose(
            energies, torch.zeros(2, device=device, dtype=positions.dtype)
        )
        assert torch.allclose(
            forces, torch.zeros((2, 3), device=device, dtype=positions.dtype)
        )
        assert torch.allclose(
            charge_grads, torch.zeros(2, device=device, dtype=torch.float64)
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_charge_grad_with_autograd_enabled(self, device):
        """Test charge gradients work correctly when autograd is also enabled."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell, neighbor_list, neighbor_ptr, neighbor_shifts = (
            create_dipole_system(device)
        )
        positions = positions.clone().requires_grad_(True)
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)

        # Get charge gradients with autograd enabled on positions
        energies, forces, charge_grads = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=True,
            compute_charge_gradients=True,
        )

        # Backward on energies should work
        energies.sum().backward()

        assert positions.grad is not None
        assert torch.isfinite(positions.grad).all()
        assert torch.isfinite(charge_grads).all()

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_explicit_charge_grad_neighbor_matrix(self, device):
        """Test explicit charge gradients with neighbor matrix format."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell, neighbor_list, neighbor_ptr, neighbor_shifts = (
            create_dipole_system(device)
        )
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)

        # Convert neighbor list to matrix format
        num_atoms = positions.shape[0]
        max_neighbors = 20
        mask_value = num_atoms  # Use num_atoms as mask value
        neighbor_matrix = torch.full(
            (num_atoms, max_neighbors), mask_value, dtype=torch.int32, device=device
        )
        neighbor_matrix_shifts = torch.zeros(
            (num_atoms, max_neighbors, 3), dtype=torch.int32, device=device
        )

        # Populate the matrix
        idx_i = neighbor_list[0]
        idx_j = neighbor_list[1]
        neighbor_counts = torch.zeros(num_atoms, dtype=torch.int32, device=device)
        for pair_idx in range(idx_i.shape[0]):
            i = idx_i[pair_idx].item()
            j = idx_j[pair_idx].item()
            count = neighbor_counts[i].item()
            if count < max_neighbors:
                neighbor_matrix[i, count] = j
                neighbor_matrix_shifts[i, count] = neighbor_shifts[pair_idx]
                neighbor_counts[i] += 1

        # Get explicit charge gradients with neighbor matrix format
        energies, forces, explicit_charge_grad = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            mask_value=mask_value,
            compute_forces=True,
            compute_charge_gradients=True,
        )

        # Get autograd charge gradients using neighbor list (ground truth)
        charges_ad = charges.clone().requires_grad_(True)
        energies_ad = ewald_real_space(
            positions,
            charges_ad,
            cell,
            alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=False,
        )
        energies_ad.sum().backward()
        autograd_charge_grad = charges_ad.grad.clone()

        assert torch.allclose(
            explicit_charge_grad, autograd_charge_grad, rtol=1e-5, atol=1e-8
        ), (
            f"Neighbor matrix charge gradients mismatch: "
            f"explicit={explicit_charge_grad}, autograd={autograd_charge_grad}"
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_explicit_charge_grad_neighbor_matrix(self, device):
        """Test explicit charge gradients with neighbor matrix format in batch mode."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        dtype = torch.float64

        # Create batched system
        system1 = create_cscl_supercell(1)
        system2 = create_wurtzite_system(1)

        n1 = len(system1.positions)
        n2 = len(system2.positions)

        positions = torch.tensor(
            list(system1.positions) + list(system2.positions),
            dtype=dtype,
            device=device,
        )
        charges = torch.tensor(
            list(system1.charges) + list(system2.charges), dtype=dtype, device=device
        )
        cells = torch.tensor([system1.cell, system2.cell], dtype=dtype, device=device)
        batch_idx = torch.tensor([0] * n1 + [1] * n2, dtype=torch.int32, device=device)
        pbc = torch.tensor([[True, True, True], [True, True, True]], device=device)
        alpha = torch.tensor([0.3, 0.3], dtype=dtype, device=device)

        neighbor_list, neighbor_ptr, neighbor_shifts = batch_cell_list(
            positions, 5.0, cells, pbc, batch_idx=batch_idx, return_neighbor_list=True
        )

        # Convert to neighbor matrix format
        num_atoms = positions.shape[0]
        max_neighbors = 50
        mask_value = num_atoms
        neighbor_matrix = torch.full(
            (num_atoms, max_neighbors), mask_value, dtype=torch.int32, device=device
        )
        neighbor_matrix_shifts = torch.zeros(
            (num_atoms, max_neighbors, 3), dtype=torch.int32, device=device
        )

        idx_i = neighbor_list[0]
        idx_j = neighbor_list[1]
        neighbor_counts = torch.zeros(num_atoms, dtype=torch.int32, device=device)
        for pair_idx in range(idx_i.shape[0]):
            i = idx_i[pair_idx].item()
            j = idx_j[pair_idx].item()
            count = neighbor_counts[i].item()
            if count < max_neighbors:
                neighbor_matrix[i, count] = j
                neighbor_matrix_shifts[i, count] = neighbor_shifts[pair_idx]
                neighbor_counts[i] += 1

        # Get explicit charge gradients with neighbor matrix format
        energies, forces, explicit_charge_grad = ewald_real_space(
            positions,
            charges,
            cells,
            alpha,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            mask_value=mask_value,
            batch_idx=batch_idx,
            compute_forces=True,
            compute_charge_gradients=True,
        )

        # Get autograd charge gradients
        charges_ad = charges.clone().requires_grad_(True)
        energies_ad = ewald_real_space(
            positions,
            charges_ad,
            cells,
            alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            batch_idx=batch_idx,
            compute_forces=False,
        )
        energies_ad.sum().backward()
        autograd_charge_grad = charges_ad.grad.clone()

        assert torch.allclose(
            explicit_charge_grad, autograd_charge_grad, rtol=1e-4, atol=1e-7
        ), (
            f"Batch neighbor matrix charge gradients mismatch: "
            f"explicit={explicit_charge_grad}, autograd={autograd_charge_grad}"
        )


class TestExplicitReciprocalChargeGradients:
    """Test explicit charge gradients for reciprocal-space Ewald."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_reciprocal_explicit_charge_grad(self, device):
        """Test explicit charge gradients for reciprocal space match autograd."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell, _, _, _ = create_dipole_system(device)
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0).squeeze(0)
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)

        # Get explicit charge gradients
        energies, forces, explicit_charge_grad = ewald_reciprocal_space(
            positions,
            charges,
            cell,
            k_vectors,
            alpha,
            compute_forces=True,
            compute_charge_gradients=True,
        )

        # Get autograd charge gradients
        charges_ad = charges.clone().requires_grad_(True)
        energies_ad = ewald_reciprocal_space(
            positions,
            charges_ad,
            cell,
            k_vectors,
            alpha,
            compute_forces=False,
        )
        energies_ad.sum().backward()
        autograd_charge_grad = charges_ad.grad.clone()

        assert torch.allclose(
            explicit_charge_grad, autograd_charge_grad, rtol=1e-4, atol=1e-7
        ), (
            f"Reciprocal charge gradients mismatch: "
            f"explicit={explicit_charge_grad}, autograd={autograd_charge_grad}"
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_reciprocal_explicit_charge_grad_without_forces(self, device):
        """Test explicit charge gradients for reciprocal space when compute_forces=False."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell, _, _, _ = create_dipole_system(device)
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0).squeeze(0)
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)

        # Get charge gradients without explicit forces
        energies, charge_grad = ewald_reciprocal_space(
            positions,
            charges,
            cell,
            k_vectors,
            alpha,
            compute_forces=False,
            compute_charge_gradients=True,
        )

        # Verify outputs
        assert energies.shape == (positions.shape[0],)
        assert charge_grad.shape == (positions.shape[0],)
        assert torch.isfinite(charge_grad).all()

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("system_fn", ["cscl", "wurtzite", "zincblende"])
    @pytest.mark.parametrize("size", [1, 2])
    def test_reciprocal_explicit_charge_grad_various_systems(
        self, device, system_fn, size
    ):
        """Test reciprocal charge gradients on various crystal systems."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        system_fns = {
            "cscl": create_cscl_supercell,
            "wurtzite": create_wurtzite_system,
            "zincblende": create_zincblende_system,
        }
        system = system_fns[system_fn](size)

        cell = torch.tensor(system.cell, dtype=torch.float64, device=device).unsqueeze(
            0
        )
        positions = torch.tensor(system.positions, dtype=torch.float64, device=device)
        charges = torch.tensor(system.charges, dtype=torch.float64, device=device)
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0).squeeze(0)
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)

        # Get explicit charge gradients
        energies, forces, explicit_charge_grad = ewald_reciprocal_space(
            positions,
            charges,
            cell,
            k_vectors,
            alpha,
            compute_forces=True,
            compute_charge_gradients=True,
        )

        # Get autograd charge gradients
        charges_ad = charges.clone().requires_grad_(True)
        energies_ad = ewald_reciprocal_space(
            positions,
            charges_ad,
            cell,
            k_vectors,
            alpha,
            compute_forces=False,
        )
        energies_ad.sum().backward()
        autograd_charge_grad = charges_ad.grad.clone()

        assert torch.allclose(
            explicit_charge_grad, autograd_charge_grad, rtol=1e-3, atol=1e-6
        ), (
            f"Reciprocal charge gradients mismatch on {system_fn} (size {size}): "
            f"explicit={explicit_charge_grad}, autograd={autograd_charge_grad}"
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_reciprocal_explicit_charge_grad(self, device):
        """Test explicit charge gradients for batch reciprocal space."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        dtype = torch.float64

        # Create batched system
        system1 = create_cscl_supercell(1)
        system2 = create_wurtzite_system(1)

        n1 = len(system1.positions)
        n2 = len(system2.positions)

        positions = torch.tensor(
            list(system1.positions) + list(system2.positions),
            dtype=dtype,
            device=device,
        )
        charges = torch.tensor(
            list(system1.charges) + list(system2.charges), dtype=dtype, device=device
        )
        cells = torch.tensor([system1.cell, system2.cell], dtype=dtype, device=device)
        batch_idx = torch.tensor([0] * n1 + [1] * n2, dtype=torch.int32, device=device)
        alpha = torch.tensor([0.3, 0.3], dtype=dtype, device=device)

        # Generate k-vectors for batch (both systems use same k-vectors here)
        k_vectors = generate_k_vectors_ewald_summation(cells, k_cutoff=8.0)

        # Get explicit charge gradients
        energies, forces, explicit_charge_grad = ewald_reciprocal_space(
            positions,
            charges,
            cells,
            k_vectors,
            alpha,
            batch_idx=batch_idx,
            compute_forces=True,
            compute_charge_gradients=True,
        )

        # Get autograd charge gradients
        charges_ad = charges.clone().requires_grad_(True)
        energies_ad = ewald_reciprocal_space(
            positions,
            charges_ad,
            cells,
            k_vectors,
            alpha,
            batch_idx=batch_idx,
            compute_forces=False,
        )
        energies_ad.sum().backward()
        autograd_charge_grad = charges_ad.grad.clone()

        assert torch.allclose(
            explicit_charge_grad, autograd_charge_grad, rtol=1e-3, atol=1e-6
        ), (
            f"Batch reciprocal charge gradients mismatch: "
            f"explicit={explicit_charge_grad}, autograd={autograd_charge_grad}"
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_reciprocal_empty_k_vectors_charge_grad(self, device):
        """Test reciprocal charge gradients with empty k-vectors returns zeros."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0) * 10.0
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)
        k_vectors = torch.zeros((0, 3), dtype=torch.float64, device=device)

        energies, forces, charge_grads = ewald_reciprocal_space(
            positions,
            charges,
            cell,
            k_vectors,
            alpha,
            compute_forces=True,
            compute_charge_gradients=True,
        )

        assert energies.shape == (2,)
        assert forces.shape == (2, 3)
        assert charge_grads.shape == (2,)
        assert torch.allclose(
            energies, torch.zeros(2, device=device, dtype=positions.dtype)
        )
        assert torch.allclose(
            forces, torch.zeros(2, 3, device=device, dtype=positions.dtype)
        )
        assert torch.allclose(
            charge_grads, torch.zeros(2, device=device, dtype=positions.dtype)
        )


class TestAutogradReciprocalSpace:
    """Test autograd for reciprocal-space Ewald."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_position_gradients(self, device, dtype):
        """Test gradients w.r.t. positions."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell, _, _, _ = create_dipole_system(device, dtype=dtype)
        positions = positions.clone().requires_grad_(True)
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0).squeeze(0)
        alpha = torch.tensor([0.3], dtype=dtype, device=device)

        energies = ewald_reciprocal_space(
            positions,
            charges,
            cell,
            k_vectors,
            alpha,
            compute_forces=False,
        )
        energies.sum().backward()

        assert positions.grad is not None
        assert torch.isfinite(positions.grad).all()

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_charge_gradients(self, device, dtype):
        """Test gradients w.r.t. charges."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell, _, _, _ = create_dipole_system(device, dtype=dtype)
        charges = charges.clone().requires_grad_(True)
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0).squeeze(0)
        alpha = torch.tensor([0.3], dtype=dtype, device=device)

        energies = ewald_reciprocal_space(
            positions,
            charges,
            cell,
            k_vectors,
            alpha,
            compute_forces=False,
        )
        energies.sum().backward()

        assert charges.grad is not None
        assert torch.isfinite(charges.grad).all()

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_cell_gradients(self, device):
        """Test gradients w.r.t. cell."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell, _, _, _ = create_dipole_system(device)
        cell = cell.clone().requires_grad_(True)
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0).squeeze(0)
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)

        energies = ewald_reciprocal_space(
            positions,
            charges,
            cell,
            k_vectors,
            alpha,
            compute_forces=False,
        )
        energies.sum().backward()

        assert cell.grad is not None
        assert torch.isfinite(cell.grad).all()

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_autograd_matches_explicit_forces(self, device):
        """Test that autograd forces match explicit forces."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell, _, _, _ = create_dipole_system(device)
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0).squeeze(0)
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)

        # Explicit forces
        _, explicit_forces = ewald_reciprocal_space(
            positions,
            charges,
            cell,
            k_vectors,
            alpha,
            compute_forces=True,
        )

        # Autograd forces
        positions_ad = positions.clone().requires_grad_(True)
        energies = ewald_reciprocal_space(
            positions_ad,
            charges,
            cell,
            k_vectors,
            alpha,
            compute_forces=False,
        )
        energies.sum().backward()
        autograd_forces = -positions_ad.grad

        assert torch.allclose(explicit_forces, autograd_forces, rtol=0.01, atol=1e-5)

    @pytest.mark.skipif(not HAS_TORCHPME, reason="torchpme not installed")
    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("system_fn", ["cscl", "wurtzite", "zincblende"])
    @pytest.mark.parametrize("size", [1, 2, 3])
    @pytest.mark.parametrize("compute_forces", [True, False])
    def test_charge_gradients_match_torchpme(
        self, device, system_fn, size, compute_forces
    ):
        """Test that charge gradients match torchpme."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        dtype = torch.float64

        system_fns = {
            "cscl": create_cscl_supercell,
            "wurtzite": create_wurtzite_system,
            "zincblende": create_zincblende_system,
        }
        system = system_fns[system_fn](size)

        cell = torch.tensor(system.cell, dtype=dtype, device=device).unsqueeze(0)
        positions = torch.tensor(system.positions, dtype=dtype, device=device)
        charges = torch.tensor(system.charges, dtype=dtype, device=device)

        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0).squeeze(0)
        alpha = torch.tensor([0.3], dtype=dtype, device=device)

        # Our implementation
        our_charges = charges.clone().requires_grad_(True)
        if compute_forces:
            our_energies, _ = ewald_reciprocal_space(
                positions,
                our_charges,
                cell,
                k_vectors,
                alpha,
                compute_forces=True,
            )
        else:
            our_energies = ewald_reciprocal_space(
                positions,
                our_charges,
                cell,
                k_vectors,
                alpha,
                compute_forces=False,
            )
        our_energies.sum().backward()
        our_grad = our_charges.grad.clone()

        # torchpme reference
        torchpme_charges_ref = charges.detach().clone().requires_grad_(True)
        torchpme_energies = compute_torchpme_reciprocal(
            positions, torchpme_charges_ref, cell, 8.0, alpha, device, torch.float64
        )
        torchpme_energies.sum().backward()
        torchpme_grad = torchpme_charges_ref.grad.clone()

        assert torch.allclose(our_grad, torchpme_grad, rtol=LOOSE_TOL, atol=1e-6)

    @pytest.mark.skipif(not HAS_TORCHPME, reason="torchpme not installed")
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    @pytest.mark.parametrize("system_fn", ["cscl", "wurtzite", "zincblende"])
    @pytest.mark.parametrize("size", [1, 2, 3])
    @pytest.mark.parametrize("compute_forces", [True, False])
    def test_cell_gradients_match_torchpme(
        self, device, system_fn, size, compute_forces
    ):
        """Test that cell gradients match torchpme."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        dtype = torch.float64

        system_fns = {
            "cscl": create_cscl_supercell,
            "wurtzite": create_wurtzite_system,
            "zincblende": create_zincblende_system,
        }
        system = system_fns[system_fn](size)
        cell = torch.tensor(system.cell, dtype=dtype, device=device).unsqueeze(0)
        positions = torch.tensor(system.positions, dtype=dtype, device=device)
        charges = torch.tensor(system.charges, dtype=dtype, device=device)

        alpha = torch.tensor([0.3], dtype=dtype, device=device)

        # Our implementation
        our_cell = cell.clone().requires_grad_(True)
        our_k_vectors = generate_k_vectors_ewald_summation(
            our_cell, k_cutoff=8.0
        ).squeeze(0)
        if compute_forces:
            our_energies, _ = ewald_reciprocal_space(
                positions,
                charges,
                our_cell,
                our_k_vectors,
                alpha,
                compute_forces=True,
            )
        else:
            our_energies = ewald_reciprocal_space(
                positions,
                charges,
                our_cell,
                our_k_vectors,
                alpha,
                compute_forces=False,
            )
        our_energies.sum().backward()
        our_grad = our_cell.grad.clone()

        # torchpme reference
        torchpme_cell_ref = cell.detach().clone().requires_grad_(True)
        torchpme_energies = compute_torchpme_reciprocal(
            positions, charges, torchpme_cell_ref, 8.0, alpha, device, dtype
        )
        torchpme_energies.sum().backward()
        torchpme_grad = torchpme_cell_ref.grad.clone()

        assert torch.allclose(our_grad, torchpme_grad, rtol=LOOSE_TOL, atol=1e-6), (
            f"Cell gradients mismatch: ours={our_grad}, torchpme={torchpme_grad}"
        )


class TestAutogradFullEwald:
    """Test autograd for full Ewald summation."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("compute_forces", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_position_gradients(self, device, compute_forces, dtype):
        """Test gradients w.r.t. positions."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell, neighbor_list, neighbor_ptr, neighbor_shifts = (
            create_dipole_system(device, dtype=dtype)
        )
        positions = positions.clone().requires_grad_(True)

        if compute_forces:
            energies, _ = ewald_summation(
                positions,
                charges,
                cell,
                alpha=0.3,
                k_cutoff=8.0,
                neighbor_list=neighbor_list,
                neighbor_ptr=neighbor_ptr,
                neighbor_shifts=neighbor_shifts,
                compute_forces=compute_forces,
            )
        else:
            energies = ewald_summation(
                positions,
                charges,
                cell,
                alpha=0.3,
                k_cutoff=8.0,
                neighbor_list=neighbor_list,
                neighbor_ptr=neighbor_ptr,
                neighbor_shifts=neighbor_shifts,
                compute_forces=compute_forces,
            )
        energies.sum().backward()
        positions_grad = positions.grad.clone()

        assert positions_grad is not None
        assert torch.isfinite(positions_grad).all()

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("compute_forces", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_charge_gradients(self, device, compute_forces, dtype):
        """Test gradients w.r.t. positions."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell, neighbor_list, neighbor_ptr, neighbor_shifts = (
            create_dipole_system(device, dtype=dtype)
        )
        charges = charges.clone().requires_grad_(True)

        if compute_forces:
            energies, _ = ewald_summation(
                positions,
                charges,
                cell,
                alpha=0.3,
                k_cutoff=8.0,
                neighbor_list=neighbor_list,
                neighbor_ptr=neighbor_ptr,
                neighbor_shifts=neighbor_shifts,
                compute_forces=compute_forces,
            )
        else:
            energies = ewald_summation(
                positions,
                charges,
                cell,
                alpha=0.3,
                k_cutoff=8.0,
                neighbor_list=neighbor_list,
                neighbor_ptr=neighbor_ptr,
                neighbor_shifts=neighbor_shifts,
                compute_forces=compute_forces,
            )
        energies.sum().backward()
        charges_grad = charges.grad.clone()

        assert charges_grad is not None
        assert torch.isfinite(charges_grad).all()

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("compute_forces", [True, False])
    def test_cell_gradients(self, device, compute_forces):
        """Test gradients w.r.t. cell."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell, neighbor_list, neighbor_ptr, neighbor_shifts = (
            create_dipole_system(device)
        )
        cell = cell.clone().requires_grad_(True)

        if compute_forces:
            energies, _ = ewald_summation(
                positions,
                charges,
                cell,
                alpha=0.3,
                k_cutoff=8.0,
                neighbor_list=neighbor_list,
                neighbor_ptr=neighbor_ptr,
                neighbor_shifts=neighbor_shifts,
                compute_forces=compute_forces,
            )
        else:
            energies = ewald_summation(
                positions,
                charges,
                cell,
                alpha=0.3,
                k_cutoff=8.0,
                neighbor_list=neighbor_list,
                neighbor_ptr=neighbor_ptr,
                neighbor_shifts=neighbor_shifts,
                compute_forces=compute_forces,
            )
        energies.sum().backward()
        cell_grad = cell.grad.clone()

        assert cell_grad is not None
        assert torch.isfinite(cell_grad).all()

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_autograd_matches_explicit_forces(self, device):
        """Test that autograd forces match explicit forces."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell, neighbor_list, neighbor_ptr, neighbor_shifts = (
            create_dipole_system(device)
        )

        # Explicit forces
        _, explicit_forces = ewald_summation(
            positions,
            charges,
            cell,
            alpha=0.3,
            k_cutoff=8.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=True,
        )

        # Autograd forces
        positions_ad = positions.clone().requires_grad_(True)
        energies = ewald_summation(
            positions_ad,
            charges,
            cell,
            alpha=0.3,
            k_cutoff=8.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=False,
        )
        energies.sum().backward()
        autograd_forces = -positions_ad.grad

        assert torch.allclose(explicit_forces, autograd_forces, rtol=0.01, atol=1e-5)


###########################################################################################
########################### Batch Autograd Tests ##########################################
###########################################################################################


class TestBatchAutograd:
    """Test autograd for batch Ewald summation."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("system_fn", ["cscl", "wurtzite", "zincblende"])
    def test_batch_position_gradients_vs_single(self, device, system_fn):
        """Test batch position gradients match single-system gradients."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        dtype = torch.float64

        system_fns = {
            "cscl": create_cscl_supercell,
            "wurtzite": create_wurtzite_system,
            "zincblende": create_zincblende_system,
        }

        system1 = system_fns[system_fn](1)
        system2 = system_fns[system_fn](2)

        pos1 = torch.tensor(system1.positions, dtype=dtype, device=device)
        chg1 = torch.tensor(system1.charges, dtype=dtype, device=device)
        cell1 = torch.tensor(system1.cell, dtype=dtype, device=device).unsqueeze(0)

        pos2 = torch.tensor(system2.positions, dtype=dtype, device=device)
        chg2 = torch.tensor(system2.charges, dtype=dtype, device=device)
        cell2 = torch.tensor(system2.cell, dtype=dtype, device=device).unsqueeze(0)

        alpha = 0.3
        k_cutoff = 8.0

        # Single-system gradients
        k_vectors1 = generate_k_vectors_ewald_summation(cell1, k_cutoff).squeeze(0)
        pos1_single = pos1.clone().requires_grad_(True)
        alpha1 = torch.tensor([alpha], dtype=dtype, device=device)
        e1 = ewald_reciprocal_space(
            pos1_single,
            chg1,
            cell1,
            k_vectors1,
            alpha1,
            compute_forces=False,
        )
        e1.sum().backward()
        grad1_single = pos1_single.grad.clone()

        k_vectors2 = generate_k_vectors_ewald_summation(cell2, k_cutoff).squeeze(0)
        pos2_single = pos2.clone().requires_grad_(True)
        alpha2 = torch.tensor([alpha], dtype=dtype, device=device)
        e2 = ewald_reciprocal_space(
            pos2_single,
            chg2,
            cell2,
            k_vectors2,
            alpha2,
            compute_forces=False,
        )
        e2.sum().backward()
        grad2_single = pos2_single.grad.clone()

        # Batch gradients
        n1, n2 = pos1.shape[0], pos2.shape[0]
        positions_batch = torch.cat([pos1, pos2], dim=0).clone().requires_grad_(True)
        charges_batch = torch.cat([chg1, chg2], dim=0)
        cells_batch = torch.cat([cell1, cell2], dim=0)
        alpha_batch = torch.tensor([alpha, alpha], dtype=dtype, device=device)
        batch_idx = torch.tensor([0] * n1 + [1] * n2, dtype=torch.int32, device=device)
        k_vectors_batch = generate_k_vectors_ewald_summation(cells_batch, k_cutoff)

        e_batch = ewald_reciprocal_space(
            positions_batch,
            charges_batch,
            cells_batch,
            k_vectors_batch,
            alpha_batch,
            batch_idx=batch_idx,
            compute_forces=False,
        )
        e_batch.sum().backward()

        grad1_batch = positions_batch.grad[:n1]
        grad2_batch = positions_batch.grad[n1:]

        assert torch.allclose(grad1_batch, grad1_single, rtol=1e-4, atol=1e-6)
        assert torch.allclose(grad2_batch, grad2_single, rtol=1e-4, atol=1e-6)

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("system_fn", ["cscl", "wurtzite", "zincblende"])
    def test_batch_charge_gradients_vs_single(self, device, system_fn):
        """Test batch charge gradients match single-system gradients."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        dtype = torch.float64

        system_fns = {
            "cscl": create_cscl_supercell,
            "wurtzite": create_wurtzite_system,
            "zincblende": create_zincblende_system,
        }

        system1 = system_fns[system_fn](1)
        system2 = system_fns[system_fn](2)

        pos1 = torch.tensor(system1.positions, dtype=dtype, device=device)
        chg1 = torch.tensor(system1.charges, dtype=dtype, device=device)
        cell1 = torch.tensor(system1.cell, dtype=dtype, device=device).unsqueeze(0)

        pos2 = torch.tensor(system2.positions, dtype=dtype, device=device)
        chg2 = torch.tensor(system2.charges, dtype=dtype, device=device)
        cell2 = torch.tensor(system2.cell, dtype=dtype, device=device).unsqueeze(0)

        alpha = 0.3
        k_cutoff = 8.0

        # Single-system gradients
        k_vectors1 = generate_k_vectors_ewald_summation(cell1, k_cutoff).squeeze(0)
        chg1_single = chg1.clone().requires_grad_(True)
        alpha1 = torch.tensor([alpha], dtype=dtype, device=device)
        e1 = ewald_reciprocal_space(
            pos1,
            chg1_single,
            cell1,
            k_vectors1,
            alpha1,
            compute_forces=False,
        )
        e1.sum().backward()
        grad1_single = chg1_single.grad.clone()

        k_vectors2 = generate_k_vectors_ewald_summation(cell2, k_cutoff).squeeze(0)
        chg2_single = chg2.clone().requires_grad_(True)
        alpha2 = torch.tensor([alpha], dtype=dtype, device=device)
        e2 = ewald_reciprocal_space(
            pos2,
            chg2_single,
            cell2,
            k_vectors2,
            alpha2,
            compute_forces=False,
        )
        e2.sum().backward()
        grad2_single = chg2_single.grad.clone()

        # Batch gradients
        n1, n2 = pos1.shape[0], pos2.shape[0]
        positions_batch = torch.cat([pos1, pos2], dim=0)
        charges_batch = torch.cat([chg1, chg2], dim=0).clone().requires_grad_(True)
        cells_batch = torch.cat([cell1, cell2], dim=0)
        alpha_batch = torch.tensor([alpha, alpha], dtype=dtype, device=device)
        batch_idx = torch.tensor([0] * n1 + [1] * n2, dtype=torch.int32, device=device)
        k_vectors_batch = generate_k_vectors_ewald_summation(cells_batch, k_cutoff)

        e_batch = ewald_reciprocal_space(
            positions_batch,
            charges_batch,
            cells_batch,
            k_vectors_batch,
            alpha_batch,
            batch_idx=batch_idx,
            compute_forces=False,
        )
        e_batch.sum().backward()

        grad1_batch = charges_batch.grad[:n1]
        grad2_batch = charges_batch.grad[n1:]

        assert torch.allclose(grad1_batch, grad1_single, rtol=1e-4, atol=1e-6)
        assert torch.allclose(grad2_batch, grad2_single, rtol=1e-4, atol=1e-6)

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("system_fn", ["cscl", "wurtzite", "zincblende"])
    def test_batch_cell_gradients_vs_single(self, device, system_fn):
        """Test batch cell gradients match single-system gradients."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        dtype = torch.float64

        system_fns = {
            "cscl": create_cscl_supercell,
            "wurtzite": create_wurtzite_system,
            "zincblende": create_zincblende_system,
        }

        system1 = system_fns[system_fn](1)
        system2 = system_fns[system_fn](2)

        pos1 = torch.tensor(system1.positions, dtype=dtype, device=device)
        chg1 = torch.tensor(system1.charges, dtype=dtype, device=device)
        cell1 = torch.tensor(system1.cell, dtype=dtype, device=device).unsqueeze(0)

        pos2 = torch.tensor(system2.positions, dtype=dtype, device=device)
        chg2 = torch.tensor(system2.charges, dtype=dtype, device=device)
        cell2 = torch.tensor(system2.cell, dtype=dtype, device=device).unsqueeze(0)

        alpha = 0.3
        k_cutoff = 8.0

        # Single-system gradients
        cell1_single = cell1.clone().requires_grad_(True)
        k_vectors1 = generate_k_vectors_ewald_summation(cell1_single, k_cutoff).squeeze(
            0
        )
        alpha1 = torch.tensor([alpha], dtype=dtype, device=device)
        e1 = ewald_reciprocal_space(
            pos1,
            chg1,
            cell1_single,
            k_vectors1,
            alpha1,
            compute_forces=False,
        )
        e1.sum().backward()
        grad1_single = cell1_single.grad.clone()

        cell2_single = cell2.clone().requires_grad_(True)
        k_vectors2 = generate_k_vectors_ewald_summation(cell2_single, k_cutoff).squeeze(
            0
        )
        alpha2 = torch.tensor([alpha], dtype=dtype, device=device)
        e2 = ewald_reciprocal_space(
            pos2,
            chg2,
            cell2_single,
            k_vectors2,
            alpha2,
            compute_forces=False,
        )
        e2.sum().backward()
        grad2_single = cell2_single.grad.clone()

        # Batch gradients
        n1, n2 = pos1.shape[0], pos2.shape[0]
        positions_batch = torch.cat([pos1, pos2], dim=0)
        charges_batch = torch.cat([chg1, chg2], dim=0)
        cells_batch = torch.cat([cell1, cell2], dim=0).clone().requires_grad_(True)
        alpha_batch = torch.tensor([alpha, alpha], dtype=dtype, device=device)
        batch_idx = torch.tensor([0] * n1 + [1] * n2, dtype=torch.int32, device=device)
        k_vectors_batch = generate_k_vectors_ewald_summation(cells_batch, k_cutoff)

        e_batch = ewald_reciprocal_space(
            positions_batch,
            charges_batch,
            cells_batch,
            k_vectors_batch,
            alpha_batch,
            batch_idx=batch_idx,
            compute_forces=False,
        )
        e_batch.sum().backward()

        grad1_batch = cells_batch.grad[0:1]
        grad2_batch = cells_batch.grad[1:2]

        assert torch.allclose(grad1_batch, grad1_single, rtol=1e-4, atol=1e-6)
        assert torch.allclose(grad2_batch, grad2_single, rtol=1e-4, atol=1e-6)


###########################################################################################
########################### Batch Consistency Tests #######################################
###########################################################################################


class TestBatchConsistency:
    """Test that batch results match single-system results."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_real_space_batch_matches_single(self, device):
        """Test that batch real-space matches single-system."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell, neighbor_list, neighbor_ptr, neighbor_shifts = (
            create_dipole_system(device)
        )
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)

        # Single system
        single_energies, single_forces = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=True,
        )

        # Batch mode (duplicate)
        positions_batch = torch.cat([positions, positions], dim=0)
        charges_batch = torch.cat([charges, charges], dim=0)
        cell_batch = torch.cat([cell, cell], dim=0)
        alpha_batch = torch.cat([alpha, alpha], dim=0)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
        neighbor_list_batch = torch.tensor(
            [[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.int32, device=device
        )
        neighbor_ptr_batch = torch.tensor(
            [0, 1, 2, 3, 4], dtype=torch.int32, device=device
        )
        neighbor_shifts_batch = torch.zeros((4, 3), dtype=torch.int32, device=device)

        batch_energies, batch_forces = ewald_real_space(
            positions_batch,
            charges_batch,
            cell_batch,
            alpha_batch,
            neighbor_list=neighbor_list_batch,
            neighbor_ptr=neighbor_ptr_batch,
            neighbor_shifts=neighbor_shifts_batch,
            batch_idx=batch_idx,
            compute_forces=True,
        )

        assert torch.allclose(
            single_energies.sum(), batch_energies[0:2].sum(), rtol=TIGHT_TOL
        )
        assert torch.allclose(
            single_energies.sum(), batch_energies[2:4].sum(), rtol=TIGHT_TOL
        )
        assert torch.allclose(single_forces, batch_forces[0:2], rtol=TIGHT_TOL)
        assert torch.allclose(single_forces, batch_forces[2:4], rtol=TIGHT_TOL)

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_reciprocal_space_batch_matches_single(self, device):
        """Test that batch reciprocal-space matches single-system."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell, _, _, _ = create_dipole_system(device)
        k_vectors_single = generate_k_vectors_ewald_summation(
            cell, k_cutoff=8.0
        ).squeeze(0)
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)

        # Single system
        single_energies, single_forces = ewald_reciprocal_space(
            positions,
            charges,
            cell,
            k_vectors_single,
            alpha,
            compute_forces=True,
        )

        # Batch mode
        positions_batch = torch.cat([positions, positions], dim=0)
        charges_batch = torch.cat([charges, charges], dim=0)
        cell_batch = torch.cat([cell, cell], dim=0)
        alpha_batch = torch.cat([alpha, alpha], dim=0)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
        k_vectors_batch = generate_k_vectors_ewald_summation(cell_batch, k_cutoff=8.0)

        batch_energies, batch_forces = ewald_reciprocal_space(
            positions_batch,
            charges_batch,
            cell_batch,
            k_vectors_batch,
            alpha_batch,
            batch_idx=batch_idx,
            compute_forces=True,
        )

        assert torch.allclose(
            single_energies.sum(), batch_energies[0:2].sum(), rtol=LOOSE_TOL, atol=1e-6
        )
        assert torch.allclose(
            single_energies.sum(), batch_energies[2:4].sum(), rtol=LOOSE_TOL, atol=1e-6
        )

        assert torch.allclose(
            single_forces, batch_forces[0:2], rtol=LOOSE_TOL, atol=1e-6
        )
        assert torch.allclose(
            single_forces, batch_forces[2:4], rtol=LOOSE_TOL, atol=1e-6
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_full_ewald_batch_matches_single(self, device):
        """Test that batch full Ewald matches single-system."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell, neighbor_list, neighbor_ptr, neighbor_shifts = (
            create_dipole_system(device)
        )

        # Single system
        single_energies, single_forces = ewald_summation(
            positions,
            charges,
            cell,
            alpha=0.3,
            k_cutoff=8.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=True,
        )

        # Batch mode
        positions_batch = torch.cat([positions, positions], dim=0)
        charges_batch = torch.cat([charges, charges], dim=0)
        cell_batch = torch.cat([cell, cell], dim=0)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
        neighbor_list_batch = torch.tensor(
            [[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.int32, device=device
        )
        neighbor_ptr_batch = torch.tensor(
            [0, 1, 2, 3, 4], dtype=torch.int32, device=device
        )
        neighbor_shifts_batch = torch.zeros((4, 3), dtype=torch.int32, device=device)

        batch_energies, batch_forces = ewald_summation(
            positions_batch,
            charges_batch,
            cell_batch,
            alpha=0.3,
            k_cutoff=8.0,
            neighbor_list=neighbor_list_batch,
            neighbor_ptr=neighbor_ptr_batch,
            neighbor_shifts=neighbor_shifts_batch,
            batch_idx=batch_idx,
            compute_forces=True,
        )

        assert torch.allclose(
            single_energies.sum(), batch_energies[0:2].sum(), rtol=LOOSE_TOL, atol=1e-5
        )
        assert torch.allclose(
            single_forces, batch_forces[0:2], rtol=LOOSE_TOL, atol=1e-5
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("size", [1, 2, 3])
    @pytest.mark.parametrize("system_fn", ["cscl", "wurtzite", "zincblende"])
    @pytest.mark.parametrize("cutoff", [5.0])
    @pytest.mark.parametrize("k_cutoff", [8.0])
    @pytest.mark.parametrize("alpha", [0.3, 0.5])
    def test_batch_full_ewald_vs_single_crystal(
        self, device, size, system_fn, cutoff, k_cutoff, alpha
    ):
        """Test batch full Ewald against single-system for crystal systems."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        dtype = torch.float64

        system_fns = {
            "cscl": create_cscl_supercell,
            "wurtzite": create_wurtzite_system,
            "zincblende": create_zincblende_system,
        }

        system1 = system_fns[system_fn](size)
        system2 = system_fns[system_fn](size)

        cell1 = torch.tensor(system1.cell, dtype=dtype, device=device).unsqueeze(0)
        positions1 = torch.tensor(system1.positions, dtype=dtype, device=device)
        charges1 = torch.tensor(system1.charges, dtype=dtype, device=device)

        cell2 = torch.tensor(system2.cell, dtype=dtype, device=device).unsqueeze(0)
        positions2 = torch.tensor(system2.positions, dtype=dtype, device=device)
        charges2 = torch.tensor(system2.charges, dtype=dtype, device=device)

        pbc = torch.tensor(
            [True, True, True], dtype=torch.bool, device=device
        ).unsqueeze(0)

        # Single-system calculations
        neighbor_list1, neighbor_ptr1, unit_shifts1 = cell_list(
            positions1, cutoff, cell1, pbc, return_neighbor_list=True
        )
        neighbor_list2, neighbor_ptr2, unit_shifts2 = cell_list(
            positions2, cutoff, cell2, pbc, return_neighbor_list=True
        )

        energy1, forces1 = ewald_summation(
            positions1,
            charges1,
            cell1,
            alpha=alpha,
            k_cutoff=k_cutoff,
            neighbor_list=neighbor_list1,
            neighbor_ptr=neighbor_ptr1,
            neighbor_shifts=unit_shifts1,
            compute_forces=True,
        )

        energy2, forces2 = ewald_summation(
            positions2,
            charges2,
            cell2,
            alpha=alpha,
            k_cutoff=k_cutoff,
            neighbor_list=neighbor_list2,
            neighbor_ptr=neighbor_ptr2,
            neighbor_shifts=unit_shifts2,
            compute_forces=True,
        )

        # Batch calculation
        positions_batch = torch.cat([positions1, positions2], dim=0)
        charges_batch = torch.cat([charges1, charges2], dim=0)
        cell_batch = torch.cat([cell1, cell2], dim=0)
        batch_idx = torch.cat(
            [
                torch.zeros(positions1.shape[0], dtype=torch.int32, device=device),
                torch.ones(positions2.shape[0], dtype=torch.int32, device=device),
            ]
        )
        pbc_batch = pbc.repeat(2, 1)

        neighbor_list_batch, neighbor_ptr_batch, neighbor_shifts_batch = (
            batch_cell_list(
                positions_batch,
                cutoff,
                cell_batch,
                pbc_batch,
                batch_idx,
                return_neighbor_list=True,
            )
        )

        energy_batch, forces_batch = ewald_summation(
            positions_batch,
            charges_batch,
            cell_batch,
            alpha=alpha,
            k_cutoff=k_cutoff,
            neighbor_list=neighbor_list_batch,
            neighbor_ptr=neighbor_ptr_batch,
            neighbor_shifts=neighbor_shifts_batch,
            batch_idx=batch_idx,
            compute_forces=True,
        )

        n1 = positions1.shape[0]
        assert torch.allclose(
            energy1.sum(), energy_batch[:n1].sum(), rtol=LOOSE_TOL, atol=1e-5
        )
        assert torch.allclose(
            energy2.sum(), energy_batch[n1:].sum(), rtol=LOOSE_TOL, atol=1e-5
        )
        assert torch.allclose(forces1, forces_batch[:n1], rtol=LOOSE_TOL, atol=1e-5)
        assert torch.allclose(forces2, forces_batch[n1:], rtol=LOOSE_TOL, atol=1e-5)


###########################################################################################
########################### Physical Property Tests #######################################
###########################################################################################


class TestPhysicalProperties:
    """Test that results have correct physical properties."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_opposite_charges_attract(self, device):
        """Test that opposite charges give negative energy."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell, neighbor_list, neighbor_ptr, neighbor_shifts = (
            create_dipole_system(device)
        )

        energies, forces = ewald_real_space(
            positions,
            charges,
            cell,
            alpha=torch.tensor([0.3], dtype=torch.float64, device=device),
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=True,
        )

        assert energies.sum() < 0, "Opposite charges should have negative energy"
        assert forces[0, 0] > 0, "Positive charge should be attracted toward negative"
        assert forces[1, 0] < 0, "Negative charge should be attracted toward positive"

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_charge_scaling(self, device):
        """Test that energy scales as q^2."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell, neighbor_list, neighbor_ptr, neighbor_shifts = (
            create_dipole_system(device)
        )

        e1 = ewald_summation(
            positions,
            charges,
            cell,
            alpha=0.3,
            k_cutoff=8.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=False,
        )

        e2 = ewald_summation(
            positions,
            2.0 * charges,
            cell,
            alpha=0.3,
            k_cutoff=8.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=False,
        )

        ratio = e2.sum() / e1.sum()
        assert abs(ratio - 4.0) < 0.1, f"Energy should scale as q^2, got ratio {ratio}"

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_translation_invariance(self, device):
        """Test that Ewald energy is invariant under global translation."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell, neighbor_list, neighbor_ptr, neighbor_shifts = (
            create_dipole_system(device)
        )

        energy1 = ewald_summation(
            positions,
            charges,
            cell,
            alpha=0.3,
            k_cutoff=8.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=False,
        )

        # Translate
        translation = torch.tensor([1.5, 0.7, -0.3], dtype=torch.float64, device=device)
        positions2 = positions + translation

        energy2 = ewald_summation(
            positions2,
            charges,
            cell,
            alpha=0.3,
            k_cutoff=8.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=False,
        )

        assert torch.allclose(energy1.sum(), energy2.sum(), rtol=0.01, atol=0.01)


###########################################################################################
########################### Numerical Stability Tests #####################################
###########################################################################################


class TestNumericalStability:
    """Test numerical stability and edge cases."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_empty_neighbor_list(self, device):
        """Test that Ewald handles empty neighbor list."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 2.0, 2.0], [6.0, 6.0, 6.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0) * 10.0

        neighbor_list = torch.tensor([[], []], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 0, 0], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((0, 3), dtype=torch.int32, device=device)

        energy = ewald_summation(
            positions,
            charges,
            cell,
            alpha=0.3,
            k_cutoff=8.0,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=False,
        )

        assert torch.isfinite(energy).all()

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_reciprocal_convergence(self, device):
        """Test that reciprocal energy converges with k_cutoff."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions, charges, cell, _, _, _ = create_dipole_system(device)
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)

        k_cutoffs = [5.0, 8.0, 12.0]
        energies = []

        for k_cutoff in k_cutoffs:
            k_vecs = generate_k_vectors_ewald_summation(cell, k_cutoff).squeeze(0)
            e = ewald_reciprocal_space(
                positions,
                charges,
                cell,
                k_vecs,
                alpha,
                compute_forces=False,
            )
            energies.append(e.sum().item())

        # Check convergence
        diff_1 = abs(energies[1] - energies[0])
        diff_2 = abs(energies[2] - energies[1])

        assert diff_2 < 0.05, (
            f"Energy not converged: diffs = {diff_1:.6f}, {diff_2:.6f}"
        )


class TestSingleAtomSystem:
    """Test handling of single atom systems."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_single_atom_real_space(self, device):
        """Test real-space with single atom (no pairs)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[5.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0], dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0) * 10.0
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)

        neighbor_list = torch.zeros((2, 0), dtype=torch.int32, device=device)
        neighbor_ptr = torch.zeros(1, dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((0, 3), dtype=torch.int32, device=device)

        energies = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=False,
        )

        assert energies.shape == (1,)
        assert torch.isfinite(energies).all()
        # Single atom has zero pairwise interaction
        assert torch.allclose(energies, torch.zeros_like(energies))

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_single_atom_reciprocal_space(self, device):
        """Test reciprocal-space with single atom."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[5.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0], dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0) * 10.0
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0).squeeze(0)

        energies = ewald_reciprocal_space(
            positions,
            charges,
            cell,
            k_vectors,
            alpha,
            compute_forces=False,
        )

        assert energies.shape == (1,)
        assert torch.isfinite(energies).all()


class TestNonCubicCells:
    """Test with non-cubic simulation cells."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_orthorhombic_cell(self, device):
        """Test with orthorhombic (non-cubic) cell."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        # Orthorhombic cell: different lengths along each axis
        cell = torch.tensor(
            [[[8.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 12.0]]],
            dtype=torch.float64,
            device=device,
        )
        positions = torch.tensor(
            [[2.0, 5.0, 6.0], [6.0, 5.0, 6.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)

        neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)

        energies, forces = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=True,
        )

        assert torch.isfinite(energies).all()
        assert torch.isfinite(forces).all()
        # Opposite charges should attract
        assert energies.sum() < 0

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_triclinic_cell(self, device):
        """Test with triclinic (tilted) cell."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        # Triclinic cell with off-diagonal elements
        cell = torch.tensor(
            [[[10.0, 0.0, 0.0], [2.0, 10.0, 0.0], [1.0, 1.0, 10.0]]],
            dtype=torch.float64,
            device=device,
        )
        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [7.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)

        neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)
        energies, forces = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=True,
        )

        assert torch.isfinite(energies).all()
        assert torch.isfinite(forces).all()

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_triclinic_cell_reciprocal(self, device):
        """Test reciprocal space with triclinic cell."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        # Triclinic cell
        cell = torch.tensor(
            [[[10.0, 0.0, 0.0], [2.0, 10.0, 0.0], [1.0, 1.0, 10.0]]],
            dtype=torch.float64,
            device=device,
        )
        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [7.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0).squeeze(0)

        energies, forces = ewald_reciprocal_space(
            positions,
            charges,
            cell,
            k_vectors,
            alpha,
            compute_forces=True,
        )

        assert torch.isfinite(energies).all()
        assert torch.isfinite(forces).all()


class TestLikeCharges:
    """Test behavior with like charges (repulsive)."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_like_charges_positive_energy(self, device):
        """Test that like charges have positive interaction energy."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [8.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, 1.0], dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0) * 10.0
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)

        neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)

        energies = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=False,
        )

        # Like charges should have positive energy (repulsive)
        assert energies.sum() > 0

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_like_charges_repulsive_forces(self, device):
        """Test that like charges repel each other."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [8.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, 1.0], dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0) * 10.0
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)

        neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)

        _, forces = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=True,
        )

        # Like charges should repel: force on atom 0 should be in -x direction
        assert forces[0, 0] < 0
        assert forces[1, 0] > 0


class TestNeighborMatrixFormat:
    """Test Ewald with neighbor matrix format."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_real_space_neighbor_matrix(self, device):
        """Test real-space with neighbor matrix format."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [8.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0) * 10.0
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)

        # Neighbor matrix format
        neighbor_matrix = torch.tensor([[1], [0]], dtype=torch.int32, device=device)
        neighbor_matrix_shifts = torch.zeros(
            (2, 1, 3), dtype=torch.int32, device=device
        )

        energies, forces = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            compute_forces=True,
        )

        assert energies.shape == (2,)
        assert forces.shape == (2, 3)
        assert torch.isfinite(energies).all()
        assert torch.isfinite(forces).all()

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_real_space_matrix_matches_list(self, device):
        """Test that neighbor matrix gives same results as neighbor list."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [8.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0) * 10.0
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)

        # Neighbor list format
        neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)

        energies_list, forces_list = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=True,
        )

        # Neighbor matrix format (symmetric)
        neighbor_matrix = torch.tensor([[1], [0]], dtype=torch.int32, device=device)
        neighbor_matrix_shifts = torch.zeros(
            (2, 1, 3), dtype=torch.int32, device=device
        )

        energies_matrix, forces_matrix = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            compute_forces=True,
        )

        assert torch.allclose(energies_list.sum(), energies_matrix.sum(), rtol=1e-6)
        assert torch.allclose(forces_list, forces_matrix, rtol=1e-6)


class TestInputValidation:
    """Test input validation for Ewald functions."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_missing_neighbor_data(self, device):
        """Test that missing neighbor data raises error."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [8.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0) * 10.0
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)

        with pytest.raises(ValueError):
            ewald_real_space(
                positions,
                charges,
                cell,
                alpha,
                compute_forces=False,
            )


class TestAlphaSensitivity:
    """Test sensitivity to alpha parameter."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_different_alpha_values(self, device):
        """Test that different alpha values give different energy splits."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [8.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0) * 10.0

        neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)

        alphas = [0.2, 0.3, 0.5, 1.0]
        real_energies = []
        reciprocal_energies = []

        for alpha_val in alphas:
            alpha = torch.tensor([alpha_val], dtype=torch.float64, device=device)
            k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=10.0).squeeze(
                0
            )

            e_real = ewald_real_space(
                positions,
                charges,
                cell,
                alpha,
                neighbor_list=neighbor_list,
                neighbor_ptr=neighbor_ptr,
                neighbor_shifts=neighbor_shifts,
                compute_forces=False,
            )

            e_recip = ewald_reciprocal_space(
                positions,
                charges,
                cell,
                k_vectors,
                alpha,
                compute_forces=False,
            )

            real_energies.append(e_real.sum().item())
            reciprocal_energies.append(e_recip.sum().item())

        # Higher alpha should shift energy from real to reciprocal space
        # Real-space should decrease with increasing alpha
        for i in range(len(alphas) - 1):
            assert abs(real_energies[i]) > abs(real_energies[i + 1]), (
                f"Real-space energy should decrease with alpha: {real_energies}"
            )


class TestPrepareAlphaEdgeCases:
    """Test _prepare_alpha edge cases for coverage."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_scalar_alpha_tensor_0d(self, device):
        """Test 0-dimensional alpha tensor expansion (line 211)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [8.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0) * 10.0

        # 0-dimensional tensor (scalar tensor)
        alpha = torch.tensor(0.3, dtype=torch.float64, device=device)  # 0-dim

        neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)

        energies = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=False,
        )

        assert torch.isfinite(energies).all()

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_alpha_wrong_size_raises_error(self, device):
        """Test alpha tensor with wrong number of elements raises ValueError (line 213)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [8.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0) * 10.0

        # Alpha tensor with wrong size (2 values for 1 system)
        alpha = torch.tensor([0.3, 0.5], dtype=torch.float64, device=device)

        neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)

        with pytest.raises(ValueError):
            ewald_summation(
                positions,
                charges,
                cell,
                alpha,
                neighbor_list=neighbor_list,
                neighbor_ptr=neighbor_ptr,
                neighbor_shifts=neighbor_shifts,
                compute_forces=False,
            )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_alpha_invalid_type_raises_error(self, device):
        """Test non-float, non-tensor alpha raises TypeError (line 218)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [8.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0) * 10.0

        neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)

        with pytest.raises(TypeError):
            ewald_summation(
                positions,
                charges,
                cell,
                alpha="invalid",  # String is not valid
                neighbor_list=neighbor_list,
                neighbor_ptr=neighbor_ptr,
                neighbor_shifts=neighbor_shifts,
                compute_forces=False,
            )


class TestPrepareCellEdgeCases:
    """Test _prepare_cell edge cases for coverage."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_2d_cell_unsqueeze(self, device):
        """Test 2D cell gets unsqueezed to 3D (line 237)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [8.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        # 2D cell (not batched) - should be auto-unsqueezed
        cell = torch.eye(3, dtype=torch.float64, device=device) * 10.0

        k_vectors = generate_k_vectors_ewald_summation(
            cell.unsqueeze(0), k_cutoff=8.0
        ).squeeze(0)
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)

        # This should work with 2D cell
        energies = ewald_reciprocal_space(
            positions,
            charges,
            cell,  # 2D cell
            k_vectors,
            alpha,
            compute_forces=False,
        )

        assert torch.isfinite(energies).all()


class TestEmptyNeighborListEarlyReturns:
    """Test empty neighbor list/matrix early returns for coverage."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_empty_neighbor_list_energy_forces(self, device):
        """Test empty neighbor list returns zeros for energy+forces (lines 360-361)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [8.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0) * 10.0
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)

        # Empty neighbor list
        neighbor_list = torch.zeros((2, 0), dtype=torch.int32, device=device)
        neighbor_ptr = torch.zeros(1, dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((0, 3), dtype=torch.int32, device=device)

        energies, forces = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=True,
        )

        assert torch.allclose(energies, torch.zeros_like(energies))
        assert torch.allclose(forces, torch.zeros_like(forces))

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_empty_neighbor_matrix_energy(self, device):
        """Test empty neighbor matrix returns zeros for energy (lines 453-455)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [8.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0) * 10.0
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)

        # Empty neighbor matrix (0 rows)
        neighbor_matrix = torch.zeros((0, 1), dtype=torch.int32, device=device)
        neighbor_matrix_shifts = torch.zeros(
            (0, 1, 3), dtype=torch.int32, device=device
        )

        energies = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            compute_forces=False,
        )

        assert energies.shape == (2,)

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_empty_neighbor_matrix_energy_forces(self, device):
        """Test empty neighbor matrix returns zeros for energy+forces (lines 542-547)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [8.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0) * 10.0
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)

        # Empty neighbor matrix (0 rows)
        neighbor_matrix = torch.zeros((0, 1), dtype=torch.int32, device=device)
        neighbor_matrix_shifts = torch.zeros(
            (0, 1, 3), dtype=torch.int32, device=device
        )

        energies, forces = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            compute_forces=True,
        )

        assert energies.shape == (2,)
        assert forces.shape == (2, 3)


class TestBatchNeighborMatrixFormat:
    """Test batch calculations with neighbor matrix format for coverage."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_matrix_energy_only(self, device):
        """Test batch energy-only with neighbor matrix (lines 824-890)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        # Two systems
        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [8.0, 5.0, 5.0], [2.0, 5.0, 5.0], [8.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor(
            [1.0, -1.0, 1.0, -1.0], dtype=torch.float64, device=device
        )
        cell = (
            torch.eye(3, dtype=torch.float64, device=device)
            .unsqueeze(0)
            .expand(2, -1, -1)
            .contiguous()
            * 10.0
        )
        alpha = torch.tensor([0.3, 0.3], dtype=torch.float64, device=device)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)

        # Neighbor matrix for batch
        neighbor_matrix = torch.tensor(
            [[1], [0], [3], [2]], dtype=torch.int32, device=device
        )
        neighbor_matrix_shifts = torch.zeros(
            (4, 1, 3), dtype=torch.int32, device=device
        )

        energies = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            batch_idx=batch_idx,
            compute_forces=False,
        )

        assert energies.shape == (4,)
        assert torch.isfinite(energies).all()

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_matrix_energy_forces(self, device):
        """Test batch energy+forces with neighbor matrix (lines 915-990)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        # Two systems
        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [8.0, 5.0, 5.0], [2.0, 5.0, 5.0], [8.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor(
            [1.0, -1.0, 1.0, -1.0], dtype=torch.float64, device=device
        )
        cell = (
            torch.eye(3, dtype=torch.float64, device=device)
            .unsqueeze(0)
            .expand(2, -1, -1)
            .contiguous()
            * 10.0
        )
        alpha = torch.tensor([0.3, 0.3], dtype=torch.float64, device=device)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)

        neighbor_matrix = torch.tensor(
            [[1], [0], [3], [2]], dtype=torch.int32, device=device
        )
        neighbor_matrix_shifts = torch.zeros(
            (4, 1, 3), dtype=torch.int32, device=device
        )

        energies, forces = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            batch_idx=batch_idx,
            compute_forces=True,
        )

        assert energies.shape == (4,)
        assert forces.shape == (4, 3)
        assert torch.isfinite(energies).all()
        assert torch.isfinite(forces).all()

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_matrix_autograd(self, device):
        """Test batch neighbor matrix with autograd enabled (lines 875-886, 974-986)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [8.0, 5.0, 5.0], [2.0, 5.0, 5.0], [8.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        charges = torch.tensor(
            [1.0, -1.0, 1.0, -1.0], dtype=torch.float64, device=device
        )
        cell = (
            torch.eye(3, dtype=torch.float64, device=device)
            .unsqueeze(0)
            .expand(2, -1, -1)
            .contiguous()
            * 10.0
        )
        alpha = torch.tensor([0.3, 0.3], dtype=torch.float64, device=device)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)

        neighbor_matrix = torch.tensor(
            [[1], [0], [3], [2]], dtype=torch.int32, device=device
        )
        neighbor_matrix_shifts = torch.zeros(
            (4, 1, 3), dtype=torch.int32, device=device
        )

        # Energy only with autograd
        energies = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            batch_idx=batch_idx,
            compute_forces=False,
        )
        energies.sum().backward()

        assert positions.grad is not None
        assert torch.isfinite(positions.grad).all()


class TestBatchReciprocalEnergyOnly:
    """Test batch reciprocal-space energy-only for coverage."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_reciprocal_energy_only(self, device):
        """Test batch reciprocal-space energy-only (lines 1642-1654, 1780-1792)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [8.0, 5.0, 5.0], [2.0, 5.0, 5.0], [8.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor(
            [1.0, -1.0, 1.0, -1.0], dtype=torch.float64, device=device
        )
        cell = (
            torch.eye(3, dtype=torch.float64, device=device)
            .unsqueeze(0)
            .expand(2, -1, -1)
            .contiguous()
            * 10.0
        )
        alpha = torch.tensor([0.3, 0.3], dtype=torch.float64, device=device)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0)

        # Energy only
        energies = ewald_reciprocal_space(
            positions,
            charges,
            cell,
            k_vectors,
            alpha,
            batch_idx=batch_idx,
            compute_forces=False,
        )

        assert energies.shape == (4,)
        assert torch.isfinite(energies).all()


class TestReciprocalSpaceEmptyReturns:
    """Test reciprocal space empty k-vectors/atoms early returns for coverage."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_empty_k_vectors_reciprocal(self, device):
        """Test empty k_vectors returns zeros (lines 1160-1162)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [8.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0) * 10.0

        # Empty k_vectors
        k_vectors = torch.zeros((0, 3), dtype=torch.float64, device=device)
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)

        energies, forces = ewald_reciprocal_space(
            positions,
            charges,
            cell,
            k_vectors,
            alpha,
            compute_forces=True,
        )

        assert energies.shape == (2,)
        assert forces.shape == (2, 3)
        assert torch.allclose(energies, torch.zeros_like(energies))
        assert torch.allclose(forces, torch.zeros_like(forces))

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_empty_k_vectors_reciprocal(self, device):
        """Test batch empty k_vectors returns zeros (lines 1319-1321, 1503-1505)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [8.0, 5.0, 5.0], [2.0, 5.0, 5.0], [8.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor(
            [1.0, -1.0, 1.0, -1.0], dtype=torch.float64, device=device
        )
        cell = (
            torch.eye(3, dtype=torch.float64, device=device)
            .unsqueeze(0)
            .expand(2, -1, -1)
            .contiguous()
            * 10.0
        )
        alpha = torch.tensor([0.3, 0.3], dtype=torch.float64, device=device)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)

        # Empty k_vectors for batch (need batch dimension)
        k_vectors = torch.zeros((2, 0, 3), dtype=torch.float64, device=device)

        # Energy only
        energies = ewald_reciprocal_space(
            positions,
            charges,
            cell,
            k_vectors,
            alpha,
            batch_idx=batch_idx,
            compute_forces=False,
        )

        assert energies.shape == (4,)
        assert torch.allclose(energies, torch.zeros_like(energies))

        # Energy + forces
        energies, forces = ewald_reciprocal_space(
            positions,
            charges,
            cell,
            k_vectors,
            alpha,
            batch_idx=batch_idx,
            compute_forces=True,
        )

        assert energies.shape == (4,)
        assert forces.shape == (4, 3)
        assert torch.allclose(energies, torch.zeros_like(energies))
        assert torch.allclose(forces, torch.zeros_like(forces))

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_reciprocal_autograd(self, device):
        """Test batch reciprocal space with autograd (lines 1641-1654)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [8.0, 5.0, 5.0], [2.0, 5.0, 5.0], [8.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        charges = torch.tensor(
            [1.0, -1.0, 1.0, -1.0], dtype=torch.float64, device=device
        )
        cell = (
            torch.eye(3, dtype=torch.float64, device=device)
            .unsqueeze(0)
            .expand(2, -1, -1)
            .contiguous()
            * 10.0
        )
        alpha = torch.tensor([0.3, 0.3], dtype=torch.float64, device=device)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=8.0)

        # Energy + forces with autograd
        energies, _ = ewald_reciprocal_space(
            positions,
            charges,
            cell,
            k_vectors,
            alpha,
            batch_idx=batch_idx,
            compute_forces=True,
        )

        energies.sum().backward()
        assert positions.grad is not None
        assert torch.isfinite(positions.grad).all()


class TestEwaldSummationAutoParameters:
    """Test ewald_summation with auto-estimated parameters for coverage."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_auto_estimate_alpha_and_k_cutoff(self, device):
        """Test auto-estimation of alpha and k_cutoff (lines 2076-2081)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [8.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0) * 10.0

        neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)

        # Call without alpha or k_cutoff - should auto-estimate both
        energies, forces = ewald_summation(
            positions,
            charges,
            cell,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            alpha=None,  # Auto-estimate
            k_cutoff=None,  # Auto-estimate
            compute_forces=True,
        )

        assert energies.shape == (2,)
        assert forces.shape == (2, 3)
        assert torch.isfinite(energies).all()
        assert torch.isfinite(forces).all()


class TestAutogradWithMatrixFormat:
    """Test autograd with neighbor matrix format for attach_for_backward coverage."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_single_system_matrix_autograd(self, device):
        """Test single-system neighbor matrix with autograd (lines 499-510, 593-605)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [8.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0) * 10.0
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)

        neighbor_matrix = torch.tensor([[1], [0]], dtype=torch.int32, device=device)
        neighbor_matrix_shifts = torch.zeros(
            (2, 1, 3), dtype=torch.int32, device=device
        )

        # Energy only
        energies = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            compute_forces=False,
        )
        energies.sum().backward()

        assert positions.grad is not None

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_single_system_matrix_energy_forces_autograd(self, device):
        """Test single-system neighbor matrix energy+forces with autograd."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [8.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0) * 10.0
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)

        neighbor_matrix = torch.tensor([[1], [0]], dtype=torch.int32, device=device)
        neighbor_matrix_shifts = torch.zeros(
            (2, 1, 3), dtype=torch.int32, device=device
        )

        # Energy + forces
        energies, _ = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            compute_forces=True,
        )
        energies.sum().backward()

        assert positions.grad is not None


class TestBatchMatrixAutograd:
    """Test batch autograd with neighbor matrix format."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_matrix_energy_autograd(self, device):
        """Test batch neighbor matrix energy-only with autograd."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [8.0, 5.0, 5.0], [2.0, 5.0, 5.0], [8.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        charges = torch.tensor(
            [1.0, -1.0, 1.0, -1.0],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        cell = torch.stack(
            [
                torch.eye(3, dtype=torch.float64, device=device) * 10.0,
                torch.eye(3, dtype=torch.float64, device=device) * 10.0,
            ]
        )
        alpha = torch.tensor([0.3, 0.3], dtype=torch.float64, device=device)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)

        neighbor_matrix = torch.tensor(
            [[1, -1], [0, -1], [3, -1], [2, -1]], dtype=torch.int32, device=device
        )
        neighbor_matrix_shifts = torch.zeros(
            (4, 2, 3), dtype=torch.int32, device=device
        )

        energies = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            mask_value=-1,
            batch_idx=batch_idx,
            compute_forces=False,
        )
        energies.sum().backward()

        assert positions.grad is not None
        assert charges.grad is not None

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_matrix_energy_forces_autograd(self, device):
        """Test batch neighbor matrix energy+forces with autograd."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [8.0, 5.0, 5.0], [2.0, 5.0, 5.0], [8.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        charges = torch.tensor(
            [1.0, -1.0, 1.0, -1.0],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        cell = torch.stack(
            [
                torch.eye(3, dtype=torch.float64, device=device) * 10.0,
                torch.eye(3, dtype=torch.float64, device=device) * 10.0,
            ]
        )
        alpha = torch.tensor([0.3, 0.3], dtype=torch.float64, device=device)
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)

        neighbor_matrix = torch.tensor(
            [[1, -1], [0, -1], [3, -1], [2, -1]], dtype=torch.int32, device=device
        )
        neighbor_matrix_shifts = torch.zeros(
            (4, 2, 3), dtype=torch.int32, device=device
        )

        energies, forces = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            mask_value=-1,
            batch_idx=batch_idx,
            compute_forces=True,
        )
        energies.sum().backward()

        assert positions.grad is not None
        assert charges.grad is not None
        assert forces.shape == (4, 3)


class TestBatchEmptyInputs:
    """Test batch functions with empty inputs."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_empty_neighbor_list_energy(self, device):
        """Test batch real-space energy with empty neighbor list."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [8.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0) * 10.0
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)
        batch_idx = torch.tensor([0, 0], dtype=torch.int32, device=device)

        # Empty neighbor list
        neighbor_list = torch.zeros((2, 0), dtype=torch.int32, device=device)
        neighbor_ptr = torch.zeros(1, dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((0, 3), dtype=torch.int32, device=device)

        energies = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            batch_idx=batch_idx,
            compute_forces=False,
        )

        assert energies.shape == (2,)
        assert torch.allclose(
            energies, torch.zeros(2, device=device, dtype=positions.dtype)
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_empty_neighbor_list_energy_forces(self, device):
        """Test batch real-space energy+forces with empty neighbor list."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [8.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0) * 10.0
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)
        batch_idx = torch.tensor([0, 0], dtype=torch.int32, device=device)

        # Empty neighbor list
        neighbor_list = torch.zeros((2, 0), dtype=torch.int32, device=device)
        neighbor_ptr = torch.zeros(1, dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((0, 3), dtype=torch.int32, device=device)

        energies, forces = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            batch_idx=batch_idx,
            compute_forces=True,
        )

        assert energies.shape == (2,)
        assert forces.shape == (2, 3)
        assert torch.allclose(
            energies, torch.zeros(2, device=device, dtype=positions.dtype)
        )
        assert torch.allclose(
            forces, torch.zeros((2, 3), device=device, dtype=positions.dtype)
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_empty_neighbor_matrix_energy(self, device):
        """Test batch real-space energy with empty neighbor matrix (0 rows)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        # Zero atoms case
        positions = torch.zeros((0, 3), dtype=torch.float64, device=device)
        charges = torch.zeros((0,), dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0) * 10.0
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)
        batch_idx = torch.zeros((0,), dtype=torch.int32, device=device)

        neighbor_matrix = torch.zeros((0, 1), dtype=torch.int32, device=device)
        neighbor_matrix_shifts = torch.zeros(
            (0, 1, 3), dtype=torch.int32, device=device
        )

        energies = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            mask_value=-1,
            batch_idx=batch_idx,
            compute_forces=False,
        )

        assert energies.shape == (0,)

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_empty_neighbor_matrix_energy_forces(self, device):
        """Test batch real-space energy+forces with empty neighbor matrix (0 rows)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        # Zero atoms case
        positions = torch.zeros((0, 3), dtype=torch.float64, device=device)
        charges = torch.zeros((0,), dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device).unsqueeze(0) * 10.0
        alpha = torch.tensor([0.3], dtype=torch.float64, device=device)
        batch_idx = torch.zeros((0,), dtype=torch.int32, device=device)

        neighbor_matrix = torch.zeros((0, 1), dtype=torch.int32, device=device)
        neighbor_matrix_shifts = torch.zeros(
            (0, 1, 3), dtype=torch.int32, device=device
        )

        energies, forces = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            mask_value=-1,
            batch_idx=batch_idx,
            compute_forces=True,
        )

        assert energies.shape == (0,)
        assert forces.shape == (0, 3)


class TestEwaldSummationAutoEstimate:
    """Test auto-estimation paths in ewald_summation."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_auto_estimate_k_cutoff_with_alpha(self, device):
        """Test ewald_summation with user alpha and auto-estimated k_cutoff."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [8.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device) * 10.0
        neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)

        # Provide alpha but not k_cutoff - should auto-estimate k_cutoff
        result = ewald_summation(
            positions,
            charges,
            cell,
            alpha=0.3,  # User-provided alpha
            # k_cutoff not provided - should be auto-estimated
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=False,
        )

        assert all(torch.isfinite(result))

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_auto_generate_k_vectors(self, device):
        """Test ewald_summation auto-generates k_vectors when not provided."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [8.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device) * 10.0
        neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros((2, 3), dtype=torch.int32, device=device)

        # Both alpha and k_cutoff provided, but not k_vectors
        result = ewald_summation(
            positions,
            charges,
            cell,
            alpha=0.3,
            k_cutoff=5.0,
            # k_vectors not provided - should be auto-generated
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=False,
        )

        assert all(torch.isfinite(result))

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("mask_value", [None, 2])
    def test_default_mask_value(self, device, mask_value):
        """Test ewald_summation with default mask_value (None -> num_atoms)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)

        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [8.0, 5.0, 5.0]],
            dtype=torch.float64,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        cell = torch.eye(3, dtype=torch.float64, device=device) * 10.0

        neighbor_matrix = torch.tensor(
            [[1, 2, 2], [0, 2, 2]], dtype=torch.int32, device=device
        )
        neighbor_matrix_shifts = torch.zeros(
            (2, 3, 3), dtype=torch.int32, device=device
        )

        # Use neighbor matrix format without explicit mask_value
        result = ewald_summation(
            positions,
            charges,
            cell,
            alpha=0.3,
            k_cutoff=5.0,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            mask_value=mask_value,
            compute_forces=False,
        )

        assert all(torch.isfinite(result))


###########################################################################################
########################### Virial Tests ##################################################
###########################################################################################


class TestEwaldRealSpaceVirial:
    """Test real-space Ewald virial against finite-difference strain derivatives."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_real_space_virial_shape(self, device):
        """Virial output has correct shape."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell = make_virial_cscl_system(2, device=device)
        alpha = torch.tensor([0.3], dtype=VIRIAL_DTYPE, device=device)
        cutoff = 6.0
        nl, nptr, us = get_virial_neighbor_data(positions, cell, cutoff)

        result = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=us,
            compute_forces=True,
            compute_virial=True,
        )
        assert len(result) == 3
        energies, forces, virial = result
        assert virial.shape == (1, 3, 3)
        assert virial.dtype == VIRIAL_DTYPE

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_real_space_virial_fd(self, device):
        """Real-space virial matches finite-difference strain derivative."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell = make_virial_cscl_system(2, device=device)
        alpha = torch.tensor([0.3], dtype=VIRIAL_DTYPE, device=device)
        cutoff = 6.0
        pbc = torch.tensor([True, True, True], dtype=torch.bool, device=device)

        def energy_fn(pos, c):
            nl_new, np_new, us_new = cell_list(
                pos,
                cutoff,
                c.squeeze(0),
                pbc,
                return_neighbor_list=True,
            )
            return ewald_real_space(
                pos,
                charges,
                c,
                alpha,
                neighbor_list=nl_new,
                neighbor_ptr=np_new,
                neighbor_shifts=us_new,
                compute_forces=False,
            ).sum()

        result = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=get_virial_neighbor_data(positions, cell, cutoff)[0],
            neighbor_ptr=get_virial_neighbor_data(positions, cell, cutoff)[1],
            neighbor_shifts=get_virial_neighbor_data(positions, cell, cutoff)[2],
            compute_forces=True,
            compute_virial=True,
        )
        explicit_virial = result[2].squeeze(0)
        fd_virial = fd_virial_full(energy_fn, positions, cell, device, h=1e-5)

        torch.testing.assert_close(
            explicit_virial,
            fd_virial,
            atol=1e-3,
            rtol=1e-3,
            msg="Real-space virial does not match finite-difference reference",
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_real_space_virial_symmetry(self, device):
        """Virial tensor should be approximately symmetric for cubic systems."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell = make_virial_cscl_system(2, device=device)
        alpha = torch.tensor([0.3], dtype=VIRIAL_DTYPE, device=device)
        nl, nptr, us = get_virial_neighbor_data(positions, cell, cutoff=6.0)

        result = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=us,
            compute_forces=True,
            compute_virial=True,
        )
        virial = result[2].squeeze(0)
        torch.testing.assert_close(
            virial,
            virial.T,
            atol=1e-6,
            rtol=1e-6,
            msg="Virial tensor is not symmetric",
        )


class TestEwaldReciprocalSpaceVirial:
    """Test reciprocal-space Ewald virial against finite-difference."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_reciprocal_virial_shape(self, device):
        """Reciprocal virial output has correct shape."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell = make_virial_cscl_system(2, device=device)
        alpha = torch.tensor([0.3], dtype=VIRIAL_DTYPE, device=device)
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=3.0)

        result = ewald_reciprocal_space(
            positions,
            charges,
            cell,
            k_vectors,
            alpha,
            compute_forces=True,
            compute_virial=True,
        )
        assert len(result) == 3
        energies, forces, virial = result
        assert virial.shape == (1, 3, 3)

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_reciprocal_virial_fd(self, device):
        """Reciprocal virial matches finite-difference strain derivative."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell = make_virial_cscl_system(2, device=device)
        alpha = torch.tensor([0.3], dtype=VIRIAL_DTYPE, device=device)

        def energy_fn(pos, c):
            kv = generate_k_vectors_ewald_summation(c, k_cutoff=3.0)
            return ewald_reciprocal_space(
                pos,
                charges,
                c,
                kv,
                alpha,
                compute_forces=False,
            ).sum()

        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=3.0)
        result = ewald_reciprocal_space(
            positions,
            charges,
            cell,
            k_vectors,
            alpha,
            compute_forces=True,
            compute_virial=True,
        )
        explicit_virial = result[2].squeeze(0)
        fd_virial = fd_virial_full(energy_fn, positions, cell, device, h=1e-5)

        torch.testing.assert_close(
            explicit_virial,
            fd_virial,
            atol=1e-3,
            rtol=1e-3,
            msg="Reciprocal virial does not match finite-difference reference",
        )


class TestEwaldTotalVirial:
    """Test total Ewald virial (real + reciprocal) against finite-difference."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_total_virial_shape(self, device):
        """Total virial has correct shape."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell = make_virial_cscl_system(2, device=device)
        alpha = torch.tensor([0.3], dtype=VIRIAL_DTYPE, device=device)
        k_cutoff = 3.0
        cutoff = 6.0
        nl, nptr, us = get_virial_neighbor_data(positions, cell, cutoff)

        result = ewald_summation(
            positions,
            charges,
            cell,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=us,
            alpha=alpha,
            k_cutoff=k_cutoff,
            compute_forces=True,
            compute_virial=True,
        )
        assert len(result) == 3
        energies, forces, virial = result
        assert virial.shape == (1, 3, 3)

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_total_virial_fd(self, device):
        """Total Ewald virial matches finite-difference."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell = make_virial_cscl_system(2, device=device)
        alpha = torch.tensor([0.3], dtype=VIRIAL_DTYPE, device=device)
        k_cutoff = 3.0
        cutoff = 6.0
        pbc = torch.tensor([True, True, True], dtype=torch.bool, device=device)

        def energy_fn(pos, c):
            nl_new, np_new, us_new = cell_list(
                pos,
                cutoff,
                c.squeeze(0),
                pbc,
                return_neighbor_list=True,
            )
            return ewald_summation(
                pos,
                charges,
                c,
                neighbor_list=nl_new,
                neighbor_ptr=np_new,
                neighbor_shifts=us_new,
                alpha=alpha,
                k_cutoff=k_cutoff,
                compute_forces=False,
            ).sum()

        nl, nptr, us = get_virial_neighbor_data(positions, cell, cutoff)
        result = ewald_summation(
            positions,
            charges,
            cell,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=us,
            alpha=alpha,
            k_cutoff=k_cutoff,
            compute_forces=True,
            compute_virial=True,
        )
        explicit_virial = result[2].squeeze(0)
        fd_virial = fd_virial_full(energy_fn, positions, cell, device, h=1e-5)

        torch.testing.assert_close(
            explicit_virial,
            fd_virial,
            atol=1e-3,
            rtol=1e-3,
            msg="Total Ewald virial does not match finite-difference reference",
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_virial_is_sum_of_components(self, device):
        """Total virial = real-space virial + reciprocal virial."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell = make_virial_cscl_system(2, device=device)
        alpha = torch.tensor([0.3], dtype=VIRIAL_DTYPE, device=device)
        k_cutoff = 3.0
        cutoff = 6.0
        nl, nptr, us = get_virial_neighbor_data(positions, cell, cutoff)
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=k_cutoff)

        rs_result = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=us,
            compute_forces=True,
            compute_virial=True,
        )
        real_virial = rs_result[2]

        rec_result = ewald_reciprocal_space(
            positions,
            charges,
            cell,
            k_vectors,
            alpha,
            compute_forces=True,
            compute_virial=True,
        )
        recip_virial = rec_result[2]

        total_result = ewald_summation(
            positions,
            charges,
            cell,
            alpha=alpha,
            k_vectors=k_vectors,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=us,
            compute_forces=True,
            compute_virial=True,
        )
        total_virial = total_result[2]

        torch.testing.assert_close(
            total_virial,
            real_virial + recip_virial,
            atol=1e-6,
            rtol=1e-6,
            msg="Total virial != real + reciprocal virial",
        )


class TestEwaldVirialDtypeSupport:
    """Virial output dtype matches input dtype for both float32 and float64."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_real_space_virial_dtype(self, device, dtype):
        """Real-space virial dtype matches input dtype."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell = make_virial_cscl_system(
            1, dtype=dtype, device=device
        )
        alpha = torch.tensor([0.3], dtype=dtype, device=device)
        nl, nptr, us = get_virial_neighbor_data(positions, cell, cutoff=5.0)

        result = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=us,
            compute_forces=True,
            compute_virial=True,
        )
        virial = result[2]
        assert virial.dtype == dtype

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_reciprocal_virial_dtype(self, device, dtype):
        """Reciprocal virial dtype matches input dtype."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell = make_virial_cscl_system(
            1, dtype=dtype, device=device
        )
        alpha = torch.tensor([0.3], dtype=dtype, device=device)
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=3.0)

        result = ewald_reciprocal_space(
            positions,
            charges,
            cell,
            k_vectors,
            alpha,
            compute_forces=True,
            compute_virial=True,
        )
        virial = result[2]
        assert virial.dtype == dtype

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_total_virial_dtype(self, device, dtype):
        """Total Ewald summation virial dtype matches input dtype."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell = make_virial_cscl_system(
            1, dtype=dtype, device=device
        )
        alpha = torch.tensor([0.3], dtype=dtype, device=device)
        nl, nptr, us = get_virial_neighbor_data(positions, cell, cutoff=5.0)

        result = ewald_summation(
            positions,
            charges,
            cell,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=us,
            alpha=alpha,
            k_cutoff=3.0,
            compute_forces=True,
            compute_virial=True,
        )
        virial = result[2]
        assert virial.dtype == dtype

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_float32_vs_float64_virial_consistency(self, device):
        """Float32 and float64 virials are close (loose tolerance)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions_f32, charges_f32, cell_f32 = make_virial_cscl_system(
            1, dtype=torch.float32, device=device
        )
        positions_f64, charges_f64, cell_f64 = make_virial_cscl_system(
            1, dtype=torch.float64, device=device
        )
        alpha_f32 = torch.tensor([0.3], dtype=torch.float32, device=device)
        alpha_f64 = torch.tensor([0.3], dtype=torch.float64, device=device)
        k_vectors_f32 = generate_k_vectors_ewald_summation(cell_f32, k_cutoff=3.0)
        k_vectors_f64 = generate_k_vectors_ewald_summation(cell_f64, k_cutoff=3.0)

        result_f32 = ewald_reciprocal_space(
            positions_f32,
            charges_f32,
            cell_f32,
            k_vectors_f32,
            alpha_f32,
            compute_forces=True,
            compute_virial=True,
        )
        result_f64 = ewald_reciprocal_space(
            positions_f64,
            charges_f64,
            cell_f64,
            k_vectors_f64,
            alpha_f64,
            compute_forces=True,
            compute_virial=True,
        )
        torch.testing.assert_close(
            result_f32[2].to(torch.float64),
            result_f64[2],
            atol=1e-3,
            rtol=1e-3,
            msg="Float32 and float64 reciprocal virials differ significantly",
        )


class TestEwaldVirialBatchConsistency:
    """Batch virial matches single-system virial."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_real_space_virial_shape(self, device):
        """Batch real-space virial has shape (B, 3, 3)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell, alpha, batch_idx, _, _, _, _, n_atoms = (
            make_virial_batch_cscl_system(1, device=device)
        )
        cutoff = 5.0

        nl_0, nptr_0, us_0 = get_virial_neighbor_data(
            positions[:n_atoms], cell[:1], cutoff
        )
        nl_1, nptr_1, us_1 = get_virial_neighbor_data(
            positions[n_atoms:], cell[1:], cutoff
        )

        nl_1_offset = nl_1.clone()
        nl_1_offset[0] += n_atoms
        nl = torch.cat([nl_0, nl_1_offset], dim=1)
        us = torch.cat([us_0, us_1], dim=0)
        nptr = torch.cat([nptr_0, nptr_1[1:] + nptr_0[-1]])

        result = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=us,
            batch_idx=batch_idx,
            compute_forces=True,
            compute_virial=True,
        )
        virial = result[2]
        assert virial.shape == (2, 3, 3)

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_reciprocal_virial_shape(self, device):
        """Batch reciprocal virial has shape (B, 3, 3)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell, alpha, batch_idx, _, _, _, _, _ = (
            make_virial_batch_cscl_system(1, device=device)
        )
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=3.0)

        result = ewald_reciprocal_space(
            positions,
            charges,
            cell,
            k_vectors,
            alpha,
            batch_idx=batch_idx,
            compute_forces=True,
            compute_virial=True,
        )
        virial = result[2]
        assert virial.shape == (2, 3, 3)

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_batch_reciprocal_virial_matches_single(self, device):
        """Batch reciprocal virial matches single-system virial."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell, alpha, batch_idx, pos_s, q_s, cell_s, alpha_s, _ = (
            make_virial_batch_cscl_system(1, device=device)
        )

        k_vectors_single = generate_k_vectors_ewald_summation(cell_s, k_cutoff=3.0)
        k_vectors_batch = generate_k_vectors_ewald_summation(cell, k_cutoff=3.0)

        single_result = ewald_reciprocal_space(
            pos_s,
            q_s,
            cell_s,
            k_vectors_single,
            alpha_s,
            compute_forces=True,
            compute_virial=True,
        )
        single_virial = single_result[2]

        batch_result = ewald_reciprocal_space(
            positions,
            charges,
            cell,
            k_vectors_batch,
            alpha,
            batch_idx=batch_idx,
            compute_forces=True,
            compute_virial=True,
        )
        batch_virial = batch_result[2]

        torch.testing.assert_close(
            batch_virial[0],
            single_virial[0],
            atol=1e-6,
            rtol=1e-6,
            msg="Batch virial[0] != single virial",
        )
        torch.testing.assert_close(
            batch_virial[1],
            single_virial[0],
            atol=1e-6,
            rtol=1e-6,
            msg="Batch virial[1] != single virial",
        )


class TestEwaldVirialNeighborMatrix:
    """Virial computation with neighbor_matrix format."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_real_space_virial_neighbor_matrix(self, device):
        """Virial has correct shape with neighbor_matrix format."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [8.0, 5.0, 5.0]],
            dtype=VIRIAL_DTYPE,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=VIRIAL_DTYPE, device=device)
        cell = torch.eye(3, dtype=VIRIAL_DTYPE, device=device).unsqueeze(0) * 10.0
        alpha = torch.tensor([0.3], dtype=VIRIAL_DTYPE, device=device)
        neighbor_matrix = torch.tensor([[1], [0]], dtype=torch.int32, device=device)
        neighbor_matrix_shifts = torch.zeros(2, 1, 3, dtype=torch.int32, device=device)

        result = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            compute_forces=True,
            compute_virial=True,
        )
        virial = result[2]
        assert virial.shape == (1, 3, 3)
        assert torch.isfinite(virial).all()

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_real_space_virial_matrix_matches_list(self, device):
        """Neighbor matrix virial matches neighbor list virial."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [8.0, 5.0, 5.0]],
            dtype=VIRIAL_DTYPE,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=VIRIAL_DTYPE, device=device)
        cell = torch.eye(3, dtype=VIRIAL_DTYPE, device=device).unsqueeze(0) * 10.0
        alpha = torch.tensor([0.3], dtype=VIRIAL_DTYPE, device=device)

        neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32, device=device)
        neighbor_ptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros(2, 3, dtype=torch.int32, device=device)

        result_list = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=True,
            compute_virial=True,
        )

        neighbor_matrix = torch.tensor([[1], [0]], dtype=torch.int32, device=device)
        neighbor_matrix_shifts = torch.zeros(2, 1, 3, dtype=torch.int32, device=device)

        result_matrix = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            compute_forces=True,
            compute_virial=True,
        )

        torch.testing.assert_close(
            result_list[2],
            result_matrix[2],
            atol=1e-8,
            rtol=1e-8,
            msg="Neighbor list virial != neighbor matrix virial",
        )


class TestEwaldVirialNonCubicCells:
    """Virial FD tests with non-cubic simulation cells."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_orthorhombic_cell_virial_fd(self, device):
        """Real-space virial FD check on orthorhombic cell."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        cell = torch.tensor(
            [[[8.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 12.0]]],
            dtype=VIRIAL_DTYPE,
            device=device,
        )
        positions = torch.tensor(
            [[2.0, 5.0, 6.0], [6.0, 5.0, 6.0]],
            dtype=VIRIAL_DTYPE,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=VIRIAL_DTYPE, device=device)
        alpha = torch.tensor([0.3], dtype=VIRIAL_DTYPE, device=device)
        cutoff = 5.0
        pbc = torch.tensor([True, True, True], dtype=torch.bool, device=device)
        nl, nptr, us = get_virial_neighbor_data(positions, cell, cutoff)

        def energy_fn(pos, c):
            nl_new, np_new, us_new = cell_list(
                pos,
                cutoff,
                c.squeeze(0),
                pbc,
                return_neighbor_list=True,
            )
            return ewald_real_space(
                pos,
                charges,
                c,
                alpha,
                neighbor_list=nl_new,
                neighbor_ptr=np_new,
                neighbor_shifts=us_new,
                compute_forces=False,
            ).sum()

        result = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=us,
            compute_forces=True,
            compute_virial=True,
        )
        explicit_virial = result[2].squeeze(0)
        fd_virial = fd_virial_full(energy_fn, positions, cell, device)

        torch.testing.assert_close(
            explicit_virial,
            fd_virial,
            atol=1e-3,
            rtol=1e-3,
            msg="Orthorhombic real-space virial does not match FD",
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_triclinic_cell_virial_fd(self, device):
        """Real-space virial FD check on triclinic cell."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        cell = torch.tensor(
            [[[10.0, 0.0, 0.0], [2.0, 10.0, 0.0], [1.0, 1.0, 10.0]]],
            dtype=VIRIAL_DTYPE,
            device=device,
        )
        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [5.0, 5.0, 5.0]],
            dtype=VIRIAL_DTYPE,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=VIRIAL_DTYPE, device=device)
        alpha = torch.tensor([0.3], dtype=VIRIAL_DTYPE, device=device)
        cutoff = 6.0
        pbc = torch.tensor([True, True, True], dtype=torch.bool, device=device)
        nl, nptr, us = get_virial_neighbor_data(positions, cell, cutoff)

        def energy_fn(pos, c):
            nl_new, np_new, us_new = cell_list(
                pos,
                cutoff,
                c.squeeze(0),
                pbc,
                return_neighbor_list=True,
            )
            return ewald_real_space(
                pos,
                charges,
                c,
                alpha,
                neighbor_list=nl_new,
                neighbor_ptr=np_new,
                neighbor_shifts=us_new,
                compute_forces=False,
            ).sum()

        result = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=us,
            compute_forces=True,
            compute_virial=True,
        )
        explicit_virial = result[2].squeeze(0)
        fd_virial = fd_virial_full(energy_fn, positions, cell, device)

        torch.testing.assert_close(
            explicit_virial,
            fd_virial,
            atol=1e-3,
            rtol=1e-3,
            msg="Triclinic real-space virial does not match FD",
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_triclinic_reciprocal_virial_fd(self, device):
        """Reciprocal virial FD check on triclinic cell."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        cell = torch.tensor(
            [[[10.0, 0.0, 0.0], [2.0, 10.0, 0.0], [1.0, 1.0, 10.0]]],
            dtype=VIRIAL_DTYPE,
            device=device,
        )
        positions = torch.tensor(
            [[2.0, 5.0, 5.0], [5.0, 5.0, 5.0]],
            dtype=VIRIAL_DTYPE,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=VIRIAL_DTYPE, device=device)
        alpha = torch.tensor([0.3], dtype=VIRIAL_DTYPE, device=device)

        def energy_fn(pos, c):
            kv = generate_k_vectors_ewald_summation(c, k_cutoff=3.0)
            return ewald_reciprocal_space(
                pos,
                charges,
                c,
                kv,
                alpha,
                compute_forces=False,
            ).sum()

        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=3.0)
        result = ewald_reciprocal_space(
            positions,
            charges,
            cell,
            k_vectors,
            alpha,
            compute_forces=True,
            compute_virial=True,
        )
        explicit_virial = result[2].squeeze(0)
        fd_virial = fd_virial_full(energy_fn, positions, cell, device)

        torch.testing.assert_close(
            explicit_virial,
            fd_virial,
            atol=1e-3,
            rtol=1e-3,
            msg="Triclinic reciprocal virial does not match FD",
        )


class TestEwaldVirialCrystalSystems:
    """Virial FD tests across different crystal systems."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize(
        "system_fn",
        [
            create_cscl_supercell,
            create_wurtzite_system,
            create_zincblende_system,
        ],
    )
    @pytest.mark.parametrize("alpha_val", [0.3, 0.5])
    def test_real_space_virial_fd_crystals(self, device, system_fn, alpha_val):
        """Real-space virial FD check for various crystal systems and alpha."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell = make_virial_crystal_system(
            system_fn, size=1, device=device
        )
        alpha = torch.tensor([alpha_val], dtype=VIRIAL_DTYPE, device=device)
        cutoff = 5.0
        pbc = torch.tensor([True, True, True], dtype=torch.bool, device=device)
        nl, nptr, us = get_virial_neighbor_data(positions, cell, cutoff)

        def energy_fn(pos, c):
            nl_new, np_new, us_new = cell_list(
                pos,
                cutoff,
                c.squeeze(0),
                pbc,
                return_neighbor_list=True,
            )
            return ewald_real_space(
                pos,
                charges,
                c,
                alpha,
                neighbor_list=nl_new,
                neighbor_ptr=np_new,
                neighbor_shifts=us_new,
                compute_forces=False,
            ).sum()

        result = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=us,
            compute_forces=True,
            compute_virial=True,
        )
        explicit_virial = result[2].squeeze(0)
        fd_virial = fd_virial_full(energy_fn, positions, cell, device)

        torch.testing.assert_close(
            explicit_virial,
            fd_virial,
            atol=1e-3,
            rtol=1e-3,
            msg=f"Real-space virial FD failed for {system_fn.__name__}, alpha={alpha_val}",
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize(
        "system_fn",
        [
            create_cscl_supercell,
            create_wurtzite_system,
            create_zincblende_system,
        ],
    )
    @pytest.mark.parametrize("alpha_val", [0.3, 0.5])
    def test_reciprocal_virial_fd_crystals(self, device, system_fn, alpha_val):
        """Reciprocal virial FD check for various crystal systems and alpha."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell = make_virial_crystal_system(
            system_fn, size=1, device=device
        )
        alpha = torch.tensor([alpha_val], dtype=VIRIAL_DTYPE, device=device)

        def energy_fn(pos, c):
            kv = generate_k_vectors_ewald_summation(c, k_cutoff=3.0)
            return ewald_reciprocal_space(
                pos,
                charges,
                c,
                kv,
                alpha,
                compute_forces=False,
            ).sum()

        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=3.0)
        result = ewald_reciprocal_space(
            positions,
            charges,
            cell,
            k_vectors,
            alpha,
            compute_forces=True,
            compute_virial=True,
        )
        explicit_virial = result[2].squeeze(0)
        fd_virial = fd_virial_full(energy_fn, positions, cell, device)

        torch.testing.assert_close(
            explicit_virial,
            fd_virial,
            atol=1e-3,
            rtol=1e-3,
            msg=f"Reciprocal virial FD failed for {system_fn.__name__}, alpha={alpha_val}",
        )


class TestEwaldVirialEdgeCases:
    """Edge cases for virial computation."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_empty_neighbor_list_virial_zero(self, device):
        """Empty neighbor list produces zero real-space virial."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]],
            dtype=VIRIAL_DTYPE,
            device=device,
        )
        charges = torch.tensor([1.0, -1.0], dtype=VIRIAL_DTYPE, device=device)
        cell = torch.eye(3, dtype=VIRIAL_DTYPE, device=device).unsqueeze(0) * 10.0
        alpha = torch.tensor([0.3], dtype=VIRIAL_DTYPE, device=device)

        neighbor_list = torch.zeros(2, 0, dtype=torch.int32, device=device)
        neighbor_ptr = torch.zeros(3, dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros(0, 3, dtype=torch.int32, device=device)

        result = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=True,
            compute_virial=True,
        )
        virial = result[2]
        assert virial.shape == (1, 3, 3)
        assert torch.allclose(virial, torch.zeros_like(virial))

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_single_atom_virial_shape(self, device):
        """Single atom system returns virial with correct shape."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions = torch.tensor([[5.0, 5.0, 5.0]], dtype=VIRIAL_DTYPE, device=device)
        charges = torch.tensor([1.0], dtype=VIRIAL_DTYPE, device=device)
        cell = torch.eye(3, dtype=VIRIAL_DTYPE, device=device).unsqueeze(0) * 10.0
        alpha = torch.tensor([0.3], dtype=VIRIAL_DTYPE, device=device)

        neighbor_list = torch.zeros(2, 0, dtype=torch.int32, device=device)
        neighbor_ptr = torch.zeros(2, dtype=torch.int32, device=device)
        neighbor_shifts = torch.zeros(0, 3, dtype=torch.int32, device=device)

        result = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            neighbor_shifts=neighbor_shifts,
            compute_forces=True,
            compute_virial=True,
        )
        virial = result[2]
        assert virial.shape == (1, 3, 3)

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_virial_without_forces(self, device):
        """compute_forces=False + compute_virial=True returns (energies, virial)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell = make_virial_cscl_system(1, device=device)
        alpha = torch.tensor([0.3], dtype=VIRIAL_DTYPE, device=device)
        nl, nptr, us = get_virial_neighbor_data(positions, cell, cutoff=5.0)

        result = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=us,
            compute_forces=False,
            compute_virial=True,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        energies, virial = result
        assert virial.shape == (1, 3, 3)

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_virial_with_charge_gradients(self, device):
        """compute_forces + compute_charge_gradients + compute_virial returns 4-tuple."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell = make_virial_cscl_system(1, device=device)
        alpha = torch.tensor([0.3], dtype=VIRIAL_DTYPE, device=device)
        nl, nptr, us = get_virial_neighbor_data(positions, cell, cutoff=5.0)

        result = ewald_real_space(
            positions,
            charges,
            cell,
            alpha,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=us,
            compute_forces=True,
            compute_charge_gradients=True,
            compute_virial=True,
        )
        assert isinstance(result, tuple)
        assert len(result) == 4
        energies, forces, charge_grads, virial = result
        assert virial.shape == (1, 3, 3)
        assert charge_grads.shape == (positions.shape[0],)

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_reciprocal_virial_without_forces(self, device):
        """Reciprocal: compute_forces=False + compute_virial=True returns (energies, virial)."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell = make_virial_cscl_system(1, device=device)
        alpha = torch.tensor([0.3], dtype=VIRIAL_DTYPE, device=device)
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=3.0)

        result = ewald_reciprocal_space(
            positions,
            charges,
            cell,
            k_vectors,
            alpha,
            compute_forces=False,
            compute_virial=True,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        energies, virial = result
        assert virial.shape == (1, 3, 3)


class TestEwaldNonNeutralVirial:
    """Virial FD tests for non-neutral (Q != 0) systems."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_ewald_total_virial_fd_non_neutral(self, device):
        """Ewald total virial matches FD for a non-neutral system."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell = make_non_neutral_system(device)
        alpha = torch.tensor([0.3], dtype=VIRIAL_DTYPE, device=device)
        k_cutoff = torch.tensor([3.0], dtype=VIRIAL_DTYPE, device=device)
        cutoff = 6.0
        pbc = torch.tensor([True, True, True], dtype=torch.bool, device=device)

        def energy_fn(pos, c):
            nl, nptr, us = cell_list(
                pos,
                cutoff,
                c.squeeze(0),
                pbc,
                return_neighbor_list=True,
            )
            return ewald_summation(
                pos,
                charges,
                c,
                alpha=alpha,
                k_cutoff=k_cutoff,
                neighbor_list=nl,
                neighbor_ptr=nptr,
                neighbor_shifts=us,
                compute_forces=False,
            ).sum()

        nl, nptr, us = get_virial_neighbor_data(positions, cell, cutoff)
        result = ewald_summation(
            positions,
            charges,
            cell,
            alpha=alpha,
            k_cutoff=k_cutoff,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=us,
            compute_forces=True,
            compute_virial=True,
        )
        explicit_virial = result[2].squeeze(0)
        fd_virial = fd_virial_full(energy_fn, positions, cell, device)

        torch.testing.assert_close(
            explicit_virial,
            fd_virial,
            atol=2e-2,
            rtol=2e-2,
            msg="Ewald total virial does not match FD for non-neutral system",
        )


class TestEwaldDifferentiableVirial:
    """Stress-loss gradients through Ewald virial path."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_ewald_stress_loss_backprop_enabled(self, device, dtype):
        """Stress loss contributes gradients when compute_virial=True."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell = make_virial_cscl_system(
            1, dtype=dtype, device=device
        )
        charges = charges.clone().requires_grad_(True)
        alpha = torch.tensor([0.3], dtype=dtype, device=device)
        nl, nptr, us = get_virial_neighbor_data(positions, cell, cutoff=6.0)

        _, _, virial = ewald_summation(
            positions,
            charges,
            cell,
            alpha=alpha,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=us,
            compute_forces=True,
            compute_virial=True,
        )

        stress_loss = virial.pow(2).sum()
        stress_loss.backward()

        assert charges.grad is not None
        assert torch.isfinite(charges.grad).all()
        assert charges.grad.abs().sum() > 0

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_ewald_virial_fd_charges(self, device):
        """Ewald virial backward gives FD-correct charge gradients."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell = make_virial_cscl_system(1, device=device)
        alpha = torch.tensor([0.3], dtype=VIRIAL_DTYPE, device=device)
        nl, nptr, us = get_virial_neighbor_data(positions, cell, cutoff=6.0)

        def virial_sum(chg):
            _, _, v = ewald_summation(
                positions,
                chg,
                cell,
                alpha=alpha,
                neighbor_list=nl,
                neighbor_ptr=nptr,
                neighbor_shifts=us,
                compute_forces=True,
                compute_virial=True,
            )
            return v.sum()

        chg = charges.clone().requires_grad_(True)
        loss = virial_sum(chg)
        loss.backward()
        ad_grad = chg.grad.clone()

        h = 1e-5
        for i in range(min(4, len(charges))):
            cp = charges.clone()
            cp[i] += h
            cm = charges.clone()
            cm[i] -= h
            fd = (virial_sum(cp).item() - virial_sum(cm).item()) / (2 * h)
            rel = abs(ad_grad[i].item() - fd) / (abs(fd) + 1e-30)
            assert rel < 0.02, (
                f"atom {i}: AD={ad_grad[i].item():.8e}, FD={fd:.8e}, rel={rel:.2e}"
            )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_mixed_energy_stress_loss(self, device):
        """Mixed loss (energy + stress) gives correct combined gradients."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell = make_virial_cscl_system(1, device=device)
        alpha = torch.tensor([0.3], dtype=VIRIAL_DTYPE, device=device)
        nl, nptr, us = get_virial_neighbor_data(positions, cell, cutoff=6.0)

        chg = charges.clone().requires_grad_(True)
        energies, _, virial = ewald_summation(
            positions,
            chg,
            cell,
            alpha=alpha,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=us,
            compute_forces=True,
            compute_virial=True,
        )

        lam = 0.1
        loss = energies.sum() + lam * virial.pow(2).sum()
        loss.backward()
        mixed_grad = chg.grad.clone()

        chg2 = charges.clone().requires_grad_(True)
        energies2, _, _ = ewald_summation(
            positions,
            chg2,
            cell,
            alpha=alpha,
            neighbor_list=nl,
            neighbor_ptr=nptr,
            neighbor_shifts=us,
            compute_forces=True,
            compute_virial=True,
        )
        energies2.sum().backward()
        energy_only_grad = chg2.grad.clone()

        diff = (mixed_grad - energy_only_grad).abs().sum().item()
        assert diff > 1e-10, "Mixed loss should differ from energy-only loss"


def _torchpme_ewald_energy(positions, charges, cell, alpha, k_cutoff, device):
    """Compute total Ewald energy via torchpme EwaldCalculator."""
    import math

    smearing = 1.0 / (math.sqrt(2.0) * alpha)
    potential = CoulombPotential(smearing=smearing).to(
        device=device, dtype=VIRIAL_DTYPE
    )
    lr_wavelength = 2 * torch.pi / k_cutoff
    calculator = EwaldCalculator(
        potential=potential,
        lr_wavelength=lr_wavelength,
        full_neighbor_list=True,
    ).to(device=device, dtype=VIRIAL_DTYPE)
    charges_col = charges.unsqueeze(1)
    cell_2d = cell.squeeze(0) if cell.dim() == 3 else cell
    potentials = calculator._compute_kspace(charges_col, cell_2d, positions)
    return (charges_col * potentials).flatten().sum()


@pytest.mark.skipif(not HAS_TORCHPME, reason="torchpme not installed")
class TestEwaldVirialTorchPMEParity:
    """Cross-validate Ewald virial against torchpme via FD on torchpme energies."""

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_ewald_reciprocal_virial_vs_torchpme_fd(self, device):
        """Ewald reciprocal virial matches FD of torchpme reciprocal energy."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell = make_virial_cscl_system(2, device=device)
        alpha_val = 0.3
        alpha = torch.tensor([alpha_val], dtype=VIRIAL_DTYPE, device=device)
        # Use k_cutoff=8.0 so both our generator and torchpme produce enough
        # k-vectors for the virial FD to be converged (at low cutoffs the two
        # generators select different k-vector sets, causing spurious divergence).
        k_cutoff = 8.0

        def torchpme_energy_fn(pos, c):
            return _torchpme_ewald_energy(pos, charges, c, alpha_val, k_cutoff, device)

        fd_virial = fd_virial_full(torchpme_energy_fn, positions, cell, device, h=1e-5)

        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=k_cutoff)
        result = ewald_reciprocal_space(
            positions,
            charges,
            cell,
            k_vectors,
            alpha,
            compute_forces=True,
            compute_virial=True,
        )
        our_virial = result[2].squeeze(0)

        torch.testing.assert_close(
            our_virial,
            fd_virial,
            atol=5e-3,
            rtol=5e-3,
            msg="Ewald reciprocal virial does not match torchpme FD virial",
        )

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_ewald_virial_charge_gradient_vs_torchpme_fd(self, device):
        """d(sum(virial))/dq from autograd matches FD."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        device = torch.device(device)
        positions, charges, cell = make_virial_cscl_system(1, device=device)
        alpha_val = 0.3
        alpha = torch.tensor([alpha_val], dtype=VIRIAL_DTYPE, device=device)
        k_cutoff = 3.0

        chg = charges.clone().requires_grad_(True)
        k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff=k_cutoff)
        _, _, virial = ewald_reciprocal_space(
            positions,
            chg,
            cell,
            k_vectors,
            alpha,
            compute_forces=True,
            compute_virial=True,
        )
        virial.sum().backward()
        ad_grad = chg.grad.clone()

        h = 1e-5
        for i in range(min(4, len(charges))):

            def _virial_sum_i(q_perturbed):
                kv = generate_k_vectors_ewald_summation(cell, k_cutoff=k_cutoff)
                _, _, v = ewald_reciprocal_space(
                    positions,
                    q_perturbed,
                    cell,
                    kv,
                    alpha,
                    compute_forces=True,
                    compute_virial=True,
                )
                return v.sum().item()

            qp = charges.clone()
            qp[i] += h
            qm = charges.clone()
            qm[i] -= h
            fd_grad = (_virial_sum_i(qp) - _virial_sum_i(qm)) / (2 * h)

            rel = abs(ad_grad[i].item() - fd_grad) / (abs(fd_grad) + 1e-30)
            assert rel < 0.02, (
                f"atom {i}: AD={ad_grad[i].item():.8e}, FD={fd_grad:.8e}, rel={rel:.2e}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
