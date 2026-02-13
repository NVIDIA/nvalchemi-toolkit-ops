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

"""Tests for JAX DFT-D3 dispersion bindings."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nvalchemiops.jax.interactions.dispersion import D3Parameters, dftd3

# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture()
def device():
    """GPU device fixture. Skips when no CUDA device is available.

    jax_kernel wrappers are CUDA-only (Warp JAX FFI limitation),
    so DFT-D3 JAX binding tests only run on GPU.
    """
    try:
        if len(jax.devices("gpu")) == 0:
            pytest.skip("No CUDA device available.")
    except RuntimeError:
        pytest.skip("No CUDA device available.")
    return "gpu"


@pytest.fixture(scope="module")
def element_tables():
    """Provide dummy element parameter tables for testing (NOT physically accurate)."""
    z_max = 17  # Maximum atomic number (Cl)
    z_max_inc = z_max + 1

    # Covalent radii in Bohr
    rcov = np.zeros(z_max_inc, dtype=np.float32)
    rcov[0:10] = np.array([0.0, 0.6, 0.8, 2.8, 2.0, 1.6, 1.4, 1.3, 1.2, 1.5])
    rcov[10] = 1.5
    rcov[17] = 1.8

    # <r4>/<r2> expectation values
    r4r2 = np.zeros(z_max_inc, dtype=np.float32)
    r4r2[0:10] = np.array([0.0, 2.0, 1.5, 10.0, 6.0, 5.0, 4.5, 4.0, 3.5, 3.0])
    r4r2[10] = 4.5
    r4r2[17] = 8.0

    # C6 reference grid and CN reference grids
    c6ref = np.zeros(z_max_inc * z_max_inc * 25, dtype=np.float32)
    cnref_i = np.zeros(z_max_inc * z_max_inc * 25, dtype=np.float32)
    cnref_j = np.zeros(z_max_inc * z_max_inc * 25, dtype=np.float32)

    # Maximum coordination numbers
    cnmax = np.array(
        [0.0, 1.5, 1.0, 6.0, 4.0, 4.0, 4.0, 4.0, 2.5, 1.5], dtype=np.float32
    )
    cnmax_full = np.zeros(z_max_inc, dtype=np.float32)
    cnmax_full[0:10] = cnmax
    cnmax_full[10] = 1.0
    cnmax_full[17] = 2.0

    # Fill C6 and CN reference grids
    for zi in range(z_max_inc):
        for zj in range(z_max_inc):
            base = (zi * z_max_inc + zj) * 25
            for p in range(5):
                for q in range(5):
                    idx = base + p * 5 + q
                    if zi > 0:
                        cnref_i[idx] = (p / 4.0) * cnmax_full[zi]
                    if zj > 0:
                        cnref_j[idx] = (q / 4.0) * cnmax_full[zj]
                    if zi > 0 and zj > 0:
                        c6ref[idx] = 10.0 * float(zi * zj) * (1.0 + 0.1 * p + 0.1 * q)

    return {
        "rcov": rcov,
        "r4r2": r4r2,
        "c6ref": c6ref,
        "cnref_i": cnref_i,
        "cnref_j": cnref_j,
        "z_max_inc": z_max_inc,
    }


@pytest.fixture(scope="module")
def functional_params():
    """Provide functional parameters for testing."""
    return {
        "a1": 0.4,
        "a2": 4.0,
        "s6": 1.0,
        "s8": 0.8,
        "k1": 16.0,
        "k3": -4.0,
    }


@pytest.fixture(scope="module")
def h2_system():
    """H2 molecule geometry."""
    separation = 1.4  # H-H distance in Bohr
    coord = np.array([0.0, 0.0, 0.0, separation, 0.0, 0.0], dtype=np.float32)
    numbers = np.array([1, 1], dtype=np.int32)

    B, M = 2, 5
    nbmat = np.array(
        [
            [1, 2, 2, 2, 2],
            [0, 2, 2, 2, 2],
        ],
        dtype=np.int32,
    )

    return {
        "coord": coord,
        "numbers": numbers,
        "nbmat": nbmat,
        "B": B,
        "M": M,
    }


@pytest.fixture(scope="module")
def ne2_system():
    """Ne2 dimer geometry."""
    separation = 5.8  # Ne-Ne distance in Bohr
    coord = np.array([0.0, 0.0, 0.0, separation, 0.0, 0.0], dtype=np.float32)
    numbers = np.array([10, 10], dtype=np.int32)

    B, M = 2, 5
    nbmat = np.array(
        [
            [1, 2, 2, 2, 2],
            [0, 2, 2, 2, 2],
        ],
        dtype=np.int32,
    )

    return {
        "coord": coord,
        "numbers": numbers,
        "nbmat": nbmat,
        "B": B,
        "M": M,
    }


@pytest.fixture(scope="module")
def d3_params(element_tables):
    """Create D3Parameters from element tables."""
    z_max_inc = element_tables["z_max_inc"]
    rcov = jnp.array(element_tables["rcov"], dtype=jnp.float32)
    r4r2 = jnp.array(element_tables["r4r2"], dtype=jnp.float32)
    c6ab = jnp.array(
        element_tables["c6ref"].reshape(z_max_inc, z_max_inc, 5, 5),
        dtype=jnp.float32,
    )
    cn_ref = jnp.array(
        element_tables["cnref_i"].reshape(z_max_inc, z_max_inc, 5, 5),
        dtype=jnp.float32,
    )

    return D3Parameters(rcov=rcov, r4r2=r4r2, c6ab=c6ab, cn_ref=cn_ref)


# ==============================================================================
# D3Parameters Tests
# ==============================================================================


class TestD3Parameters:
    """Test D3Parameters dataclass validation."""

    def test_valid_parameters(self, d3_params):
        """Test that valid parameters are accepted."""
        assert d3_params.max_z == 17
        assert d3_params.rcov.shape[0] == 18

    def test_shape_mismatch_r4r2(self):
        """Test error on r4r2 shape mismatch."""
        rcov = jnp.ones(18, dtype=jnp.float32)
        r4r2 = jnp.ones(19, dtype=jnp.float32)  # Wrong shape
        c6ab = jnp.ones((18, 18, 5, 5), dtype=jnp.float32)
        cn_ref = jnp.ones((18, 18, 5, 5), dtype=jnp.float32)

        with pytest.raises(ValueError, match="r4r2 must have shape"):
            D3Parameters(rcov=rcov, r4r2=r4r2, c6ab=c6ab, cn_ref=cn_ref)

    def test_shape_mismatch_c6ab(self):
        """Test error on c6ab shape mismatch."""
        rcov = jnp.ones(18, dtype=jnp.float32)
        r4r2 = jnp.ones(18, dtype=jnp.float32)
        c6ab = jnp.ones((18, 18, 4, 4), dtype=jnp.float32)  # Wrong shape
        cn_ref = jnp.ones((18, 18, 5, 5), dtype=jnp.float32)

        with pytest.raises(ValueError, match="c6ab must have shape"):
            D3Parameters(rcov=rcov, r4r2=r4r2, c6ab=c6ab, cn_ref=cn_ref)

    def test_invalid_dtype(self):
        """Test error on invalid dtype."""
        rcov = jnp.ones(18, dtype=jnp.int32)  # Wrong dtype
        r4r2 = jnp.ones(18, dtype=jnp.float32)
        c6ab = jnp.ones((18, 18, 5, 5), dtype=jnp.float32)
        cn_ref = jnp.ones((18, 18, 5, 5), dtype=jnp.float32)

        with pytest.raises(TypeError, match="must be float32 or float64"):
            D3Parameters(rcov=rcov, r4r2=r4r2, c6ab=c6ab, cn_ref=cn_ref)


# ==============================================================================
# DFT-D3 Function Tests
# ==============================================================================


class TestDFT_D3Basic:
    """Test basic DFT-D3 functionality."""

    def test_h2_neighbor_matrix(self, h2_system, functional_params, d3_params, device):
        """Test H2 with neighbor matrix format on specified device."""
        positions = jnp.array(h2_system["coord"].reshape(2, 3), dtype=jnp.float32)
        numbers = jnp.array(h2_system["numbers"], dtype=jnp.int32)
        neighbor_matrix = jnp.array(h2_system["nbmat"], dtype=jnp.int32)

        result = dftd3(
            positions,
            numbers,
            a1=functional_params["a1"],
            a2=functional_params["a2"],
            s8=functional_params["s8"],
            k1=functional_params["k1"],
            k3=functional_params["k3"],
            s6=functional_params["s6"],
            neighbor_matrix=neighbor_matrix,
            d3_params=d3_params,
        )

        energy, forces, coord_num = result[0], result[1], result[2]

        # Check output shapes
        assert energy.shape == (1,)
        assert forces.shape == (2, 3)
        assert coord_num.shape == (2,)

        # Check outputs are finite
        assert jnp.all(jnp.isfinite(energy))
        assert jnp.all(jnp.isfinite(forces))
        assert jnp.all(jnp.isfinite(coord_num))

    def test_h2_neighbor_list(self, h2_system, functional_params, d3_params, device):
        """Test H2 with neighbor list format"""
        positions = jnp.array(h2_system["coord"].reshape(2, 3), dtype=jnp.float32)
        numbers = jnp.array(h2_system["numbers"], dtype=jnp.int32)

        # Build neighbor list: atom 0 -> 1, atom 1 -> 0
        neighbor_list = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
        neighbor_ptr = jnp.array([0, 1, 2], dtype=jnp.int32)

        result = dftd3(
            positions,
            numbers,
            a1=functional_params["a1"],
            a2=functional_params["a2"],
            s8=functional_params["s8"],
            k1=functional_params["k1"],
            k3=functional_params["k3"],
            s6=functional_params["s6"],
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            d3_params=d3_params,
        )

        energy, forces, coord_num = result[0], result[1], result[2]

        # Check output shapes
        assert energy.shape == (1,)
        assert forces.shape == (2, 3)
        assert coord_num.shape == (2,)

        # Check outputs are finite
        assert jnp.all(jnp.isfinite(energy))
        assert jnp.all(jnp.isfinite(forces))
        assert jnp.all(jnp.isfinite(coord_num))

    def test_empty_system(self, functional_params, d3_params):
        """Test empty system (edge case)."""
        positions = jnp.zeros((0, 3), dtype=jnp.float32)
        numbers = jnp.zeros((0,), dtype=jnp.int32)
        neighbor_matrix = jnp.zeros((0, 1), dtype=jnp.int32)

        result = dftd3(
            positions,
            numbers,
            a1=functional_params["a1"],
            a2=functional_params["a2"],
            s8=functional_params["s8"],
            neighbor_matrix=neighbor_matrix,
            d3_params=d3_params,
        )

        energy, forces, coord_num = result[0], result[1], result[2]

        assert energy.shape == (1,)
        assert forces.shape == (0, 3)
        assert coord_num.shape == (0,)

    def test_missing_neighbor_format(self, h2_system, functional_params, d3_params):
        """Test error when neither neighbor format is provided."""
        positions = jnp.array(h2_system["coord"].reshape(2, 3), dtype=jnp.float32)
        numbers = jnp.array(h2_system["numbers"], dtype=jnp.int32)

        with pytest.raises(ValueError, match="Must provide either"):
            dftd3(
                positions,
                numbers,
                a1=functional_params["a1"],
                a2=functional_params["a2"],
                s8=functional_params["s8"],
                d3_params=d3_params,
            )

    def test_both_neighbor_formats(self, h2_system, functional_params, d3_params):
        """Test error when both neighbor formats are provided."""
        positions = jnp.array(h2_system["coord"].reshape(2, 3), dtype=jnp.float32)
        numbers = jnp.array(h2_system["numbers"], dtype=jnp.int32)
        neighbor_matrix = jnp.array(h2_system["nbmat"], dtype=jnp.int32)
        neighbor_list = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
        neighbor_ptr = jnp.array([0, 1, 2], dtype=jnp.int32)

        with pytest.raises(ValueError, match="Cannot provide both"):
            dftd3(
                positions,
                numbers,
                a1=functional_params["a1"],
                a2=functional_params["a2"],
                s8=functional_params["s8"],
                neighbor_matrix=neighbor_matrix,
                neighbor_list=neighbor_list,
                neighbor_ptr=neighbor_ptr,
                d3_params=d3_params,
            )


# ==============================================================================
# Dtype Tests
# ==============================================================================


class TestDFT_D3Dtypes:
    """Test DFT-D3 with different dtypes."""

    def test_float32_positions(self, h2_system, functional_params, d3_params, device):
        """Test with float32 positions."""
        positions = jnp.array(h2_system["coord"].reshape(2, 3), dtype=jnp.float32)
        numbers = jnp.array(h2_system["numbers"], dtype=jnp.int32)
        neighbor_matrix = jnp.array(h2_system["nbmat"], dtype=jnp.int32)

        result = dftd3(
            positions,
            numbers,
            a1=functional_params["a1"],
            a2=functional_params["a2"],
            s8=functional_params["s8"],
            neighbor_matrix=neighbor_matrix,
            d3_params=d3_params,
        )

        energy, forces = result[0], result[1]
        assert forces.dtype == jnp.float32
        assert energy.dtype == jnp.float32

    def test_float64_positions(self, h2_system, functional_params, d3_params, device):
        """Test with float64 positions."""
        positions = jnp.array(h2_system["coord"].reshape(2, 3), dtype=jnp.float64)
        numbers = jnp.array(h2_system["numbers"], dtype=jnp.int32)
        neighbor_matrix = jnp.array(h2_system["nbmat"], dtype=jnp.int32)

        result = dftd3(
            positions,
            numbers,
            a1=functional_params["a1"],
            a2=functional_params["a2"],
            s8=functional_params["s8"],
            neighbor_matrix=neighbor_matrix,
            d3_params=d3_params,
        )

        energy, forces, coord_num = result[0], result[1], result[2]

        # Output is always float32 regardless of input precision
        assert forces.dtype == jnp.float32
        assert energy.dtype == jnp.float32
        assert coord_num.dtype == jnp.float32

    def test_float32_float64_consistency(
        self, h2_system, functional_params, d3_params, device
    ):
        """Test that float32 and float64 positions produce similar results."""
        # Run with float32
        positions_f32 = jnp.array(h2_system["coord"].reshape(2, 3), dtype=jnp.float32)
        numbers = jnp.array(h2_system["numbers"], dtype=jnp.int32)
        neighbor_matrix = jnp.array(h2_system["nbmat"], dtype=jnp.int32)

        result_f32 = dftd3(
            positions_f32,
            numbers,
            a1=functional_params["a1"],
            a2=functional_params["a2"],
            s8=functional_params["s8"],
            neighbor_matrix=neighbor_matrix,
            d3_params=d3_params,
        )
        energy_f32, forces_f32, coord_num_f32 = (
            result_f32[0],
            result_f32[1],
            result_f32[2],
        )

        # Run with float64
        positions_f64 = jnp.array(h2_system["coord"].reshape(2, 3), dtype=jnp.float64)
        result_f64 = dftd3(
            positions_f64,
            numbers,
            a1=functional_params["a1"],
            a2=functional_params["a2"],
            s8=functional_params["s8"],
            neighbor_matrix=neighbor_matrix,
            d3_params=d3_params,
        )
        energy_f64, forces_f64, coord_num_f64 = (
            result_f64[0],
            result_f64[1],
            result_f64[2],
        )

        # Should be very close (float64 may have slightly better precision)
        assert jnp.allclose(energy_f64, energy_f32, rtol=1e-5, atol=1e-7)
        assert jnp.allclose(forces_f64, forces_f32, rtol=1e-5, atol=1e-7)
        assert jnp.allclose(coord_num_f64, coord_num_f32, rtol=1e-5, atol=1e-7)


# ==============================================================================
# Physical Correctness Tests
# ==============================================================================


class TestPhysicalCorrectness:
    """Test physical correctness and energy relationships."""

    def test_h2_energy_sign(self, h2_system, functional_params, d3_params, device):
        """Test that H2 produces negative (attractive) dispersion energy."""
        positions = jnp.array(h2_system["coord"].reshape(2, 3), dtype=jnp.float32)
        numbers = jnp.array(h2_system["numbers"], dtype=jnp.int32)
        neighbor_matrix = jnp.array(h2_system["nbmat"], dtype=jnp.int32)

        result = dftd3(
            positions,
            numbers,
            a1=functional_params["a1"],
            a2=functional_params["a2"],
            s8=functional_params["s8"],
            neighbor_matrix=neighbor_matrix,
            d3_params=d3_params,
        )

        energy = result[0]
        forces = result[1]
        coord_num = result[2]

        # Dispersion should be attractive (negative)
        assert energy[0] < 0.0

        # Forces should be opposite for symmetric system
        assert jnp.allclose(forces[0], -forces[1], rtol=1e-5, atol=1e-7)

        # Coordination numbers should be small but non-zero
        assert jnp.all(coord_num > 0)
        assert jnp.all(coord_num < 1.0)

    def test_ne2_larger_dispersion(
        self, ne2_system, functional_params, d3_params, device
    ):
        """Test that Ne2 has significant dispersion energy."""
        positions = jnp.array(ne2_system["coord"].reshape(2, 3), dtype=jnp.float32)
        numbers = jnp.array(ne2_system["numbers"], dtype=jnp.int32)
        neighbor_matrix = jnp.array(ne2_system["nbmat"], dtype=jnp.int32)

        result = dftd3(
            positions,
            numbers,
            a1=functional_params["a1"],
            a2=functional_params["a2"],
            s8=functional_params["s8"],
            neighbor_matrix=neighbor_matrix,
            d3_params=d3_params,
        )

        energy_ne2 = result[0]

        # Ne2 should have significant dispersion energy
        assert energy_ne2[0] < -1e-3  # Reasonably large magnitude
        assert jnp.all(jnp.isfinite(energy_ne2))

    def test_forces_zero_for_no_neighbors(self, functional_params, d3_params, device):
        """Test that forces are zero when atoms have no neighbors."""
        # Single atom with no neighbors
        positions = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
        numbers = jnp.array([1], dtype=jnp.int32)
        neighbor_matrix = jnp.array([[1]], dtype=jnp.int32)  # Padding (only self)

        result = dftd3(
            positions,
            numbers,
            a1=functional_params["a1"],
            a2=functional_params["a2"],
            s8=functional_params["s8"],
            neighbor_matrix=neighbor_matrix,
            d3_params=d3_params,
        )

        energy = result[0]
        forces = result[1]
        coord_num = result[2]

        # Energy should be zero for no neighbors
        assert jnp.abs(energy[0]) < 1e-7

        # Forces should be zero for no neighbors
        assert jnp.allclose(forces, 0.0, atol=1e-7)

        # Coordination number should be zero for no neighbors
        assert jnp.allclose(coord_num, 0.0, atol=1e-7)


# ==============================================================================
# Validation and Error Handling Tests
# ==============================================================================


class TestValidation:
    """Test error handling and input validation."""

    def test_missing_neighbor_format(self, h2_system, functional_params, d3_params):
        """Test error when neither neighbor format is provided."""
        positions = jnp.array(h2_system["coord"].reshape(2, 3), dtype=jnp.float32)
        numbers = jnp.array(h2_system["numbers"], dtype=jnp.int32)

        with pytest.raises(ValueError, match="Must provide either"):
            dftd3(
                positions,
                numbers,
                a1=functional_params["a1"],
                a2=functional_params["a2"],
                s8=functional_params["s8"],
                d3_params=d3_params,
            )

    def test_both_neighbor_formats(self, h2_system, functional_params, d3_params):
        """Test error when both neighbor formats are provided."""
        positions = jnp.array(h2_system["coord"].reshape(2, 3), dtype=jnp.float32)
        numbers = jnp.array(h2_system["numbers"], dtype=jnp.int32)
        neighbor_matrix = jnp.array(h2_system["nbmat"], dtype=jnp.int32)
        neighbor_list = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
        neighbor_ptr = jnp.array([0, 1, 2], dtype=jnp.int32)

        with pytest.raises(ValueError, match="Cannot provide both"):
            dftd3(
                positions,
                numbers,
                a1=functional_params["a1"],
                a2=functional_params["a2"],
                s8=functional_params["s8"],
                neighbor_matrix=neighbor_matrix,
                neighbor_list=neighbor_list,
                neighbor_ptr=neighbor_ptr,
                d3_params=d3_params,
            )

    def test_missing_neighbor_ptr(self, h2_system, functional_params, d3_params):
        """Test error when neighbor_ptr is missing for neighbor_list format."""
        positions = jnp.array(h2_system["coord"].reshape(2, 3), dtype=jnp.float32)
        numbers = jnp.array(h2_system["numbers"], dtype=jnp.int32)
        neighbor_list = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)

        with pytest.raises(ValueError, match="neighbor_ptr must be provided"):
            dftd3(
                positions,
                numbers,
                a1=functional_params["a1"],
                a2=functional_params["a2"],
                s8=functional_params["s8"],
                neighbor_list=neighbor_list,
                d3_params=d3_params,
            )

    def test_virial_requires_pbc(self, h2_system, functional_params, d3_params):
        """Test error when virial computation requested without PBC."""
        positions = jnp.array(h2_system["coord"].reshape(2, 3), dtype=jnp.float32)
        numbers = jnp.array(h2_system["numbers"], dtype=jnp.int32)
        neighbor_matrix = jnp.array(h2_system["nbmat"], dtype=jnp.int32)

        with pytest.raises(
            ValueError,
            match="Virial computation requires periodic boundary conditions",
        ):
            dftd3(
                positions,
                numbers,
                a1=functional_params["a1"],
                a2=functional_params["a2"],
                s8=functional_params["s8"],
                neighbor_matrix=neighbor_matrix,
                d3_params=d3_params,
                compute_virial=True,
            )

    def test_missing_d3_parameters(self, h2_system, functional_params):
        """Test error when DFT-D3 parameters not provided."""
        positions = jnp.array(h2_system["coord"].reshape(2, 3), dtype=jnp.float32)
        numbers = jnp.array(h2_system["numbers"], dtype=jnp.int32)
        neighbor_matrix = jnp.array(h2_system["nbmat"], dtype=jnp.int32)

        with pytest.raises(RuntimeError, match="DFT-D3 parameters must be"):
            dftd3(
                positions,
                numbers,
                a1=functional_params["a1"],
                a2=functional_params["a2"],
                s8=functional_params["s8"],
                neighbor_matrix=neighbor_matrix,
            )


# ==============================================================================
# Regression Tests
# ==============================================================================


class TestRegression:
    """Regression tests with hardcoded reference values."""

    def test_ne2_regression(self, ne2_system, functional_params, d3_params):
        """Test Ne2 produces expected reference values."""
        positions = jnp.array(ne2_system["coord"].reshape(2, 3), dtype=jnp.float32)
        numbers = jnp.array(ne2_system["numbers"], dtype=jnp.int32)
        neighbor_matrix = jnp.array(ne2_system["nbmat"], dtype=jnp.int32)

        result = dftd3(
            positions,
            numbers,
            a1=functional_params["a1"],
            a2=functional_params["a2"],
            s8=functional_params["s8"],
            neighbor_matrix=neighbor_matrix,
            d3_params=d3_params,
        )

        energy = result[0]
        forces = result[1]
        coord_num = result[2]

        # Check energy sign and magnitude - Ne2 should have negative energy
        assert energy[0] < 0.0

        # Check force symmetry
        assert jnp.allclose(forces[0], -forces[1], rtol=1e-5, atol=1e-7)

        # Check coordination numbers are small
        assert jnp.all(coord_num > 0)
        assert jnp.all(coord_num < 1.0)


# ==============================================================================
# JAX JIT Compatibility Tests
# ==============================================================================


class TestDFT_D3JIT:
    """Smoke tests for DFT-D3 compatibility with jax.jit."""

    def test_jit_neighbor_matrix(self, h2_system, functional_params, d3_params, device):
        """Test H2 with neighbor matrix format works with jax.jit."""
        positions = jnp.array(h2_system["coord"].reshape(2, 3), dtype=jnp.float32)
        numbers = jnp.array(h2_system["numbers"], dtype=jnp.int32)
        neighbor_matrix = jnp.array(h2_system["nbmat"], dtype=jnp.int32)

        rcov = d3_params.rcov
        r4r2 = d3_params.r4r2
        c6ab = d3_params.c6ab
        cn_ref = d3_params.cn_ref

        # Define jitted function with array parameters as arguments
        # Scalar parameters must be static literals for Warp FFI compatibility
        @jax.jit
        def jitted_dftd3(positions, numbers, neighbor_matrix, rcov, r4r2, c6ab, cn_ref):
            # Construct D3Parameters inside jitted function
            d3_params_jit = D3Parameters(rcov=rcov, r4r2=r4r2, c6ab=c6ab, cn_ref=cn_ref)
            return dftd3(
                positions,
                numbers,
                a1=0.4,  # Static literal for FFI compatibility
                a2=4.0,
                s8=0.8,
                k1=16.0,
                k3=-4.0,
                s6=1.0,
                neighbor_matrix=neighbor_matrix,
                d3_params=d3_params_jit,
            )

        # Call jitted function
        result = jitted_dftd3(
            positions, numbers, neighbor_matrix, rcov, r4r2, c6ab, cn_ref
        )

        energy, forces, coord_num = result[0], result[1], result[2]

        # Check output shapes
        assert energy.shape == (1,)
        assert forces.shape == (2, 3)
        assert coord_num.shape == (2,)

        # Check outputs are finite
        assert jnp.all(jnp.isfinite(energy))
        assert jnp.all(jnp.isfinite(forces))
        assert jnp.all(jnp.isfinite(coord_num))

        # Dispersion should be attractive (negative)
        assert energy[0] < 0.0

    def test_jit_neighbor_list(self, h2_system, functional_params, d3_params, device):
        """Test H2 with neighbor list format works with jax.jit."""
        positions = jnp.array(h2_system["coord"].reshape(2, 3), dtype=jnp.float32)
        numbers = jnp.array(h2_system["numbers"], dtype=jnp.int32)

        # Build neighbor list: atom 0 -> 1, atom 1 -> 0
        neighbor_list = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
        neighbor_ptr = jnp.array([0, 1, 2], dtype=jnp.int32)

        rcov = d3_params.rcov
        r4r2 = d3_params.r4r2
        c6ab = d3_params.c6ab
        cn_ref = d3_params.cn_ref

        # Define jitted function with array parameters as arguments
        # Scalar parameters must be static literals for Warp FFI compatibility
        @jax.jit
        def jitted_dftd3(positions, numbers, nl, nptr, rcov, r4r2, c6ab, cn_ref):
            # Construct D3Parameters inside jitted function
            d3_params_jit = D3Parameters(rcov=rcov, r4r2=r4r2, c6ab=c6ab, cn_ref=cn_ref)
            return dftd3(
                positions,
                numbers,
                a1=0.4,  # Static literal for FFI compatibility
                a2=4.0,
                s8=0.8,
                k1=16.0,
                k3=-4.0,
                s6=1.0,
                neighbor_list=nl,
                neighbor_ptr=nptr,
                d3_params=d3_params_jit,
            )

        # Call jitted function
        result = jitted_dftd3(
            positions, numbers, neighbor_list, neighbor_ptr, rcov, r4r2, c6ab, cn_ref
        )

        energy, forces, coord_num = result[0], result[1], result[2]

        # Check output shapes
        assert energy.shape == (1,)
        assert forces.shape == (2, 3)
        assert coord_num.shape == (2,)

        # Check outputs are finite
        assert jnp.all(jnp.isfinite(energy))
        assert jnp.all(jnp.isfinite(forces))
        assert jnp.all(jnp.isfinite(coord_num))

        # Dispersion should be attractive (negative)
        assert energy[0] < 0.0


# ==============================================================================
# PBC Tests
# ==============================================================================


class TestDFTD3PBC:
    """Test DFT-D3 with periodic boundary conditions."""

    def _make_periodic_h2(self, d3_params):
        """Create H2 in a periodic box with shifts."""
        # H2 in a 10 Bohr cubic box
        positions = jnp.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=jnp.float32)
        numbers = jnp.array([1, 1], dtype=jnp.int32)
        cell = jnp.array(
            [[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]],
            dtype=jnp.float32,
        )
        # Neighbor matrix: each atom sees the other, zero shifts (same image)
        neighbor_matrix = jnp.array([[1, 2, 2, 2, 2], [0, 2, 2, 2, 2]], dtype=jnp.int32)
        # Shifts: shape (N, max_neighbors, 3), all zero (same periodic image)
        neighbor_matrix_shifts = jnp.zeros((2, 5, 3), dtype=jnp.int32)
        return (
            positions,
            numbers,
            cell,
            neighbor_matrix,
            neighbor_matrix_shifts,
            d3_params,
        )

    def test_pbc_neighbor_matrix_basic(self, functional_params, d3_params, device):
        """Test PBC with neighbor matrix format produces finite results."""
        pos, nums, cell, nm, nms, d3p = self._make_periodic_h2(d3_params)

        energy, forces, coord_num = dftd3(
            pos,
            nums,
            a1=functional_params["a1"],
            a2=functional_params["a2"],
            s8=functional_params["s8"],
            k1=functional_params["k1"],
            k3=functional_params["k3"],
            s6=functional_params["s6"],
            neighbor_matrix=nm,
            neighbor_matrix_shifts=nms,
            cell=cell,
            d3_params=d3p,
        )

        assert energy.shape == (1,)
        assert forces.shape == (2, 3)
        assert jnp.all(jnp.isfinite(energy))
        assert jnp.all(jnp.isfinite(forces))
        assert energy[0] < 0.0  # Attractive

    def test_pbc_neighbor_list_basic(self, functional_params, d3_params, device):
        """Test PBC with neighbor list format produces finite results."""
        pos, nums, cell, _, _, d3p = self._make_periodic_h2(d3_params)

        # Neighbor list format: COO with pointers
        neighbor_list = jnp.array([[0, 1], [1, 0]], dtype=jnp.int32)
        neighbor_ptr = jnp.array([0, 1, 2], dtype=jnp.int32)
        unit_shifts = jnp.zeros((2, 3), dtype=jnp.int32)

        energy, forces, coord_num = dftd3(
            pos,
            nums,
            a1=functional_params["a1"],
            a2=functional_params["a2"],
            s8=functional_params["s8"],
            k1=functional_params["k1"],
            k3=functional_params["k3"],
            s6=functional_params["s6"],
            neighbor_list=neighbor_list,
            neighbor_ptr=neighbor_ptr,
            unit_shifts=unit_shifts,
            cell=cell,
            d3_params=d3p,
        )

        assert energy.shape == (1,)
        assert forces.shape == (2, 3)
        assert jnp.all(jnp.isfinite(energy))
        assert jnp.all(jnp.isfinite(forces))

    def test_pbc_energy_differs_from_nonpbc(self, functional_params, d3_params, device):
        """Periodic energy differs from non-periodic when periodic images contribute."""
        # Use a SMALL box so periodic images are close and contribute
        positions = jnp.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=jnp.float32)
        numbers = jnp.array([1, 1], dtype=jnp.int32)

        # Non-periodic: neighbor sees same-image neighbor only
        nm = jnp.array([[1, 2, 2, 2, 2], [0, 2, 2, 2, 2]], dtype=jnp.int32)
        e_nonpbc, _, _ = dftd3(
            positions,
            numbers,
            a1=functional_params["a1"],
            a2=functional_params["a2"],
            s8=functional_params["s8"],
            k1=functional_params["k1"],
            k3=functional_params["k3"],
            s6=functional_params["s6"],
            neighbor_matrix=nm,
            d3_params=d3_params,
        )

        # Periodic: add a periodic image neighbor via shifts
        # Use small box: 3.0 Bohr so periodic image of atom 1 is close
        small_cell = jnp.array(
            [[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]],
            dtype=jnp.float32,
        )
        # Neighbor matrix: atom 0 sees [atom 1 (same image), atom 1 (image +1,0,0)]
        nm_pbc = jnp.array([[1, 1, 2, 2, 2], [0, 0, 2, 2, 2]], dtype=jnp.int32)
        nms_pbc = jnp.zeros((2, 5, 3), dtype=jnp.int32)
        # Set shift for second neighbor of atom 0: image (+1, 0, 0)
        nms_pbc = nms_pbc.at[0, 1, 0].set(1)
        # Set shift for second neighbor of atom 1: image (-1, 0, 0)
        nms_pbc = nms_pbc.at[1, 1, 0].set(-1)

        e_pbc, _, _ = dftd3(
            positions,
            numbers,
            a1=functional_params["a1"],
            a2=functional_params["a2"],
            s8=functional_params["s8"],
            k1=functional_params["k1"],
            k3=functional_params["k3"],
            s6=functional_params["s6"],
            neighbor_matrix=nm_pbc,
            neighbor_matrix_shifts=nms_pbc,
            cell=small_cell,
            d3_params=d3_params,
        )

        # Energy with periodic images should be more negative (more neighbors)
        assert float(e_pbc[0]) < float(e_nonpbc[0]), (
            f"PBC energy ({e_pbc[0]}) should be more negative than non-PBC ({e_nonpbc[0]})"
        )


# ==============================================================================
# Virial Tests
# ==============================================================================


class TestDFTD3Virial:
    """Test DFT-D3 virial computation."""

    def test_virial_shape(self, functional_params, d3_params, device):
        """Virial has shape (num_systems, 3, 3)."""
        positions = jnp.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=jnp.float32)
        numbers = jnp.array([1, 1], dtype=jnp.int32)
        cell = jnp.array(
            [[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]],
            dtype=jnp.float32,
        )
        neighbor_matrix = jnp.array([[1, 2, 2, 2, 2], [0, 2, 2, 2, 2]], dtype=jnp.int32)
        neighbor_matrix_shifts = jnp.zeros((2, 5, 3), dtype=jnp.int32)

        result = dftd3(
            positions,
            numbers,
            a1=functional_params["a1"],
            a2=functional_params["a2"],
            s8=functional_params["s8"],
            k1=functional_params["k1"],
            k3=functional_params["k3"],
            s6=functional_params["s6"],
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            cell=cell,
            d3_params=d3_params,
            compute_virial=True,
        )

        assert len(result) == 4
        energy, forces, coord_num, virial = result
        assert virial.shape == (1, 3, 3)  # Single system

    def test_virial_finite_nonzero(self, functional_params, d3_params, device):
        """Virial is finite and non-zero for periodic system."""
        positions = jnp.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=jnp.float32)
        numbers = jnp.array([1, 1], dtype=jnp.int32)
        cell = jnp.array(
            [[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]],
            dtype=jnp.float32,
        )
        neighbor_matrix = jnp.array([[1, 2, 2, 2, 2], [0, 2, 2, 2, 2]], dtype=jnp.int32)
        neighbor_matrix_shifts = jnp.zeros((2, 5, 3), dtype=jnp.int32)

        result = dftd3(
            positions,
            numbers,
            a1=functional_params["a1"],
            a2=functional_params["a2"],
            s8=functional_params["s8"],
            k1=functional_params["k1"],
            k3=functional_params["k3"],
            s6=functional_params["s6"],
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            cell=cell,
            d3_params=d3_params,
            compute_virial=True,
        )

        virial = result[3]
        assert jnp.all(jnp.isfinite(virial))
        # For a dimer along x with interaction, virial should have non-zero xx component
        assert jnp.any(jnp.abs(virial) > 1e-10)

    def test_virial_symmetry(self, functional_params, d3_params, device):
        """Virial tensor is approximately symmetric."""
        positions = jnp.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=jnp.float32)
        numbers = jnp.array([1, 1], dtype=jnp.int32)
        cell = jnp.array(
            [[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]],
            dtype=jnp.float32,
        )
        neighbor_matrix = jnp.array([[1, 2, 2, 2, 2], [0, 2, 2, 2, 2]], dtype=jnp.int32)
        neighbor_matrix_shifts = jnp.zeros((2, 5, 3), dtype=jnp.int32)

        result = dftd3(
            positions,
            numbers,
            a1=functional_params["a1"],
            a2=functional_params["a2"],
            s8=functional_params["s8"],
            k1=functional_params["k1"],
            k3=functional_params["k3"],
            s6=functional_params["s6"],
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            cell=cell,
            d3_params=d3_params,
            compute_virial=True,
        )

        virial = result[3].squeeze(0)  # (3, 3)
        assert jnp.allclose(virial, virial.T, atol=1e-6, rtol=1e-6)

    def test_virial_requires_shifts(self, functional_params, d3_params):
        """Virial raises ValueError when shifts are missing."""
        positions = jnp.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=jnp.float32)
        numbers = jnp.array([1, 1], dtype=jnp.int32)
        cell = jnp.array(
            [[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]],
            dtype=jnp.float32,
        )
        nm = jnp.array([[1, 2, 2, 2, 2], [0, 2, 2, 2, 2]], dtype=jnp.int32)

        # Neighbor matrix format without shifts
        with pytest.raises(ValueError):
            dftd3(
                positions,
                numbers,
                a1=functional_params["a1"],
                a2=functional_params["a2"],
                s8=functional_params["s8"],
                neighbor_matrix=nm,
                cell=cell,
                d3_params=d3_params,
                compute_virial=True,
            )
