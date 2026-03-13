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


"""
DSF Accuracy Benchmark
======================

CLI tool to benchmark DSF electrostatic accuracy against OpenMM PME reference.
Results are saved with GPU-specific naming:
``dsf_accuracy_<system>_<gpu_sku>.csv``

Tests two systems:
1. NaCl crystal (rock salt) with random displacements
2. 0.1M NaCl in water (TIP3P/amber14)

For each system, sweeps over DSF cutoff and alpha parameters in both
float32 and float64, comparing energy and forces against OpenMM PME.

Usage:
    python benchmark_dsf_accuracy.py --config benchmark_dsf_accuracy_config.yaml
    python benchmark_dsf_accuracy.py --config benchmark_dsf_accuracy_config.yaml --output-dir ./results
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import warp as wp

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import yaml

from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors
from nvalchemiops.torch.interactions.electrostatics import dsf_coulomb
from nvalchemiops.torch.neighbors import neighbor_list

try:
    import openmm
    import openmm.app as app
    import openmm.unit as unit

    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False


# ==============================================================================
# Constants
# ==============================================================================

# Coulomb constant ke = e^2 / (4*pi*eps0) in eV*Angstrom/e^2
# nvalchemiops DSF returns energy in units of e^2/Angstrom (no ke factor),
# so multiply by KE_EVA to convert to eV.
KE_EVA = 14.399645351950548

# OpenMM returns energy in kJ/mol, forces in kJ/(mol*nm)
KJ_PER_MOL_TO_EV = 1.0 / 96.4853074954296
KJ_PER_MOL_NM_TO_EV_PER_A = KJ_PER_MOL_TO_EV / 10.0


# ==============================================================================
# Utilities
# ==============================================================================


def get_gpu_sku() -> str:
    """Get GPU SKU name for filename generation."""
    if not torch.cuda.is_available():
        return "cpu"

    try:
        gpu_name = torch.cuda.get_device_name(0)
        sku = gpu_name.replace(" ", "-").replace("_", "-")
        sku = sku.replace("NVIDIA-", "").replace("GeForce-", "")
        return sku.lower()
    except Exception:
        return "unknown_gpu"


def load_config(config_path: Path) -> dict:
    """Load benchmark configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def compute_min_perpendicular_width(cell: np.ndarray) -> float:
    """Compute the minimum perpendicular width of a unit cell.

    For a non-orthogonal cell, the perpendicular width along direction i
    is V / |a_j x a_k| where V is the cell volume and j, k are the other
    two lattice vector indices.

    Parameters
    ----------
    cell : np.ndarray, shape (3, 3)
        Unit cell matrix (row vectors).

    Returns
    -------
    float
        Minimum perpendicular distance between opposite faces.
    """
    a1, a2, a3 = cell[0], cell[1], cell[2]
    volume = abs(np.dot(a1, np.cross(a2, a3)))
    d1 = volume / np.linalg.norm(np.cross(a2, a3))
    d2 = volume / np.linalg.norm(np.cross(a1, a3))
    d3 = volume / np.linalg.norm(np.cross(a1, a2))
    return min(d1, d2, d3)


# ==============================================================================
# NaCl Crystal System
# ==============================================================================


def create_nacl_supercell(size: int) -> dict:
    """Create NaCl (rock salt) supercell using the conventional cubic cell.

    The conventional cubic cell has 8 atoms (4 Na + 4 Cl) with lattice
    constant a = 5.64 Angstroms. This produces an orthogonal box compatible
    with OpenMM's periodic boundary conditions.

    Parameters
    ----------
    size : int
        Linear supercell size. Total atoms = 8 * size^3.

    Returns
    -------
    dict
        Dictionary with 'positions' (N, 3), 'cell' (3, 3), 'charges' (N,)
        as numpy arrays.
    """
    a = 5.64  # Conventional cubic lattice constant in Angstroms

    # Conventional cubic cell (orthogonal)
    base_cell = np.eye(3) * a

    # Fractional positions in the conventional cubic cell (8 atoms)
    # Na at FCC positions
    # Cl at FCC positions shifted by (0.5, 0.5, 0.5)
    base_frac = np.array(
        [
            # Na (4 atoms at FCC positions)
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            # Cl (4 atoms at FCC + (0.5, 0.5, 0.5))
            [0.5, 0.5, 0.5],
            [0.0, 0.0, 0.5],
            [0.0, 0.5, 0.0],
            [0.5, 0.0, 0.0],
        ]
    )
    base_charges = np.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])

    positions_list = []
    charges_list = []

    for i in range(size):
        for j in range(size):
            for k in range(size):
                offset = np.array([i, j, k])
                for idx, frac_pos in enumerate(base_frac):
                    cart_pos = (frac_pos + offset) @ base_cell
                    positions_list.append(cart_pos)
                    charges_list.append(base_charges[idx])

    supercell = base_cell * size

    return {
        "positions": np.array(positions_list),
        "cell": supercell,
        "charges": np.array(charges_list),
    }


def generate_nacl_configurations(
    size: int,
    num_configs: int,
    displacement_std: float,
    displacement_max: float,
    rng: np.random.Generator,
) -> list[dict]:
    """Generate displaced NaCl crystal configurations.

    Parameters
    ----------
    size : int
        Supercell linear dimension.
    num_configs : int
        Number of configurations to generate.
    displacement_std : float
        Standard deviation of Gaussian displacement in Angstroms.
    displacement_max : float
        Maximum displacement magnitude in Angstroms (clip).
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    list[dict]
        List of configuration dicts with 'positions', 'cell', 'charges'.
    """
    base = create_nacl_supercell(size)
    configs = []

    for _ in range(num_configs):
        disp = rng.normal(0.0, displacement_std, size=base["positions"].shape)
        norms = np.linalg.norm(disp, axis=1, keepdims=True)
        mask = norms > displacement_max
        disp = np.where(mask, disp * displacement_max / norms, disp)

        configs.append(
            {
                "positions": base["positions"] + disp,
                "cell": base["cell"].copy(),
                "charges": base["charges"].copy(),
            }
        )

    return configs


# ==============================================================================
# NaCl in Water System
# ==============================================================================


def build_nacl_water_system(config: dict) -> tuple:
    """Build a 0.1M NaCl in water system using OpenMM.

    Parameters
    ----------
    config : dict
        Configuration dict with 'box_size', 'ionic_strength', 'forcefield',
        'minimize_max_iterations', 'temperature' keys.

    Returns
    -------
    tuple
        (topology, system, positions_nm, charges_e, box_vectors_nm)
        where positions_nm is in nanometers, charges_e in elementary charges,
        box_vectors_nm are the periodic box vectors in nanometers.
    """
    ff_files = config["forcefield"]
    forcefield = app.ForceField(*ff_files)

    box_size_nm = config["box_size"]

    # Create a minimal topology and add solvent
    modeller = app.Modeller(app.Topology(), [])
    modeller.topology.setUnitCellDimensions(
        openmm.Vec3(box_size_nm, box_size_nm, box_size_nm) * unit.nanometers
    )
    modeller.addSolvent(
        forcefield,
        model="tip3p",
        boxSize=openmm.Vec3(box_size_nm, box_size_nm, box_size_nm) * unit.nanometers,
        ionicStrength=config["ionic_strength"] * unit.molar,
    )

    # Create system for minimization / NVT
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometers,
        constraints=app.HBonds,
    )

    # Extract charges from NonbondedForce
    nb_force = None
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            nb_force = force
            break

    num_particles = system.getNumParticles()
    charges_e = np.zeros(num_particles)
    for i in range(num_particles):
        charge, sigma, epsilon = nb_force.getParticleParameters(i)
        charges_e[i] = charge.value_in_unit(unit.elementary_charge)

    # Run coarse minimization
    integrator = openmm.LangevinMiddleIntegrator(
        config["temperature"] * unit.kelvin,
        1.0 / unit.picoseconds,
        0.002 * unit.picoseconds,
    )
    simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy(maxIterations=config["minimize_max_iterations"])

    # Get minimized state
    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
    positions_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
    box_vectors = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(
        unit.nanometers
    )

    return modeller.topology, system, positions_nm, charges_e, box_vectors


def generate_water_configurations(
    topology,
    system: openmm.System,
    base_positions_nm: np.ndarray,
    num_configs: int,
    nvt_steps: int,
    temperature: float,
    seed: int,
) -> list[np.ndarray]:
    """Generate water system configurations via short NVT runs.

    For each configuration, initializes velocities with a unique seed
    and runs a short NVT simulation.

    Parameters
    ----------
    topology : openmm.app.Topology
        System topology.
    system : openmm.System
        OpenMM system with forces.
    base_positions_nm : np.ndarray, shape (N, 3)
        Minimized positions in nanometers.
    num_configs : int
        Number of configurations to generate.
    nvt_steps : int
        Number of NVT steps per configuration (at 2 fs timestep).
    temperature : float
        Temperature in Kelvin.
    seed : int
        Base random seed for reproducibility.

    Returns
    -------
    list[np.ndarray]
        List of position arrays in nanometers, shape (N, 3).
    """
    configs = []

    for i in range(num_configs):
        integrator = openmm.LangevinMiddleIntegrator(
            temperature * unit.kelvin,
            1.0 / unit.picoseconds,
            0.002 * unit.picoseconds,
        )
        integrator.setRandomNumberSeed(seed + i)

        simulation = app.Simulation(topology, system, integrator)
        simulation.context.setPositions(base_positions_nm * unit.nanometers)
        simulation.context.setVelocitiesToTemperature(
            temperature * unit.kelvin, seed + i
        )
        simulation.step(nvt_steps)

        state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
        pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometers)
        configs.append(pos)

    return configs


# ==============================================================================
# OpenMM PME Reference
# ==============================================================================


def compute_openmm_pme_reference(
    positions_angstrom: np.ndarray,
    charges_e: np.ndarray,
    cell_angstrom: np.ndarray,
    ewald_error_tolerance: float = 1e-6,
) -> tuple[float, np.ndarray]:
    """Compute reference electrostatic energy and forces using OpenMM PME.

    Creates a pure electrostatic system (no LJ, no bonded exclusions) so that
    the result matches nvalchemiops DSF (which includes all pairs).

    Parameters
    ----------
    positions_angstrom : np.ndarray, shape (N, 3)
        Atomic positions in Angstroms.
    charges_e : np.ndarray, shape (N,)
        Atomic charges in elementary charges.
    cell_angstrom : np.ndarray, shape (3, 3)
        Unit cell matrix (row vectors) in Angstroms.
    ewald_error_tolerance : float
        PME error tolerance.

    Returns
    -------
    tuple[float, np.ndarray]
        (energy_eV, forces_eV_per_angstrom)
        Energy in eV and forces in eV/Angstrom.
    """
    system = openmm.System()
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.PME)
    nb.setEwaldErrorTolerance(ewald_error_tolerance)

    # Set periodic box vectors (convert Angstroms to nanometers)
    cell_nm = cell_angstrom / 10.0
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(*cell_nm[0]) * unit.nanometers,
        openmm.Vec3(*cell_nm[1]) * unit.nanometers,
        openmm.Vec3(*cell_nm[2]) * unit.nanometers,
    )

    # Add particles: charges only, no LJ (sigma=1, epsilon=0)
    num_atoms = len(charges_e)
    for i in range(num_atoms):
        system.addParticle(1.0)  # dummy mass
        nb.addParticle(
            charges_e[i] * unit.elementary_charge,
            1.0 * unit.angstroms,
            0.0 * unit.kilojoules_per_mole,
        )

    system.addForce(nb)

    # Create simulation context
    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    platform = openmm.Platform.getPlatformByName("CPU")
    context = openmm.Context(system, integrator, platform)

    # Set positions (convert Angstroms to nanometers)
    positions_nm = positions_angstrom / 10.0
    context.setPositions(positions_nm * unit.nanometers)

    # Get energy and forces
    state = context.getState(getEnergy=True, getForces=True)
    energy_kj_per_mol = state.getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole
    )
    forces_kj_per_mol_nm = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometers
    )

    # Convert to eV and eV/Angstrom
    energy_ev = energy_kj_per_mol * KJ_PER_MOL_TO_EV
    forces_ev_per_a = forces_kj_per_mol_nm * KJ_PER_MOL_NM_TO_EV_PER_A

    return energy_ev, forces_ev_per_a


# ==============================================================================
# nvalchemiops DSF Computation
# ==============================================================================


def compute_dsf(
    positions_angstrom: np.ndarray,
    charges_e: np.ndarray,
    cell_angstrom: np.ndarray,
    cutoff: float,
    alpha: float,
    dtype: torch.dtype,
    device: str,
    max_neighbors: int,
) -> tuple[float, np.ndarray]:
    """Compute DSF electrostatic energy and forces using nvalchemiops.

    Parameters
    ----------
    positions_angstrom : np.ndarray, shape (N, 3)
        Atomic positions in Angstroms.
    charges_e : np.ndarray, shape (N,)
        Atomic charges in elementary charges.
    cell_angstrom : np.ndarray, shape (3, 3)
        Unit cell matrix (row vectors) in Angstroms.
    cutoff : float
        Cutoff radius in Angstroms.
    alpha : float
        DSF damping parameter.
    dtype : torch.dtype
        Computation dtype (float32 or float64).
    device : str
        Torch device string.
    max_neighbors : int
        Maximum number of neighbors per atom.

    Returns
    -------
    tuple[float, np.ndarray]
        (energy_eV, forces_eV_per_angstrom)
        Energy in eV and forces in eV/Angstrom.
    """
    positions = torch.tensor(positions_angstrom, dtype=dtype, device=device)
    charges = torch.tensor(charges_e, dtype=dtype, device=device)
    cell = torch.tensor(cell_angstrom, dtype=dtype, device=device).unsqueeze(0)
    pbc = torch.tensor([True, True, True], device=device)

    # Build neighbor list
    nl_data, nl_ptr, nl_shifts = neighbor_list(
        positions,
        cutoff,
        cell=cell,
        pbc=pbc,
        return_neighbor_list=True,
        max_neighbors=max_neighbors,
    )

    # Compute DSF
    energy, forces = dsf_coulomb(
        positions=positions,
        charges=charges,
        cutoff=cutoff,
        alpha=alpha,
        cell=cell,
        neighbor_list=nl_data,
        neighbor_ptr=nl_ptr,
        unit_shifts=nl_shifts,
        compute_forces=True,
    )

    # Convert from reduced Coulomb units to eV
    energy_ev = energy.item() * KE_EVA
    forces_ev_per_a = forces.detach().cpu().numpy() * KE_EVA

    return energy_ev, forces_ev_per_a


# ==============================================================================
# Error Metrics
# ==============================================================================


def compute_error_metrics(
    energy_ref: float,
    energy_dsf: float,
    forces_ref: np.ndarray,
    forces_dsf: np.ndarray,
    num_atoms: int,
) -> dict:
    """Compute accuracy metrics between DSF and reference.

    Parameters
    ----------
    energy_ref : float
        Reference energy in eV.
    energy_dsf : float
        DSF energy in eV.
    forces_ref : np.ndarray, shape (N, 3)
        Reference forces in eV/Angstrom.
    forces_dsf : np.ndarray, shape (N, 3)
        DSF forces in eV/Angstrom.
    num_atoms : int
        Number of atoms.

    Returns
    -------
    dict
        Dictionary with error metrics.
    """
    # Energy error
    energy_abs_err = abs(energy_dsf - energy_ref)
    energy_abs_err_per_atom_mev = energy_abs_err / num_atoms * 1000.0

    # Reference force norms: ||F_ref_i||
    ref_force_norms = np.sqrt(np.sum(forces_ref**2, axis=1))
    ref_force_rms = np.sqrt(np.mean(ref_force_norms**2))
    ref_force_max = np.max(ref_force_norms)

    # Force error norms: ||F_ref_i - F_dsf_i||
    force_diff = forces_ref - forces_dsf
    force_err_norms = np.sqrt(np.sum(force_diff**2, axis=1))
    force_err_rms = np.sqrt(np.mean(force_err_norms**2))
    force_err_max = np.max(force_err_norms)

    return {
        "energy_ref_eV": energy_ref,
        "energy_dsf_eV": energy_dsf,
        "energy_err_eV": energy_abs_err,
        "energy_err_meV_per_atom": energy_abs_err_per_atom_mev,
        "force_ref_rms_eV_per_A": ref_force_rms,
        "force_ref_max_eV_per_A": ref_force_max,
        "force_err_rms_eV_per_A": force_err_rms,
        "force_err_max_eV_per_A": force_err_max,
    }


def compute_relative_energy_errors(
    energies_ref: list[float],
    energies_dsf: list[float],
    num_atoms: int,
) -> dict:
    """Compute relative energy errors from all n(n-1)/2 configuration pairs.

    For each pair (i, j) with i < j, the relative energy error is:
        |dE_dsf - dE_ref|  where dE = E_i - E_j

    This measures how well DSF reproduces energy *differences* between
    configurations, which is the relevant quantity for MLIP training.

    Parameters
    ----------
    energies_ref : list[float]
        Reference energies in eV for each configuration.
    energies_dsf : list[float]
        DSF energies in eV for each configuration.
    num_atoms : int
        Number of atoms (for per-atom normalization).

    Returns
    -------
    dict
        Dictionary with 'rel_energy_rmse_eV', 'rel_energy_rmse_meV_per_atom'.
    """
    n = len(energies_ref)
    if n < 2:
        return {
            "rel_energy_rmse_eV": 0.0,
            "rel_energy_rmse_meV_per_atom": 0.0,
        }

    errors_sq = []
    for i in range(n):
        for j in range(i + 1, n):
            de_ref = energies_ref[i] - energies_ref[j]
            de_dsf = energies_dsf[i] - energies_dsf[j]
            errors_sq.append((de_dsf - de_ref) ** 2)

    rmse = np.sqrt(np.mean(errors_sq))
    rmse_per_atom = rmse / num_atoms * 1000.0  # meV/atom

    return {
        "rel_energy_rmse_eV": rmse,
        "rel_energy_rmse_meV_per_atom": rmse_per_atom,
    }


# ==============================================================================
# Main
# ==============================================================================


def main():
    """Main entry point for the DSF accuracy benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark DSF electrostatic accuracy against OpenMM PME reference"
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to YAML configuration file"
    )
    script_dir = Path(__file__).parent
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "accuracy_results",
        help="Output directory for CSV files (default: accuracy_results/ next to script)",
    )
    parser.add_argument(
        "--gpu-sku",
        type=str,
        help="Override GPU SKU name for output files (default: auto-detect)",
    )

    args = parser.parse_args()

    if not OPENMM_AVAILABLE:
        print("ERROR: OpenMM is required for this benchmark.")
        print("Install via: pip install openmm")
        sys.exit(1)

    # Load config
    config = load_config(args.config)

    # Get parameters
    params = config["parameters"]
    num_configs = int(params["num_configurations"])
    seed = int(params["seed"])
    device_str = params.get("device", "cuda")
    dtypes_str = params.get("dtypes", ["float64"])

    # Setup device
    device = device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu"

    # Get GPU SKU
    gpu_sku = args.gpu_sku if args.gpu_sku else get_gpu_sku()

    # Create output directory
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Warp
    wp.init()

    # DSF parameter grid
    dsf_params = config["dsf_parameters"]
    cutoffs = dsf_params["cutoffs"]
    alphas = dsf_params["alphas"]

    # Neighbor list parameters
    nl_config = config.get("neighbor_list", {})
    nl_density = nl_config.get("density", 0.2)
    nl_safety = nl_config.get("safety", 1.0)

    # OpenMM reference settings
    omm_config = config.get("openmm_reference", {})
    ewald_error_tol = omm_config.get("ewald_error_tolerance", 1e-6)

    # Print configuration
    print("=" * 70)
    print("DSF ACCURACY BENCHMARK")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"GPU SKU: {gpu_sku}")
    print(f"Dtypes: {dtypes_str}")
    print(f"Configurations per system: {num_configs}")
    print(f"Random seed: {seed}")
    print(f"DSF cutoffs: {cutoffs}")
    print(f"DSF alphas: {alphas}")
    print(f"Neighbor list density: {nl_density}, safety: {nl_safety}")
    print(f"OpenMM Ewald error tolerance: {ewald_error_tol}")
    print(f"Output directory: {output_dir}")

    # ==================================================================
    # NaCl Crystal Benchmark
    # ==================================================================

    nacl_config = config.get("nacl_crystal", {})
    supercell_sizes = nacl_config.get("supercell_sizes", [6, 8, 10, 16])
    disp_std = nacl_config.get("displacement_std", 0.05)
    disp_max = nacl_config.get("displacement_max", 0.1)

    rng = np.random.default_rng(seed)

    print(f"\n{'=' * 70}")
    print("SYSTEM: NaCl Crystal")
    print(f"{'=' * 70}")
    print(f"Supercell sizes: {supercell_sizes}")
    print(f"Displacement: std={disp_std} A, max={disp_max} A")

    nacl_results = []

    for size in supercell_sizes:
        num_atoms = 8 * size**3

        # Generate configurations
        configs = generate_nacl_configurations(
            size, num_configs, disp_std, disp_max, rng
        )
        cell = configs[0]["cell"]
        min_width = compute_min_perpendicular_width(cell)

        print(
            f"\n  Supercell {size}x{size}x{size} ({num_atoms} atoms, "
            f"min width {min_width:.1f} A)"
        )

        # Compute OpenMM PME reference for all configurations
        print("    Computing OpenMM PME reference...")
        ref_results = []
        for ci, cfg in enumerate(configs):
            energy_ref, forces_ref = compute_openmm_pme_reference(
                cfg["positions"], cfg["charges"], cfg["cell"], ewald_error_tol
            )
            ref_results.append((energy_ref, forces_ref))

        # Sweep DSF parameters
        for cutoff in cutoffs:
            if min_width < 2.0 * cutoff:
                continue

            max_nbrs = estimate_max_neighbors(cutoff, nl_density, nl_safety)

            for alpha in alphas:
                for dtype_str in dtypes_str:
                    dtype = getattr(torch, dtype_str)

                    # Collect per-config metrics
                    energy_errs = []
                    energy_errs_per_atom = []
                    config_energies_ref = []
                    config_energies_dsf = []
                    force_ref_rmses = []
                    force_ref_maxes = []
                    force_err_rmses = []
                    force_err_maxes = []

                    for ci, cfg in enumerate(configs):
                        energy_ref, forces_ref = ref_results[ci]

                        energy_dsf, forces_dsf = compute_dsf(
                            cfg["positions"],
                            cfg["charges"],
                            cfg["cell"],
                            cutoff,
                            alpha,
                            dtype,
                            device,
                            max_nbrs,
                        )

                        metrics = compute_error_metrics(
                            energy_ref, energy_dsf, forces_ref, forces_dsf, num_atoms
                        )

                        energy_errs.append(metrics["energy_err_eV"])
                        energy_errs_per_atom.append(metrics["energy_err_meV_per_atom"])
                        config_energies_ref.append(metrics["energy_ref_eV"])
                        config_energies_dsf.append(metrics["energy_dsf_eV"])
                        force_ref_rmses.append(metrics["force_ref_rms_eV_per_A"])
                        force_ref_maxes.append(metrics["force_ref_max_eV_per_A"])
                        force_err_rmses.append(metrics["force_err_rms_eV_per_A"])
                        force_err_maxes.append(metrics["force_err_max_eV_per_A"])

                        nacl_results.append(
                            {
                                "system": "nacl_crystal",
                                "size": size,
                                "num_atoms": num_atoms,
                                "cutoff": cutoff,
                                "alpha": alpha,
                                "dtype": dtype_str,
                                "config_idx": ci,
                                "energy_ref_eV": metrics["energy_ref_eV"],
                                "energy_dsf_eV": metrics["energy_dsf_eV"],
                                "energy_err_eV": metrics["energy_err_eV"],
                                "energy_err_meV_per_atom": metrics[
                                    "energy_err_meV_per_atom"
                                ],
                                "force_ref_rms_eV_per_A": metrics[
                                    "force_ref_rms_eV_per_A"
                                ],
                                "force_ref_max_eV_per_A": metrics[
                                    "force_ref_max_eV_per_A"
                                ],
                                "force_err_rms_eV_per_A": metrics[
                                    "force_err_rms_eV_per_A"
                                ],
                                "force_err_max_eV_per_A": metrics[
                                    "force_err_max_eV_per_A"
                                ],
                                "forces_dsf": forces_dsf,
                            }
                        )

                    # Absolute energy errors
                    energy_rmse = np.sqrt(np.mean(np.array(energy_errs) ** 2))
                    energy_rmse_per_atom = np.sqrt(
                        np.mean(np.array(energy_errs_per_atom) ** 2)
                    )

                    # Relative energy errors (energy differences between configs)
                    rel_metrics = compute_relative_energy_errors(
                        config_energies_ref, config_energies_dsf, num_atoms
                    )

                    f_ref_rms = np.mean(force_ref_rmses)
                    f_ref_max = np.mean(force_ref_maxes)
                    f_err_rms = np.mean(force_err_rmses)
                    f_err_max = np.mean(force_err_maxes)

                    print(
                        f"    cutoff={cutoff:5.1f} alpha={alpha:.1f} "
                        f"dtype={dtype_str:7s} | "
                        f"E_err rms={energy_rmse:11.6f} eV "
                        f"({energy_rmse_per_atom:8.4f} meV/atom) | "
                        f"dE_err rms={rel_metrics['rel_energy_rmse_eV']:11.6f} eV "
                        f"({rel_metrics['rel_energy_rmse_meV_per_atom']:8.4f} meV/atom) | "
                        f"F_ref: rms={f_ref_rms:.4f} max={f_ref_max:.4f} | "
                        f"F_err: rms={f_err_rms:.6f} max={f_err_max:.6f} eV/A"
                    )

    # Save NaCl crystal results
    if nacl_results:
        output_file = output_dir / f"dsf_accuracy_nacl_crystal_{gpu_sku}.csv"
        fieldnames = [k for k in nacl_results[0].keys() if k != "forces_dsf"]
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(nacl_results)
        print(f"\n  Results saved to: {output_file}")

    all_results = list(nacl_results)

    # ==================================================================
    # NaCl in Water Benchmark
    # ==================================================================

    water_config = config.get("nacl_water", {})
    if water_config:
        print(f"\n{'=' * 70}")
        print("SYSTEM: 0.1M NaCl in Water")
        print(f"{'=' * 70}")
        print(f"Box size: {water_config.get('box_size', 4.0)} nm")
        print(f"Ionic strength: {water_config.get('ionic_strength', 0.1)} M")

        # Build water system
        print("  Building solvated system with OpenMM...")
        topology, system, base_pos_nm, charges_e, box_nm = build_nacl_water_system(
            water_config
        )
        num_atoms = len(charges_e)
        print(f"  System: {num_atoms} atoms")

        # Generate configurations
        nvt_steps = water_config.get("nvt_steps", 500)
        temperature = water_config.get("temperature", 300)
        print(
            f"  Generating {num_configs} configurations ({nvt_steps} NVT steps each)..."
        )
        water_positions_nm = generate_water_configurations(
            topology, system, base_pos_nm, num_configs, nvt_steps, temperature, seed
        )

        # Convert to Angstroms
        water_configs = []
        for pos_nm in water_positions_nm:
            positions_a = pos_nm * 10.0  # nm -> Angstrom
            cell_a = box_nm * 10.0  # nm -> Angstrom
            water_configs.append(
                {
                    "positions": positions_a,
                    "cell": cell_a,
                    "charges": charges_e,
                }
            )

        cell_a = water_configs[0]["cell"]
        min_width = compute_min_perpendicular_width(cell_a)
        print(f"  Box min width: {min_width:.1f} A")

        # Compute OpenMM PME reference
        print("  Computing OpenMM PME reference...")
        ref_results = []
        for ci, cfg in enumerate(water_configs):
            energy_ref, forces_ref = compute_openmm_pme_reference(
                cfg["positions"], cfg["charges"], cfg["cell"], ewald_error_tol
            )
            ref_results.append((energy_ref, forces_ref))

        water_results = []

        # Sweep DSF parameters
        for cutoff in cutoffs:
            if min_width < 2.0 * cutoff:
                continue

            max_nbrs = estimate_max_neighbors(cutoff, nl_density, nl_safety)

            for alpha in alphas:
                for dtype_str in dtypes_str:
                    dtype = getattr(torch, dtype_str)

                    energy_errs = []
                    energy_errs_per_atom = []
                    config_energies_ref = []
                    config_energies_dsf = []
                    force_ref_rmses = []
                    force_ref_maxes = []
                    force_err_rmses = []
                    force_err_maxes = []

                    for ci, cfg in enumerate(water_configs):
                        energy_ref, forces_ref = ref_results[ci]

                        energy_dsf, forces_dsf = compute_dsf(
                            cfg["positions"],
                            cfg["charges"],
                            cfg["cell"],
                            cutoff,
                            alpha,
                            dtype,
                            device,
                            max_nbrs,
                        )

                        metrics = compute_error_metrics(
                            energy_ref, energy_dsf, forces_ref, forces_dsf, num_atoms
                        )

                        energy_errs.append(metrics["energy_err_eV"])
                        energy_errs_per_atom.append(metrics["energy_err_meV_per_atom"])
                        config_energies_ref.append(metrics["energy_ref_eV"])
                        config_energies_dsf.append(metrics["energy_dsf_eV"])
                        force_ref_rmses.append(metrics["force_ref_rms_eV_per_A"])
                        force_ref_maxes.append(metrics["force_ref_max_eV_per_A"])
                        force_err_rmses.append(metrics["force_err_rms_eV_per_A"])
                        force_err_maxes.append(metrics["force_err_max_eV_per_A"])

                        water_results.append(
                            {
                                "system": "nacl_water",
                                "label": f"{water_config.get('ionic_strength', 0.1)}M_NaCl",
                                "num_atoms": num_atoms,
                                "cutoff": cutoff,
                                "alpha": alpha,
                                "dtype": dtype_str,
                                "config_idx": ci,
                                "energy_ref_eV": metrics["energy_ref_eV"],
                                "energy_dsf_eV": metrics["energy_dsf_eV"],
                                "energy_err_eV": metrics["energy_err_eV"],
                                "energy_err_meV_per_atom": metrics[
                                    "energy_err_meV_per_atom"
                                ],
                                "force_ref_rms_eV_per_A": metrics[
                                    "force_ref_rms_eV_per_A"
                                ],
                                "force_ref_max_eV_per_A": metrics[
                                    "force_ref_max_eV_per_A"
                                ],
                                "force_err_rms_eV_per_A": metrics[
                                    "force_err_rms_eV_per_A"
                                ],
                                "force_err_max_eV_per_A": metrics[
                                    "force_err_max_eV_per_A"
                                ],
                                "forces_dsf": forces_dsf,
                            }
                        )

                    energy_rmse = np.sqrt(np.mean(np.array(energy_errs) ** 2))
                    energy_rmse_per_atom = np.sqrt(
                        np.mean(np.array(energy_errs_per_atom) ** 2)
                    )

                    rel_metrics = compute_relative_energy_errors(
                        config_energies_ref, config_energies_dsf, num_atoms
                    )

                    f_ref_rms = np.mean(force_ref_rmses)
                    f_ref_max = np.mean(force_ref_maxes)
                    f_err_rms = np.mean(force_err_rmses)
                    f_err_max = np.mean(force_err_maxes)

                    print(
                        f"    cutoff={cutoff:5.1f} alpha={alpha:.1f} "
                        f"dtype={dtype_str:7s} | "
                        f"E_err rms={energy_rmse:11.6f} eV "
                        f"({energy_rmse_per_atom:8.4f} meV/atom) | "
                        f"dE_err rms={rel_metrics['rel_energy_rmse_eV']:11.6f} eV "
                        f"({rel_metrics['rel_energy_rmse_meV_per_atom']:8.4f} meV/atom) | "
                        f"F_ref: rms={f_ref_rms:.4f} max={f_ref_max:.4f} | "
                        f"F_err: rms={f_err_rms:.6f} max={f_err_max:.6f} eV/A"
                    )

        # Save water results
        if water_results:
            output_file = output_dir / f"dsf_accuracy_nacl_water_{gpu_sku}.csv"
            fieldnames = [k for k in water_results[0].keys() if k != "forces_dsf"]
            with open(output_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(water_results)
            print(f"\n  Results saved to: {output_file}")
            all_results.extend(water_results)

    # ==================================================================
    # fp32 vs fp64 Analysis
    # ==================================================================
    if (
        all_results
        and len(dtypes_str) >= 2
        and "float32" in dtypes_str
        and "float64" in dtypes_str
    ):
        print(f"\n{'=' * 70}")
        print("PRECISION ANALYSIS: float32 vs float64")
        print(f"{'=' * 70}")

        # Per-system breakdown of fp32 vs fp64 differences
        for sys_name in ["nacl_crystal", "nacl_water"]:
            sys_results = [r for r in all_results if r["system"] == sys_name]
            if not sys_results:
                continue

            sys_grouped = defaultdict(dict)
            for r in sys_results:
                if sys_name == "nacl_crystal":
                    key = (r["size"], r["cutoff"], r["alpha"], r["config_idx"])
                else:
                    key = (r["label"], r["cutoff"], r["alpha"], r["config_idx"])
                sys_grouped[key][r["dtype"]] = r

            energy_abs_diffs = []
            energy_per_atom_diffs = []
            force_diff_rmses = []
            force_diff_maxes = []

            for key, dtypes in sys_grouped.items():
                if "float32" not in dtypes or "float64" not in dtypes:
                    continue
                f32 = dtypes["float32"]
                f64 = dtypes["float64"]
                n_atoms = f64["num_atoms"]

                # Absolute energy difference
                e_diff = abs(f32["energy_dsf_eV"] - f64["energy_dsf_eV"])
                energy_abs_diffs.append(e_diff)
                # Per-atom: divide by sqrt(N) since errors are normally distributed
                energy_per_atom_diffs.append(e_diff / math.sqrt(n_atoms) * 1000.0)

                # Direct force difference: ||F_fp32 - F_fp64|| per atom
                force_diff = f32["forces_dsf"] - f64["forces_dsf"]
                force_diff_norms = np.sqrt(np.sum(force_diff**2, axis=1))
                force_diff_rmses.append(np.sqrt(np.mean(force_diff_norms**2)))
                force_diff_maxes.append(np.max(force_diff_norms))

            if energy_abs_diffs:
                energy_abs_diffs = np.array(energy_abs_diffs)
                energy_per_atom_diffs = np.array(energy_per_atom_diffs)
                force_diff_rmses = np.array(force_diff_rmses)
                force_diff_maxes = np.array(force_diff_maxes)

                label = "NaCl Crystal" if sys_name == "nacl_crystal" else "NaCl Water"
                print(f"\n  {label} ({len(energy_abs_diffs)} matched pairs):")

                print("    |E_fp32 - E_fp64| (eV):")
                print(f"      mean = {np.mean(energy_abs_diffs):.4e}")
                print(f"      max  = {np.max(energy_abs_diffs):.4e}")
                print("    |E_fp32 - E_fp64| / sqrt(N) (meV/atom):")
                print(f"      mean = {np.mean(energy_per_atom_diffs):.4e}")
                print(f"      max  = {np.max(energy_per_atom_diffs):.4e}")

                print("    RMS(||F_fp32 - F_fp64||) (eV/A):")
                print(f"      mean = {np.mean(force_diff_rmses):.4e}")
                print(f"      max  = {np.max(force_diff_rmses):.4e}")
                print("    max(||F_fp32 - F_fp64||) (eV/A):")
                print(f"      mean = {np.mean(force_diff_maxes):.4e}")
                print(f"      max  = {np.max(force_diff_maxes):.4e}")

    # ==================================================================
    # Summary
    # ==================================================================

    print(f"\n{'=' * 70}")
    print("BENCHMARK COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
