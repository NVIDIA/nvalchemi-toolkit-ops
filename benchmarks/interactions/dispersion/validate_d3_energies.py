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
DFT-D3 Validation Script - Warp Kernels vs torch-dftd

This script validates the Warp kernel implementation of DFT-D3(BJ) dispersion
corrections against the torch-dftd reference implementation. It computes and
compares coordination numbers, C6 coefficients, and dispersion energies for
user-provided molecular structures.

The validation includes:
- Coordination number (CN) comparison
- C6 coefficient comparison
- Dispersion energy comparison (two-body only vs two-body + three-body)
- Detailed error analysis
- Automatic issue tracking in VALIDATION_ISSUES.md
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import click
import numpy as np
import torch
from pymatgen.core import Structure

from nvalchemiops.torch.interactions.dispersion import dftd3
from nvalchemiops.torch.neighbors import neighbor_list

# ==============================================================================
# Functional Parameters
# ==============================================================================

FUNCTIONAL_PARAMS = {
    "b3lyp": {
        "name": "B3LYP-D3(BJ)",
        "s6": 1.0,
        "s8": 1.9889,  # s18 in torch-dftd
        "a1": 0.3981,  # rs6 in torch-dftd
        "a2": 4.4211,  # rs18 in torch-dftd (Bohr)
        "k1": 16.0,
        "k3": -4.0,
    },
    "pbe": {
        "name": "PBE-D3(BJ)",
        "s6": 1.0,
        "s8": 0.7875,
        "a1": 0.4289,
        "a2": 4.4407,
        "k1": 16.0,
        "k3": -4.0,
    },
}

# Constants
ANGSTROM_TO_BOHR = 1.88973


@dataclass
class ValidationConfig:
    """Configuration for validation run."""

    input_file: Path
    functional: str
    cutoff: float | None
    cnthr: float | None
    device: str
    dtype: str
    output_format: Literal["markdown", "json"]


@dataclass
class WarpResults:
    """Results from Warp kernel computation."""

    coord_num: np.ndarray  # [num_atoms]
    energy: float  # Hartree
    energy_per_atom: np.ndarray  # [num_atoms]
    force: np.ndarray  # [num_atoms, 3]
    c6_values: dict = field(default_factory=dict)  # {(i,j): c6_ij}


@dataclass
class TorchDFTDResults:
    """Results from torch-dftd computation."""

    coord_num: np.ndarray  # [num_atoms]
    energy_2body: float  # Hartree
    energy_3body: float | None  # Hartree (if abc=True)
    force: np.ndarray  # [num_atoms, 3] (Hartree/Bohr)
    c6_values: dict = field(default_factory=dict)  # {(i,j): c6_ij}


@dataclass
class ComparisonResults:
    """Comparison statistics between implementations."""

    cn_abs_error: float
    cn_rel_error: float
    cn_rms_error: float
    energy_abs_error_2body: float
    energy_rel_error_2body: float
    force_abs_error: float
    force_rel_error: float
    force_rms_error: float
    force_max_error: float
    three_body_contribution: float | None
    issues_identified: list[str] = field(default_factory=list)


# ==============================================================================
# Warp Computation
# ==============================================================================


def load_torch_dftd_parameters() -> dict:
    """
    Load DFT-D3 reference parameters from torch-dftd.

    Returns
    -------
    dict
        Dictionary containing torch tensors:
        - c6ab: C6 reference values [95, 95, 5, 5, 3]
        - r0ab: R0 reference values [95, 95] (not used in BJ damping)
        - rcov: Covalent radii [95] in Angstrom
        - r2r4: <r²>/<r⁴> expectation values [95] (torch-dftd uses r2r4, we need r4r2)
    """
    try:
        import os
        from pathlib import Path

        import torch_dftd
    except ImportError:
        raise ImportError(
            "torch-dftd not installed. Install via: pip install torch-dftd"
        )

    # Load parameters from the .npz file in torch_dftd package
    d3_filepath = str(
        Path(os.path.abspath(torch_dftd.__file__)).parent
        / "nn"
        / "params"
        / "dftd3_params.npz"
    )
    d3_params = np.load(d3_filepath)

    # Extract needed parameters and convert to torch tensors
    c6ab = torch.tensor(
        d3_params["c6ab"], dtype=torch.float32
    )  # Shape: [95, 95, 5, 5, 3]
    r0ab = torch.tensor(d3_params["r0ab"], dtype=torch.float32)  # Shape: [95, 95]
    rcov = torch.tensor(d3_params["rcov"], dtype=torch.float32)  # Shape: [95]
    r2r4 = torch.tensor(d3_params["r2r4"], dtype=torch.float32)  # Shape: [95]

    # Note: Keep r2r4 as-is (don't invert)
    # The Warp kernel variable name "r4r2" is misleading - it actually expects r2r4 values
    return {
        "c6ab": c6ab,
        "r0ab": r0ab,
        "rcov": rcov,
        "r4r2": r2r4,  # Despite the name, this should be r2r4 values
    }


def run_warp_dftd3(
    structure: Structure,
    neighbor_matrix: torch.Tensor,
    params: dict,
    functional_params: dict,
    device: torch.device,
    dtype: str,
) -> WarpResults:
    """
    Execute Warp DFT-D3 computation using the PyTorch wrapper.

    Parameters
    ----------
    structure : Structure
        pymatgen Structure object
    neighbor_matrix : torch.Tensor
        Neighbor matrix [num_atoms, max_neighbors]
    params : dict
        DFT-D3 parameters from torch-dftd
    functional_params : dict
        Functional parameters (s6, s8, a1, a2, k1, k3)
    device : torch.device
        PyTorch device
    dtype : str
        Data type string ('float32', 'float64', 'float16')

    Returns
    -------
    WarpResults
        Coordination numbers, energy, and forces
    """
    # Convert dtype string to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    torch_dtype = dtype_map[dtype]

    # Get positions and numbers
    positions = structure.cart_coords  # Angstrom
    numbers = np.array([site.specie.Z for site in structure], dtype=np.int32)

    # Convert to Bohr and create torch tensors
    positions_bohr = torch.tensor(
        positions * ANGSTROM_TO_BOHR, dtype=torch_dtype, device=device
    )
    numbers_tensor = torch.tensor(numbers, dtype=torch.int32, device=device)

    # Extract parameters from torch-dftd format
    c6ab = params["c6ab"]  # [95, 95, 5, 5, 3]
    rcov = params["rcov"]  # [95] in Angstrom
    r4r2 = params["r4r2"]  # [95]

    # Split C6 reference arrays
    # c6ab[:,:,:,:,0] = C6 values
    # c6ab[:,:,:,:,1] = CN_i reference
    # c6ab[:,:,:,:,2] = CN_j reference
    # The new API expects c6_reference [95, 95, 5, 5] and coord_num_ref [95, 95, 5, 5]
    c6_reference = c6ab[:, :, :, :, 0].to(dtype=torch_dtype, device=device)
    # For symmetric CN reference, average CN_i and CN_j
    coord_num_ref = (c6ab[:, :, :, :, 1] + c6ab[:, :, :, :, 2]) / 2.0
    coord_num_ref = coord_num_ref.to(dtype=torch_dtype, device=device)

    # Convert rcov from Angstrom to Bohr
    rcov_bohr = (rcov * ANGSTROM_TO_BOHR).to(dtype=torch_dtype, device=device)

    # Convert r4r2 to correct dtype and device
    r4r2_tensor = r4r2.to(dtype=torch_dtype, device=device)

    # Convert neighbor_matrix to correct dtype
    neighbor_matrix_tensor = neighbor_matrix.to(dtype=torch.int32, device=device)

    # Call the PyTorch wrapper with new API
    num_atoms = len(structure)
    result = dftd3(
        positions=positions_bohr,
        numbers=numbers_tensor,
        neighbor_matrix=neighbor_matrix_tensor,
        covalent_radii=rcov_bohr,
        r4r2=r4r2_tensor,
        c6_reference=c6_reference,
        coord_num_ref=coord_num_ref,
        fill_value=num_atoms,
        k1=functional_params["k1"],
        k3=functional_params["k3"],
        a1=functional_params["a1"],
        a2=functional_params["a2"],
        s6=functional_params["s6"],
        s8=functional_params["s8"],
        device=str(device),
    )

    # Handle return value - dftd3 returns (energy, forces, coord_num) or (energy, forces, coord_num, virial)
    if len(result) == 4:
        energy, forces, coord_num, _ = result
    else:
        energy, forces, coord_num = result

    # Convert results to numpy
    coord_num_np = coord_num.detach().cpu().numpy()
    energy_scalar = float(energy.detach().cpu().numpy()[0])
    force_np = forces.detach().cpu().numpy()

    return WarpResults(
        coord_num=coord_num_np,
        energy=energy_scalar,
        energy_per_atom=np.zeros(num_atoms),  # Not returned by new API
        force=force_np,
    )


# ==============================================================================
# torch-dftd Computation
# ==============================================================================


def run_torch_dftd(
    structure: Structure,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    functional_params: dict,
    cutoff: float | None,
    cnthr: float | None,
    device: torch.device,
    abc: bool = False,
) -> tuple[float, np.ndarray, np.ndarray, dict]:
    """
    Run torch-dftd dispersion calculation with force computation via autograd.

    Parameters
    ----------
    structure : Structure
        pymatgen Structure object
    edge_src : torch.Tensor
        Source atom indices from neighbor_list
    edge_dst : torch.Tensor
        Destination atom indices from neighbor_list
    functional_params : dict
        Functional parameters
    cutoff : float or None
        Cutoff radius in Bohr (None = no cutoff)
    cnthr : float or None
        CN calculation cutoff in Bohr (None = no cutoff)
    device : torch.device
        PyTorch device
    abc : bool
        Include three-body terms

    Returns
    -------
    energy : float
        Dispersion energy in Hartree
    coord_num : np.ndarray
        Coordination numbers [num_atoms]
    force : np.ndarray
        Forces in Hartree/Bohr [num_atoms, 3]
    c6_dict : dict
        C6 coefficients {(i, j): c6_ij}
    """
    from torch_dftd.functions.dftd3 import _getc6, _ncoord, edisp
    from torch_dftd.functions.distance import calc_distances

    # Get positions - enable gradient tracking for force computation
    positions = torch.tensor(
        structure.cart_coords * ANGSTROM_TO_BOHR,
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    atomic_numbers = np.array([site.specie.Z for site in structure], dtype=np.int64)
    Z = torch.tensor(atomic_numbers, dtype=torch.int64, device=device)

    # Build edge index from provided edge lists (already computed by neighbor_list)
    edge_index = torch.stack([edge_src, edge_dst], dim=0)

    # Calculate distances
    r = calc_distances(positions, edge_index, cell=None, shift_pos=None)

    # Load parameters from torch_dftd
    import os
    from pathlib import Path

    import torch_dftd

    d3_filepath = str(
        Path(os.path.abspath(torch_dftd.__file__)).parent
        / "nn"
        / "params"
        / "dftd3_params.npz"
    )
    d3_params_np = np.load(d3_filepath)

    c6ab = torch.tensor(d3_params_np["c6ab"], dtype=torch.float32, device=device)
    r0ab = torch.tensor(d3_params_np["r0ab"], dtype=torch.float32, device=device)
    rcov = torch.tensor(d3_params_np["rcov"], dtype=torch.float32, device=device)
    r2r4 = torch.tensor(d3_params_np["r2r4"], dtype=torch.float32, device=device)

    # Convert rcov to Bohr
    rcov = rcov * ANGSTROM_TO_BOHR

    # Prepare params dict for edisp
    params = {
        "s6": functional_params["s6"],
        "s18": functional_params["s8"],
        "rs6": functional_params["a1"],
        "rs18": functional_params["a2"],
        "alp": 14.0,  # Default for BJ damping (not used)
    }

    # For abc=True, we need a cutoff for 3-body terms
    # Use a reasonable default if not provided
    if cnthr is not None:
        cnthr_effective = cnthr
    elif cutoff is not None:
        cnthr_effective = cutoff
    else:
        cnthr_effective = 50.0 * ANGSTROM_TO_BOHR  # 50 Angstrom default

    # Compute energy
    energy_tensor = edisp(
        Z=Z,
        r=r,
        edge_index=edge_index,
        c6ab=c6ab,
        r0ab=r0ab,
        rcov=rcov,
        r2r4=r2r4,
        params=params,
        cutoff=cutoff,
        cnthr=cnthr_effective,
        batch=None,
        batch_edge=None,
        shift_pos=None,
        pos=positions,
        cell=None,
        damping="bj",
        bidirectional=True,
        abc=abc,
        k1=functional_params["k1"],
        k3=functional_params["k3"],
    )

    energy = float(energy_tensor.detach().cpu().numpy()[0])

    # Compute forces via autograd (negative gradient of energy w.r.t. positions)
    # Forces = -dE/dr
    force_tensor = -torch.autograd.grad(
        outputs=energy_tensor,
        inputs=positions,
        create_graph=False,
        retain_graph=False,
    )[0]
    force = force_tensor.cpu().detach().numpy()

    # Compute CN separately for extraction
    idx_i, idx_j = edge_index
    cn = _ncoord(
        Z=Z,
        r=r,
        idx_i=idx_i,
        idx_j=idx_j,
        rcov=rcov,
        cutoff=cnthr,
        k1=functional_params["k1"],
        bidirectional=True,
    )

    # Compute C6 for extraction
    nci = cn[idx_i]
    ncj = cn[idx_j]
    z_i = Z[idx_i]
    z_j = Z[idx_j]
    c6 = _getc6(z_i, z_j, nci, ncj, c6ab=c6ab, k3=functional_params["k3"])

    # Build C6 dictionary
    c6_dict = {}
    for i, j, c6_val in zip(
        idx_i.cpu().numpy(), idx_j.cpu().numpy(), c6.detach().cpu().numpy()
    ):
        if i < j:  # Store only unique pairs
            c6_dict[(i, j)] = float(c6_val)

    return energy, cn.detach().cpu().numpy(), force, c6_dict


# ==============================================================================
# Comparison and Analysis
# ==============================================================================


def compare_results(
    warp_results: WarpResults,
    torch_2body_results: TorchDFTDResults,
    torch_3body_results: TorchDFTDResults | None,
) -> ComparisonResults:
    """
    Compare Warp and torch-dftd results.

    Parameters
    ----------
    warp_results : WarpResults
        Results from Warp kernels
    torch_2body_results : TorchDFTDResults
        Results from torch-dftd (two-body only)
    torch_3body_results : TorchDFTDResults or None
        Results from torch-dftd (two-body + three-body)

    Returns
    -------
    ComparisonResults
        Comparison statistics and identified issues
    """
    # CN comparison
    cn_diff = warp_results.coord_num - torch_2body_results.coord_num
    cn_abs_error = float(np.mean(np.abs(cn_diff)))
    cn_rel_error = float(
        np.mean(np.abs(cn_diff) / (np.abs(torch_2body_results.coord_num) + 1e-10))
    )
    cn_rms_error = float(np.sqrt(np.mean(cn_diff**2)))

    # Energy comparison
    energy_abs_error_2body = abs(warp_results.energy - torch_2body_results.energy_2body)
    energy_rel_error_2body = energy_abs_error_2body / (
        abs(torch_2body_results.energy_2body) + 1e-10
    )

    # Force comparison
    force_diff = warp_results.force - torch_2body_results.force
    force_abs_error = float(np.mean(np.abs(force_diff)))
    force_rel_error = float(
        np.mean(np.abs(force_diff) / (np.abs(torch_2body_results.force) + 1e-10))
    )
    force_rms_error = float(np.sqrt(np.mean(force_diff**2)))
    force_max_error = float(np.max(np.abs(force_diff)))

    # Three-body contribution
    three_body_contribution = None
    if torch_3body_results is not None:
        three_body_contribution = (
            torch_3body_results.energy_3body - torch_2body_results.energy_2body
        )

    # Identify issues
    issues = []
    if cn_abs_error > 0.1:
        issues.append(f"Large CN discrepancy: mean absolute error = {cn_abs_error:.4f}")
    if energy_rel_error_2body > 0.01:
        issues.append(
            f"Large energy discrepancy: relative error = {energy_rel_error_2body:.4%}"
        )
    if force_abs_error > 0.01:
        issues.append(
            f"Large force discrepancy: mean absolute error = {force_abs_error:.6f} Hartree/Bohr"
        )

    return ComparisonResults(
        cn_abs_error=cn_abs_error,
        cn_rel_error=cn_rel_error,
        cn_rms_error=cn_rms_error,
        energy_abs_error_2body=energy_abs_error_2body,
        energy_rel_error_2body=energy_rel_error_2body,
        force_abs_error=force_abs_error,
        force_rel_error=force_rel_error,
        force_rms_error=force_rms_error,
        force_max_error=force_max_error,
        three_body_contribution=three_body_contribution,
        issues_identified=issues,
    )


# ==============================================================================
# Output Formatting
# ==============================================================================


def format_markdown_report(
    config: ValidationConfig,
    structure: Structure,
    functional_params: dict,
    warp_results: WarpResults,
    torch_2body_results: TorchDFTDResults,
    torch_3body_results: TorchDFTDResults | None,
    comparison: ComparisonResults,
) -> str:
    """Generate markdown validation report."""
    lines = []
    lines.append("# DFT-D3 Validation Report\n")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Section 1: System Info
    lines.append("## System Information\n")
    lines.append(f"- **Input file**: {config.input_file}")
    lines.append(f"- **Number of atoms**: {len(structure)}")
    chemical_symbols = [site.specie.symbol for site in structure]
    lines.append(f"- **Elements**: {set(chemical_symbols)}")
    lines.append("- **Periodic boundary conditions**: [True, True, True]")
    lines.append(f"- **Functional**: {functional_params['name']}")
    lines.append(f"- **Device**: {config.device}")
    lines.append(f"- **Precision**: {config.dtype}\n")

    # Functional parameters
    lines.append("### Functional Parameters\n")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| s6 | {functional_params['s6']:.4f} |")
    lines.append(f"| s8 | {functional_params['s8']:.4f} |")
    lines.append(f"| a1 | {functional_params['a1']:.4f} |")
    lines.append(f"| a2 | {functional_params['a2']:.4f} Bohr |")
    lines.append(f"| k1 | {functional_params['k1']:.4f} Bohr⁻¹ |")
    lines.append(f"| k3 | {functional_params['k3']:.4f} |\n")

    # Section 2: Coordination Numbers
    lines.append("## Coordination Number Comparison\n")
    lines.append("| Atom | Element | Warp CN | torch-dftd CN | Difference |")
    lines.append("|------|---------|---------|---------------|------------|")
    chemical_symbols = [site.specie.symbol for site in structure]
    for i, symbol in enumerate(chemical_symbols):
        warp_cn = warp_results.coord_num[i]
        torch_cn = torch_2body_results.coord_num[i]
        diff = warp_cn - torch_cn
        lines.append(
            f"| {i} | {symbol} | {warp_cn:.4f} | {torch_cn:.4f} | {diff:+.4f} |"
        )
    lines.append("")

    lines.append("### CN Statistics\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Mean absolute error | {comparison.cn_abs_error:.6f} |")
    lines.append(f"| Mean relative error | {comparison.cn_rel_error:.6f} |")
    lines.append(f"| RMS error | {comparison.cn_rms_error:.6f} |\n")

    # Section 3: Energies
    lines.append("## Dispersion Energy Comparison\n")
    lines.append("| Implementation | Energy (Hartree) | Energy (kcal/mol) |")
    lines.append("|----------------|------------------|-------------------|")

    hartree_to_kcal = 627.509474
    lines.append(
        f"| Warp (2-body) | {warp_results.energy:.8f} | {warp_results.energy * hartree_to_kcal:.6f} |"
    )
    lines.append(
        f"| torch-dftd (2-body) | {torch_2body_results.energy_2body:.8f} | {torch_2body_results.energy_2body * hartree_to_kcal:.6f} |"
    )
    if torch_3body_results is not None:
        lines.append(
            f"| torch-dftd (2-body + 3-body) | {torch_3body_results.energy_3body:.8f} | {torch_3body_results.energy_3body * hartree_to_kcal:.6f} |"
        )
    lines.append("")

    lines.append("### Energy Error Analysis\n")
    lines.append("| Metric | Value (Hartree) | Value (kcal/mol) |")
    lines.append("|--------|-----------------|------------------|")
    lines.append(
        f"| Absolute error (2-body) | {comparison.energy_abs_error_2body:.8f} | {comparison.energy_abs_error_2body * hartree_to_kcal:.6f} |"
    )
    lines.append(
        f"| Relative error (2-body) | {comparison.energy_rel_error_2body:.6%} | - |"
    )
    if comparison.three_body_contribution is not None:
        lines.append(
            f"| 3-body contribution | {comparison.three_body_contribution:.8f} | {comparison.three_body_contribution * hartree_to_kcal:.6f} |"
        )
    lines.append("")

    # Section 4: Force Comparison
    lines.append("## Force Comparison\n")
    lines.append(
        "| Atom | Element | Warp Force (Ha/Bohr) | torch-dftd Force (Ha/Bohr) | Difference (Ha/Bohr) |"
    )
    lines.append(
        "|------|---------|----------------------|----------------------------|----------------------|"
    )
    chemical_symbols = [site.specie.symbol for site in structure]
    for i, symbol in enumerate(chemical_symbols):
        warp_f = warp_results.force[i]
        torch_f = torch_2body_results.force[i]
        diff_f = warp_f - torch_f
        # Format as vector norm
        warp_norm = float(np.linalg.norm(warp_f))
        torch_norm = float(np.linalg.norm(torch_f))
        diff_norm = float(np.linalg.norm(diff_f))
        lines.append(
            f"| {i} | {symbol} | {warp_norm:.6f} | {torch_norm:.6f} | {diff_norm:.6f} |"
        )
    lines.append("")

    lines.append("### Force Statistics\n")
    lines.append("| Metric | Value (Hartree/Bohr) | Value (kcal/mol/Å) |")
    lines.append("|--------|----------------------|--------------------|")
    # Convert Hartree/Bohr to kcal/mol/Å: (Hartree/Bohr) * (kcal/Hartree) / (Å/Bohr)
    force_conversion = hartree_to_kcal / ANGSTROM_TO_BOHR
    lines.append(
        f"| Mean absolute error | {comparison.force_abs_error:.8f} | {comparison.force_abs_error * force_conversion:.6f} |"
    )
    lines.append(f"| Mean relative error | {comparison.force_rel_error:.6f} | - |")
    lines.append(
        f"| RMS error | {comparison.force_rms_error:.8f} | {comparison.force_rms_error * force_conversion:.6f} |"
    )
    lines.append(
        f"| Maximum error | {comparison.force_max_error:.8f} | {comparison.force_max_error * force_conversion:.6f} |"
    )
    lines.append("")

    # Section 5: Issues Identified
    if comparison.issues_identified:
        lines.append("## Issues Identified\n")
        lines.extend(f"- {issue}" for issue in comparison.issues_identified)
        lines.append("")

    lines.append("---")
    return "\n".join(lines)


def format_json_report(
    config: ValidationConfig,
    structure: Structure,
    functional_params: dict,
    warp_results: WarpResults,
    torch_2body_results: TorchDFTDResults,
    torch_3body_results: TorchDFTDResults | None,
    comparison: ComparisonResults,
) -> str:
    """Generate JSON validation report."""
    chemical_symbols = [site.specie.symbol for site in structure]
    data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "input_file": str(config.input_file),
            "functional": functional_params["name"],
            "device": config.device,
            "dtype": config.dtype,
        },
        "system": {
            "num_atoms": len(structure),
            "elements": list(set(chemical_symbols)),
            "pbc": [True, True, True],
        },
        "parameters": functional_params,
        "coordination_numbers": {
            "warp": warp_results.coord_num.tolist(),
            "torch_dftd": torch_2body_results.coord_num.tolist(),
            "statistics": {
                "mean_abs_error": comparison.cn_abs_error,
                "mean_rel_error": comparison.cn_rel_error,
                "rms_error": comparison.cn_rms_error,
            },
        },
        "energies": {
            "warp_2body_hartree": warp_results.energy,
            "torch_dftd_2body_hartree": torch_2body_results.energy_2body,
            "torch_dftd_3body_hartree": (
                torch_3body_results.energy_3body if torch_3body_results else None
            ),
            "error_2body_hartree": comparison.energy_abs_error_2body,
            "relative_error_2body": comparison.energy_rel_error_2body,
            "three_body_contribution_hartree": comparison.three_body_contribution,
        },
        "forces": {
            "warp_forces_hartree_bohr": warp_results.force.tolist(),
            "torch_dftd_forces_hartree_bohr": torch_2body_results.force.tolist(),
            "statistics": {
                "mean_abs_error_hartree_bohr": comparison.force_abs_error,
                "mean_rel_error": comparison.force_rel_error,
                "rms_error_hartree_bohr": comparison.force_rms_error,
                "max_error_hartree_bohr": comparison.force_max_error,
            },
        },
        "issues": comparison.issues_identified,
    }
    return json.dumps(data, indent=2)


# ==============================================================================
# Main Workflow
# ==============================================================================


def build_neighbor_list(
    structure: Structure, cutoff_ang: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Build neighbor list for structure using internal neighbor_list function.

    Parameters
    ----------
    structure : Structure
        pymatgen Structure object
    cutoff_ang : float
        Cutoff radius in Angstrom
    device : torch.device
        PyTorch device

    Returns
    -------
    edge_src : torch.Tensor
        Source atom indices [num_edges]
    edge_dst : torch.Tensor
        Destination atom indices [num_edges]
    neighbor_matrix : torch.Tensor
        Neighbor matrix [num_atoms, max_neighbors] for Warp
    max_neighbors : int
        Maximum neighbors per atom
    """
    num_atoms = len(structure)
    positions = structure.cart_coords

    # Prepare inputs for neighbor_list
    coord = torch.tensor(positions, dtype=torch.float32, device=device)
    pbc = torch.tensor([True, True, True], dtype=torch.bool, device=device)
    cell = torch.tensor(structure.lattice.matrix, dtype=torch.float32, device=device)

    # Use internal neighbor_list function with new API
    # Returns: neighbor_matrix, num_neighbors, neighbor_matrix_shifts
    neighbor_matrix, num_neighbors, _ = neighbor_list(
        coord,
        cutoff_ang,
        cell=cell,
        pbc=pbc,
        method="cell_list",
        return_neighbor_list=False,
    )

    # Also build edge list for torch-dftd
    edge_src_list = []
    edge_dst_list = []

    for i in range(num_atoms):
        n_neighbors = int(num_neighbors[i].item())
        for k in range(n_neighbors):
            j = int(neighbor_matrix[i, k].item())
            if j < num_atoms:  # Valid neighbor (not padding)
                edge_src_list.append(i)
                edge_dst_list.append(j)

    edge_src = torch.tensor(edge_src_list, dtype=torch.int64, device=device)
    edge_dst = torch.tensor(edge_dst_list, dtype=torch.int64, device=device)

    max_neighbors = int(neighbor_matrix.shape[1])

    return edge_src, edge_dst, neighbor_matrix, max_neighbors


@click.command()
@click.option(
    "--input",
    "input_file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="XYZ or CIF file path",
)
@click.option(
    "--functional",
    type=click.Choice(["b3lyp", "pbe"], case_sensitive=False),
    default="b3lyp",
    help="DFT functional (default: b3lyp)",
)
@click.option(
    "--cutoff",
    type=float,
    default=None,
    help="Cutoff radius in Angstrom (default: None = no cutoff)",
)
@click.option(
    "--cnthr",
    type=float,
    default=None,
    help="CN calculation cutoff in Angstrom (default: None = no cutoff)",
)
@click.option(
    "--device",
    type=str,
    default="cuda:0",
    help="Compute device (default: cuda:0)",
)
@click.option(
    "--dtype",
    type=click.Choice(["float32", "float64", "float16"]),
    default="float32",
    help="Floating point precision (default: float32)",
)
@click.option(
    "--output-format",
    type=click.Choice(["markdown", "json"]),
    default="markdown",
    help="Output format (default: markdown)",
)
def main(
    input_file: Path,
    functional: str,
    cutoff: float | None,
    cnthr: float | None,
    device: str,
    dtype: str,
    output_format: str,
) -> None:
    """
    Validate DFT-D3 Warp kernels against torch-dftd reference.

    This script compares coordination numbers, C6 coefficients, and dispersion
    energies computed by Warp kernels and torch-dftd for a given molecular structure.

    Examples:

        # Validate water molecule with B3LYP
        python validate_d3_energies.py --input water.xyz

        # Validate ice crystal with custom cutoffs
        python validate_d3_energies.py --input ice.cif --cutoff 20.0 --cnthr 20.0

        # Generate JSON output without updating issues file
        python validate_d3_energies.py --input molecule.xyz --output-format json --no-update-issues
    """
    config = ValidationConfig(
        input_file=input_file,
        functional=functional.lower(),
        cutoff=cutoff * ANGSTROM_TO_BOHR if cutoff is not None else None,
        cnthr=cnthr * ANGSTROM_TO_BOHR if cnthr is not None else None,
        device=device,
        dtype=dtype,
        output_format=output_format,
    )

    # Initialize torch device
    torch_device = torch.device(device if device != "cpu" else "cpu")

    # Get functional parameters
    functional_params = FUNCTIONAL_PARAMS[config.functional]

    click.echo(f"Loading structure from {input_file}...")
    structure = Structure.from_file(str(input_file))
    chemical_symbols = [site.specie.symbol for site in structure]
    click.echo(f"  {len(structure)} atoms, elements: {set(chemical_symbols)}")

    # Build neighbor list
    cutoff_ang = config.cutoff / ANGSTROM_TO_BOHR if config.cutoff is not None else 20.0
    click.echo(f"Building neighbor list (cutoff = {cutoff_ang:.2f} Å)...")
    edge_src, edge_dst, neighbor_matrix, max_neighbors = build_neighbor_list(
        structure, cutoff_ang, torch_device
    )
    click.echo(f"  Max neighbors per atom: {max_neighbors}")
    click.echo(f"  Total edges: {len(edge_src)}")

    # Load reference parameters
    click.echo("Loading DFT-D3 reference parameters from torch-dftd...")
    ref_params = load_torch_dftd_parameters()

    # Run Warp computation using PyTorch wrapper
    click.echo("Running Warp DFT-D3 kernels...")
    warp_results = run_warp_dftd3(
        structure,
        neighbor_matrix,
        ref_params,
        functional_params,
        torch_device,
        dtype,
    )
    click.echo(
        f"  Warp energy: {warp_results.energy:.8f} Hartree ({warp_results.energy * 627.509474:.6f} kcal/mol)"
    )

    # Run torch-dftd (2-body only)
    click.echo("Running torch-dftd (2-body only)...")
    energy_2body, cn_2body, force_2body, c6_2body = run_torch_dftd(
        structure,
        edge_src,
        edge_dst,
        functional_params,
        config.cutoff,
        config.cnthr,
        torch_device,
        abc=False,
    )
    click.echo(
        f"  torch-dftd energy (2-body): {energy_2body:.8f} Hartree ({energy_2body * 627.509474:.6f} kcal/mol)"
    )

    torch_2body_results = TorchDFTDResults(
        coord_num=cn_2body,
        energy_2body=energy_2body,
        energy_3body=None,
        force=force_2body,
        c6_values=c6_2body,
    )

    # Run torch-dftd (2-body + 3-body)
    click.echo("Running torch-dftd (2-body + 3-body)...")
    energy_3body, cn_3body, force_3body, c6_3body = run_torch_dftd(
        structure,
        edge_src,
        edge_dst,
        functional_params,
        config.cutoff,
        config.cnthr,
        torch_device,
        abc=True,
    )
    click.echo(
        f"  torch-dftd energy (2+3-body): {energy_3body:.8f} Hartree ({energy_3body * 627.509474:.6f} kcal/mol)"
    )
    click.echo(
        f"  3-body contribution: {energy_3body - energy_2body:.8f} Hartree ({(energy_3body - energy_2body) * 627.509474:.6f} kcal/mol)"
    )

    torch_3body_results = TorchDFTDResults(
        coord_num=cn_3body,
        energy_2body=energy_2body,
        energy_3body=energy_3body,
        force=force_3body,
        c6_values=c6_3body,
    )

    # Compare results
    click.echo("Comparing results...")
    comparison = compare_results(warp_results, torch_2body_results, torch_3body_results)
    click.echo(f"  CN mean absolute error: {comparison.cn_abs_error:.6f}")
    click.echo(
        f"  Energy absolute error (2-body): {comparison.energy_abs_error_2body:.8f} Hartree"
    )
    click.echo(
        f"  Energy relative error (2-body): {comparison.energy_rel_error_2body:.6%}"
    )
    click.echo(
        f"  Force mean absolute error: {comparison.force_abs_error:.8f} Hartree/Bohr"
    )
    click.echo(f"  Force maximum error: {comparison.force_max_error:.8f} Hartree/Bohr")

    # Generate report
    if output_format == "markdown":
        report = format_markdown_report(
            config,
            structure,
            functional_params,
            warp_results,
            torch_2body_results,
            torch_3body_results,
            comparison,
        )
        click.echo("\n" + report)
    else:
        report = format_json_report(
            config,
            structure,
            functional_params,
            warp_results,
            torch_2body_results,
            torch_3body_results,
            comparison,
        )
        click.echo(report)
    click.echo("\nValidation complete!")


if __name__ == "__main__":
    main()
