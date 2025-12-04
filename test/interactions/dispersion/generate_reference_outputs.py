# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Temporary script to generate reference outputs for DFT-D3 kernel tests.

Run this once to generate outputs, then copy-paste them into regression tests.
After creating the tests, this script can be deleted.
"""

from __future__ import annotations

import numpy as np
import torch
import warp as wp

from nvalchemiops.interactions.dispersion.dftd3 import (
    _c6_energy_kernel,
    _cn_clamp_kernel,
    _cn_force_kernel,
    _geom_cn_kernel,
)
from test.interactions.dispersion.conftest import (
    allocate_outputs,
    from_warp,
    get_element_tables,
    get_functional_params,
    get_hcl_dimer_system,
    get_ne2_system,
    prepare_inputs,
)


def run_full_pipeline(
    system: dict, element_tables: dict, functional_params: dict, device: str = "cpu"
):
    """
    Run full 3-pass DFT-D3 pipeline and return all outputs.

    Parameters
    ----------
    system : dict
        System geometry
    element_tables : dict
        Element parameter tables
    functional_params : dict
        Functional parameters
    device : str
        Device to run on

    Returns
    -------
    dict
        All outputs from the pipeline
    """
    B, M = system["B"], system["M"]

    # Prepare inputs
    inputs = prepare_inputs(system, element_tables, device)
    outputs = allocate_outputs(B, M, device)

    # Create batch indices (all atoms in system 0)
    batch_indices = torch.zeros(B, dtype=torch.int32)
    from test.interactions.dispersion.conftest import to_warp

    batch_wp = to_warp(batch_indices, wp.int32, device)

    # Extract functional params
    k1 = functional_params["k1"]
    k3 = functional_params["k3"]
    exp_threshold = functional_params.get(
        "exp_threshold", -12.0
    )  # Default: conservative
    a1 = functional_params["a1"]
    a2 = functional_params["a2"]
    s6 = functional_params["s6"]
    s8 = functional_params["s8"]

    # S5 switching disabled for reference
    s5_on = 0.0
    s5_off = float("inf")

    z_max_inc = element_tables["z_max_inc"]

    # PASS 1: CN computation
    wp.launch(
        kernel=_geom_cn_kernel,
        dim=(B, M),
        inputs=[
            inputs["coord"],
            inputs["numbers"],
            inputs["nbmat"],
            inputs["rcov"],
            inputs["cnmax"],
            k1,
            exp_threshold,
            B,
            M,
        ],
        outputs=[
            outputs["cn"],
            outputs["inv_r"],
            outputs["df_dr"],
        ],
        device=device,
    )

    # PASS 1.5: Clamp CN
    wp.launch(
        kernel=_cn_clamp_kernel,
        dim=B,
        inputs=[
            inputs["numbers"],
            inputs["cnmax"],
            B,
            outputs["cn"],
        ],
        device=device,
    )

    # PASS 2: Energy and forces
    wp.launch(
        kernel=_c6_energy_kernel,
        dim=(B, M),
        inputs=[
            inputs["coord"],
            inputs["numbers"],
            inputs["nbmat"],
            outputs["cn"],
            outputs["inv_r"],
            inputs["r4r2"],
            inputs["c6ref"],
            inputs["cnref_i"],
            inputs["cnref_j"],
            z_max_inc,
            k3,
            exp_threshold,
            a1,
            a2,
            s6,
            s8,
            s5_on,
            s5_off,
            B,
            M,
            batch_wp,
        ],
        outputs=[
            outputs["energy_per_atom"],
            outputs["total_energy"],
            outputs["dE_dCN"],
            outputs["force"],
        ],
        device=device,
    )

    # PASS 3: CN chain forces
    wp.launch(
        kernel=_cn_force_kernel,
        dim=(B, M),
        inputs=[
            inputs["coord"],
            inputs["nbmat"],
            outputs["dE_dCN"],
            outputs["df_dr"],
            outputs["inv_r"],
            B,
            M,
            outputs["force"],
        ],
        device=device,
    )

    # Convert to numpy
    return {
        "cn": from_warp(outputs["cn"]),
        "inv_r": from_warp(outputs["inv_r"]),
        "df_dr": from_warp(outputs["df_dr"]),
        "energy_per_atom": from_warp(outputs["energy_per_atom"]),
        "total_energy": from_warp(outputs["total_energy"]),
        "dE_dCN": from_warp(outputs["dE_dCN"]),
        "force": from_warp(outputs["force"]),
    }


def format_array(arr: torch.Tensor, name: str) -> str:
    """Format PyTorch tensor as Python code for copy-paste."""
    if arr.numel() == 1:
        return f'    "{name}": torch.tensor([{arr.item():.10e}], dtype=torch.float32),'
    elif arr.numel() <= 10:
        values = ", ".join(f"{x:.10e}" for x in arr.flatten())
        return f'    "{name}": torch.tensor([{values}], dtype=torch.float32),'
    else:
        # For large arrays, format as multi-line
        lines = [f'    "{name}": torch.tensor([']
        for i in range(0, arr.numel(), 4):
            chunk = arr.flatten()[i : i + 4]
            values = ", ".join(f"{x:.10e}" for x in chunk)
            lines.append(f"        {values},")
        lines.append("    ], dtype=torch.float32),")
        return "\n".join(lines)


def main():
    """Generate and print reference outputs."""
    print("=" * 80)
    print("GENERATING REFERENCE OUTPUTS FOR DFT-D3 KERNEL TESTS")
    print("=" * 80)
    print()

    # Get parameters (Zmax=17 to include Cl)
    element_tables = get_element_tables(z_max=17)
    functional_params = get_functional_params()

    output_keys = [
        "cn",
        "inv_r",
        "df_dr",
        "energy_per_atom",
        "total_energy",
        "dE_dCN",
        "force",
    ]

    # =========================================================================
    # Test Ne2 system (noble gas dispersion)
    # =========================================================================
    print("Ne2 SYSTEM (separation=5.8 Bohr) on CPU:")
    print("-" * 80)
    print("Noble gas dimer - pure dispersion interaction")
    print()
    ne2_system = get_ne2_system(separation=5.8)
    ne2_cpu = run_full_pipeline(
        ne2_system, element_tables, functional_params, device="cpu"
    )

    print("Copy-paste this into test_dftd3_kernels.py:")
    print()
    print("NE2_REFERENCE_CPU = {")
    for key in output_keys:
        print(format_array(ne2_cpu[key], key))
    print("}")
    print()
    print(f"Ne2 total energy: {ne2_cpu['total_energy'][0]:.6e} Hartree")
    print()

    # =========================================================================
    # Test HCl dimer system (realistic molecular dispersion)
    # =========================================================================
    print()
    print("HCl DIMER SYSTEM (parallel, sep=7 Bohr) on CPU:")
    print("-" * 80)
    print("Realistic molecular dimer - heteronuclear dispersion")
    print()
    hcl_system = get_hcl_dimer_system()
    hcl_cpu = run_full_pipeline(
        hcl_system, element_tables, functional_params, device="cpu"
    )

    print("Copy-paste this into test_dftd3_kernels.py:")
    print()
    print("HCL_DIMER_REFERENCE_CPU = {")
    for key in output_keys:
        print(format_array(hcl_cpu[key], key))
    print("}")
    print()
    print(f"HCl dimer total energy: {hcl_cpu['total_energy'][0]:.6e} Hartree")
    print()

    # =========================================================================
    # GPU tests if available
    # =========================================================================
    if wp.is_cuda_available():
        print()
        print("=" * 80)
        print("GPU VERIFICATION")
        print("=" * 80)
        print()

        # Ne2 on GPU
        print("Ne2 on GPU:")
        print("-" * 80)
        ne2_gpu = run_full_pipeline(
            ne2_system, element_tables, functional_params, device="cuda:0"
        )
        print("CPU/GPU Differences:")
        for key in ne2_cpu.keys():
            diff = np.abs(ne2_cpu[key] - ne2_gpu[key]).max()
            print(f"  {key:20s}: max diff = {diff:.2e}")
        print()

        # HCl dimer on GPU
        print("HCl dimer on GPU:")
        print("-" * 80)
        hcl_gpu = run_full_pipeline(
            hcl_system, element_tables, functional_params, device="cuda:0"
        )
        print("CPU/GPU Differences:")
        for key in hcl_cpu.keys():
            diff = np.abs(hcl_cpu[key] - hcl_gpu[key]).max()
            print(f"  {key:20s}: max diff = {diff:.2e}")
    else:
        print()
        print("GPU not available, skipping GPU reference generation.")

    print()
    print("=" * 80)
    print("DONE! Copy the reference data above into your test file.")
    print("=" * 80)


if __name__ == "__main__":
    main()
