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

"""Configuration and complete-case coverage for the EL benchmark runner."""

from collections import Counter
from pathlib import Path

import pytest

from benchmarks.config import load_yaml_config
from benchmarks.interactions.electrostatics.benchmark_electrostatics_suite import (
    _configured_pme_cutoff,
    _prepare_el_families,
    dry_run_from_config,
)
from benchmarks.suite_orchestration import _benchmark_case_key
from benchmarks.suite_systems import create_system


class TestElBenchmarkConfiguration:
    """Scientific setup settings come from the versioned benchmark config."""

    @pytest.mark.parametrize(
        "mode", ["system_size", "constant_workload", "batch_scaling"]
    )
    @pytest.mark.parametrize("plan_only", [False, True])
    def test_nh3_execution_retains_physical_input(self, tmp_path, mode, plan_only):
        """Execution resolves actual PDB files and materializes the planned batch."""
        source = Path(__file__).resolve().parents[2] / "benchmarks/nh3/ammonia.pdb"
        pdb_path = tmp_path / "ammonia_pbc_4.pdb"
        pdb_path.write_text(
            "CRYST1   10.000   10.000   10.000  90.00  90.00  90.00 P 1\n"
            + source.read_text()
        )
        config = {
            "systems": {
                "nh3": {
                    "pdb_dir": str(tmp_path),
                    "atom_counts": [4],
                    "constant_atoms_sizes": [4],
                }
            },
            "scaling": {
                "system_size": {"batch_size": 1},
                "constant_workload": {
                    "target_atoms": 16,
                    "batch_sizes": {"nh3": {4: 4}},
                },
                "batch_scaling": {"max_total_atoms": 16},
            },
        }
        cases = _prepare_el_families(config, plan_only=plan_only)[("nh3", mode)]
        assert cases
        for case in cases:
            if plan_only:
                assert case["pdb_path"] is None
            else:
                assert case["pdb_path"] == pdb_path
                actual = create_system(
                    "nh3",
                    case["num_atoms"],
                    pdb_path=case["pdb_path"],
                    batch_size=case["batch_size"],
                    device="cpu",
                )
                assert actual["positions"].shape == (4 * case["batch_size"], 3)

    @pytest.mark.parametrize("cutoff", [5.0, 9.0, 15.0])
    def test_configured_pme_cutoff(self, cutoff):
        """The benchmark reads its explicit real-space cutoff methodology."""
        assert _configured_pme_cutoff({"max_real_space_cutoff": cutoff}) == cutoff

    @pytest.mark.parametrize("cutoff", [0.0, -1.0, float("nan"), float("inf")])
    def test_invalid_pme_cutoff(self, cutoff):
        """Invalid cutoff limits fail before benchmark setup."""
        with pytest.raises(ValueError, match="positive and finite"):
            _configured_pme_cutoff({"max_real_space_cutoff": cutoff})

    def test_configuration_requires_explicit_pme_cutoff(self):
        """The cutoff comes from configuration without a duplicated fallback."""
        with pytest.raises(KeyError, match="max_real_space_cutoff"):
            _configured_pme_cutoff({})

    def test_complete_reportable_plan(self):
        """The plan includes every method, backend, system, and scaling mode."""
        root = Path(__file__).resolve().parents[2]
        config = load_yaml_config(
            root / "benchmarks/interactions/electrostatics/benchmark_config.yaml"
        )
        config["runtime"] = {"plan_output": "count"}
        rows = [
            row
            for backend in ("torch", "jax")
            for row in dry_run_from_config(config, backend=backend)
        ]
        assert len(rows) == 592
        assert all(
            count == 1 for count in Counter(map(_benchmark_case_key, rows)).values()
        )
        assert Counter((row["backend"], row["method"]) for row in rows) == {
            (backend, method): 148
            for backend in ("torch", "jax")
            for method in ("pme", "ewald")
        }
        assert Counter(row["system"] for row in rows) == {"cscl": 296, "nh3": 296}
        assert {row["workload"] for row in rows} == {"energy_forces_charge_gradients"}
        assert all(
            row["compute_forces"] and row["compute_charge_gradients"] for row in rows
        )
        assert all(
            row["total_atoms"] == row["atoms_per_system"] * row["batch_size"]
            for row in rows
        )
        assert all(
            row["total_atoms"] == 131072
            for row in rows
            if row["mode"] == "constant_workload"
        )
        assert config["parameters"]["max_real_space_cutoff"] == 9.0
        pme = next(method for method in config["methods"] if method["name"] == "pme")
        assert pme["spline_order"] == 5
        cscl_sizes = {
            row["atoms_per_system"]
            for row in rows
            if row["system"] == "cscl" and row["mode"] == "system_size"
        }
        assert cscl_sizes == set(config["systems"]["cscl"]["atom_counts"])
