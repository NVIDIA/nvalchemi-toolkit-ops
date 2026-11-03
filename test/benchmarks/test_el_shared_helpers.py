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

"""EL options preserve the shared benchmark builders and planning defaults."""

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmarks.config import load_yaml_config
from benchmarks.interactions.dispersion.benchmark_dftd3 import (
    dry_run_from_config as d3_dry_run,
)
from benchmarks.neighborlist.benchmark_neighborlist import (
    dry_run_from_config as nl_dry_run,
)
from benchmarks.suite_orchestration import _benchmark_case_key
from benchmarks.suite_systems import (
    CSCL_LATTICE_CONSTANT,
    benchmark_system_metadata,
    compute_atomic_density,
    configs_for_mode,
    create_system,
    planned_atom_counts,
)
from benchmarks.suite_utils import (
    make_row_meta,
    measure_timing_batches,
    neighbor_count_metadata,
)


class TestDefaultCsClGeometry:
    """Default CsCl builders retain the upstream cubic replication behavior."""

    @pytest.mark.parametrize("requested,cells", [(2, 1), (128, 4), (250, 5), (1024, 8)])
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_cubic_replication(self, requested, cells, batch_size):
        """Positions, cell and counts match the complete upstream cubic lattice."""
        system = create_system(
            "cscl", requested, batch_size=batch_size, device="cpu", backend="torch"
        )
        expected_positions = []
        for x in range(cells):
            for y in range(cells):
                for z in range(cells):
                    origin = (
                        np.array([x, y, z], dtype=np.float32) * CSCL_LATTICE_CONSTANT
                    )
                    expected_positions.extend(
                        [
                            origin,
                            origin
                            + np.array([0.5, 0.5, 0.5], dtype=np.float32)
                            * CSCL_LATTICE_CONSTANT,
                        ]
                    )
        expected_positions = np.tile(
            np.asarray(expected_positions, dtype=np.float32), (batch_size, 1)
        )
        expected_cell = np.eye(3, dtype=np.float32) * (cells * CSCL_LATTICE_CONSTANT)
        np.testing.assert_array_equal(system["positions"].numpy(), expected_positions)
        np.testing.assert_array_equal(
            system["cell"].numpy(), np.tile(expected_cell[None], (batch_size, 1, 1))
        )
        assert system["atoms_per_system"] == 2 * cells**3
        assert system["total_atoms"] == 2 * cells**3 * batch_size
        torch.testing.assert_close(
            system["atomic_numbers"],
            torch.tensor([55, 17] * (cells**3 * batch_size), dtype=torch.int32),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            system["charges"],
            torch.tensor([1.0, -1.0] * (cells**3 * batch_size), dtype=torch.float64),
            rtol=0,
            atol=0,
        )


class TestSharedBenchmarkDefaults:
    """EL's optional controls preserve the existing NL/D3 plan dimensions."""

    @pytest.mark.parametrize(
        "runner,config_path,backend,expected_rows,rows_per_case",
        [
            (nl_dry_run, "neighborlist/benchmark_config.yaml", "torch", 1095, 15),
            (nl_dry_run, "neighborlist/benchmark_config.yaml", "jax", 876, 12),
            (
                d3_dry_run,
                "interactions/dispersion/benchmark_config.yaml",
                "torch",
                146,
                2,
            ),
            (
                d3_dry_run,
                "interactions/dispersion/benchmark_config.yaml",
                "jax",
                146,
                2,
            ),
        ],
    )
    def test_existing_plan_rows(
        self, runner, config_path, backend, expected_rows, rows_per_case
    ):
        """Published NL/D3 configurations retain upstream row keys and counts."""
        root = Path(__file__).resolve().parents[2]
        config = load_yaml_config(root / "benchmarks" / config_path)
        config["runtime"] = {"plan_output": "count"}
        rows = runner(config, backend=backend)

        assert len(rows) == expected_rows
        assert len({_benchmark_case_key(row) for row in rows}) == expected_rows
        assert Counter((row["system"], row["mode"]) for row in rows) == {
            ("cscl", "system_size"): 11 * rows_per_case,
            ("cscl", "constant_workload"): 10 * rows_per_case,
            ("cscl", "batch_scaling"): 15 * rows_per_case,
            ("nh3", "system_size"): 11 * rows_per_case,
            ("nh3", "constant_workload"): 11 * rows_per_case,
            ("nh3", "batch_scaling"): 15 * rows_per_case,
        }
        expected_keys = {
            "benchmark",
            "backend",
            "system",
            "mode",
            "atoms_per_system",
            "batch_size",
            "total_atoms",
            "method",
            "cutoff",
            "reason",
        }
        assert all(set(row) == expected_keys for row in rows)
        cscl_sizes = list(
            dict.fromkeys(
                row["atoms_per_system"]
                for row in rows
                if row["system"] == "cscl" and row["mode"] == "system_size"
            )
        )
        assert cscl_sizes == [
            128,
            432,
            686,
            1024,
            2662,
            4394,
            8192,
            18522,
            35152,
            65536,
            137842,
        ]
        for row in rows:
            assert row["total_atoms"] == row["atoms_per_system"] * row["batch_size"]
            if row["mode"] == "constant_workload":
                assert row["batch_size"] == 131072 // row["atoms_per_system"]

    @pytest.mark.parametrize(
        "mode", ["system_size", "constant_workload", "batch_scaling"]
    )
    def test_default_row_metadata(self, mode):
        """Existing callers receive the same six identity columns."""
        assert make_row_meta("cscl", mode, "torch", 128, 2, 256) == {
            "system": "cscl",
            "scaling_mode": mode,
            "backend": "torch",
            "atoms_per_system": 128,
            "batch_size": 2,
            "total_atoms": 256,
        }


class TestExplicitElGeometry:
    """Caller-supplied replication and batch counts select exact EL workloads."""

    @pytest.mark.parametrize(
        "requested,factors", [(128, (4, 4, 4)), (256, (4, 4, 8)), (512, (4, 8, 8))]
    )
    @pytest.mark.parametrize("batch_size", [1, 3])
    def test_exact_replication_counts(self, requested, factors, batch_size):
        """Optional EL factors produce exact requested counts in every replica."""
        system = create_system(
            "cscl",
            requested,
            batch_size=batch_size,
            device="cpu",
            backend="torch",
            cscl_replication_factors=factors,
        )
        assert system["atoms_per_system"] == requested
        assert system["total_atoms"] == requested * batch_size
        assert system["positions"].shape == (requested * batch_size, 3)
        expected_cell = np.diag(
            np.asarray(factors, dtype=np.float32) * CSCL_LATTICE_CONSTANT
        )
        np.testing.assert_array_equal(
            system["cell"].numpy(), np.tile(expected_cell[None], (batch_size, 1, 1))
        )
        metadata = benchmark_system_metadata(system, "cscl")
        assert tuple(metadata[f"cscl_replication_n{axis}"] for axis in "xyz") == factors
        assert metadata["cscl_cell_shape"] == (
            "cubic" if len(set(factors)) == 1 else "orthorhombic"
        )
        assert metadata["max_abs_net_charge_per_system"] == 0.0

    @pytest.mark.parametrize("system_name", ["cscl", "nh3"])
    def test_explicit_fixed_total_plan(self, system_name):
        """Explicit EL batches keep exactly 131072 atoms in every planned row."""
        batch_sizes = {128: 1024, 256: 512, 512: 256}
        mode_config = {
            "target_atoms": 131072,
            "batch_sizes": {system_name: batch_sizes},
        }
        system_config = {"atom_counts": list(batch_sizes)}
        if system_name == "cscl":
            system_config["replication_factors"] = {
                128: [4, 4, 4],
                256: [4, 4, 8],
                512: [4, 8, 8],
            }
        configs = configs_for_mode(
            "constant_workload",
            mode_config,
            system_name,
            system_config,
            plan_only=True,
        )
        assert len(configs) == len(batch_sizes)
        for cfg in configs:
            count, batch_size, total = planned_atom_counts(system_name, cfg)
            assert count == cfg["num_atoms"]
            assert batch_size == batch_sizes[count]
            assert total == 131072
            metadata = make_row_meta(
                system_name,
                "constant_workload",
                "torch",
                count,
                batch_size,
                total,
                mode_config=mode_config,
            )
            assert metadata["target_total_atoms"] == total
            assert metadata["target_delta_atoms"] == 0
            assert metadata["target_delta_fraction"] == 0.0
            assert metadata["batch_selection"] == "explicit"


class TestElInputMetadata:
    """Row diagnostics reject invalid topology counts and physical charges."""

    def test_observed_capacity_counts(self):
        """Neighbor metadata separates allocated width from observed counts."""
        assert neighbor_count_metadata(np.array([2, 0, 3], dtype=np.int32), 8) == {
            "configured_max_neighbors": 8,
            "observed_max_neighbors": 3,
            "observed_directed_neighbor_pairs": 5,
        }

    @pytest.mark.parametrize(
        "counts,capacity,message",
        [
            (np.array([1], dtype=np.int32), 0, "must be positive"),
            (np.array([1], dtype=np.int32), -1, "must be positive"),
            (np.array([3], dtype=np.int32), 2, "exceeds configured capacity"),
            (np.array([-1], dtype=np.int32), 2, "negative counts"),
            (np.array([1.0]), 2, "integer dtype"),
            (np.array([[1]], dtype=np.int32), 2, "one-dimensional"),
        ],
    )
    def test_invalid_neighbor_counts(self, counts, capacity, message):
        """Invalid capacities or count arrays fail before results are recorded."""
        with pytest.raises(ValueError, match=message):
            neighbor_count_metadata(counts, capacity)

    @pytest.mark.parametrize(
        "charges,message",
        [
            (torch.tensor([0.5, -0.5], dtype=torch.float64), "configured species"),
            (torch.tensor([float("nan"), -1.0], dtype=torch.float64), "must be finite"),
            (torch.tensor([1.0], dtype=torch.float64), "one value per atom"),
            (torch.tensor([1, -1], dtype=torch.int32), "floating dtype"),
        ],
    )
    def test_invalid_species_charges(self, charges, message):
        """Finite, shaped floating charges must match the declared species."""
        system = create_system("cscl", 2, device="cpu", backend="torch")
        system["charges"] = charges
        with pytest.raises(ValueError, match=message):
            benchmark_system_metadata(system, "cscl")


class TestAtomicDensity:
    """Density setup supports cached values and actual per-system geometry."""

    @pytest.mark.parametrize("cached_density", [0.5, 2.0])
    def test_cached_density(self, cached_density):
        """A validated cached density needs no cell or atom-count inputs."""
        assert (
            compute_atomic_density({"atomic_density": cached_density}) == cached_density
        )

    @pytest.mark.parametrize("cached_density", [0.0, -1.0, np.nan, np.inf])
    def test_invalid_cached_density(self, cached_density):
        """Invalid cached values fail before geometry fallback."""
        with pytest.raises(ValueError, match="positive and finite"):
            compute_atomic_density({"atomic_density": cached_density})

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize(
        "count_source", ["per_system", "batch_idx", "total_atoms", "positions"]
    )
    def test_geometry_density(self, backend, count_source):
        """The denser member controls density for unequal triclinic cells."""
        cell = np.array([[2.0, 0.0, 0.0], [0.5, 2.0, 0.0], [0.25, 0.5, 2.0]])
        cell = np.stack([cell, cell * 2.0])
        system = {"cell": torch.from_numpy(cell) if backend == "torch" else cell}
        if count_source == "per_system":
            system["atoms_per_system"] = 8
        elif count_source == "batch_idx":
            batch_idx = np.repeat(np.arange(2), [8, 16])
            system["batch_idx"] = (
                torch.from_numpy(batch_idx) if backend == "torch" else batch_idx
            )
        elif count_source == "total_atoms":
            system["total_atoms"] = 16
        else:
            system["positions"] = np.zeros((16, 3))
        assert compute_atomic_density(system) == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "system,message",
        [
            ({"cell": np.eye(2), "total_atoms": 8}, "cell must have shape"),
            ({"cell": np.zeros((3, 3)), "total_atoms": 8}, "volumes must be positive"),
            ({"cell": np.stack([np.eye(3)] * 2), "total_atoms": 3}, "cannot divide"),
            ({"cell": np.eye(3), "atoms_per_system": 0}, "counts must be positive"),
        ],
    )
    def test_invalid_geometry(self, system, message):
        """Shared geometry validation rejects invalid volumes and atom counts."""
        with pytest.raises(ValueError, match=message):
            compute_atomic_density(system)


class TestJaxTimingGroups:
    """The timer argument exposes scheduling checks without GPU execution."""

    @pytest.mark.parametrize("queued_warmup_batches", [0, 2])
    def test_shared_warmup(self, queued_warmup_batches):
        """Five measured groups receive warmup 3 once, followed by None."""
        calls = []

        def prepared_call():
            """Represent the prepared public benchmark workload."""
            return "prepared"

        def batch_timer(fn, num_runs, *, warmup_runs):
            """Record the injected timer's normal arguments and result."""
            assert fn() == "prepared"
            calls.append((num_runs, warmup_runs))
            return float(len(calls))

        timings = measure_timing_batches(
            prepared_call,
            num_runs=10,
            warmup_runs=3,
            timing_batches=5,
            backend="jax",
            queued_warmup_batches=queued_warmup_batches,
            jax_batch_timer=batch_timer,
        )
        assert calls == [(10, 3)] + [(10, None)] * (4 + queued_warmup_batches)
        assert timings == tuple(
            float(index)
            for index in range(queued_warmup_batches + 1, queued_warmup_batches + 6)
        )
