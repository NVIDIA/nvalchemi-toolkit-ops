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

"""Tests for benchmark planning and result-schema helpers."""

import csv
import inspect
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from benchmarks import benchmark_suite
from benchmarks.benchmark_suite import (
    validate_backend_selection,
    validate_method_selection,
)
from benchmarks.config import (
    enabled_method_names,
    load_yaml_config,
    merge_common_cli_overrides,
    normalize_method_name,
)
from benchmarks.interactions.dispersion import benchmark_dftd3
from benchmarks.interactions.dispersion.benchmark_dftd3 import (
    dry_run_from_config as dry_run_d3,
)
from benchmarks.interactions.electrostatics import (
    benchmark_electrostatics_suite as benchmark_electrostatics,
)
from benchmarks.interactions.electrostatics.benchmark_electrostatics_suite import (
    _el_unpack_params,
    _torch_neighbor_matrix_to_list_chunked,
)
from benchmarks.neighborlist import benchmark_neighborlist
from benchmarks.neighborlist.benchmark_neighborlist import (
    _nl_method_for_case,
)
from benchmarks.neighborlist.benchmark_neighborlist import (
    dry_run_from_config as dry_run_nl,
)
from benchmarks.plotting import plot_benchmarks
from benchmarks.plotting import styles as plot_styles
from benchmarks.plotting.plot_benchmarks import load_csv
from benchmarks.suite_orchestration import _report_run, _SuiteResults
from benchmarks.suite_systems import (
    configs_for_mode,
    configured_nh3_artifacts,
    filter_configs_by_total_atoms,
    planned_atom_counts,
)
from benchmarks.suite_utils import (
    build_failure_result,
    build_result,
    build_skipped_result,
    clean_jax,
    configure_jax_environment,
    failure_error_type,
    jax_timed_batch,
    jax_timed_stateful,
    measure_memory_jax,
    save_results,
    write_run_log,
)
from docs.benchmarks import generate_plots as docs_generate_plots
from docs.benchmarks import sphinxext as docs_benchmark_sphinxext


class TestBenchmarkMethodSelection:
    """Test benchmark method normalization and CLI selection."""

    @pytest.mark.parametrize(
        ("token", "expected"),
        [
            ("cell", "cell_list"),
            ("naive", "naive_neighbor_list"),
            ("naive-scalar", "naive_scalar"),
            ("naive-tile", "naive_tile"),
            ("cell-list-atom-centric", "cell_list_atom_centric"),
            ("cell-list-pair-centric", "cell_list_pair_centric"),
            ("batch-cell-list", "batch_cell_list"),
            ("batch-naive-scalar", "batch_naive_scalar"),
            ("batch-naive-tile", "batch_naive_tile"),
            (
                "batch-cell-list-atom-centric",
                "batch_cell_list_atom_centric",
            ),
            (
                "batch-cell-list-pair-centric",
                "batch_cell_list_pair_centric",
            ),
            ("batch_naive", "batch_naive_neighbor_list"),
            ("cluster-tile", "cluster_tile"),
            ("tile", "cluster_tile"),
            ("batch-cluster-tile", "batch_cluster_tile"),
            ("d3", "dftd3"),
            ("pme", "pme"),
        ],
    )
    def test_normalize_method_aliases(self, token, expected):
        """Legacy aliases normalize to the public API method names."""
        assert normalize_method_name(token) == expected

    def test_selected_methods_preserve_explicit_batch_api(self):
        """Explicit CLI methods are returned even when absent from YAML."""
        config = {
            "parameters": {},
            "runtime": {},
            "systems": {"cscl": {"enabled": True}},
            "scaling": {"system_size": {"enabled": True}},
            "methods": [{"name": "cell_list", "enabled": True}],
        }
        args = type(
            "Args",
            (),
            {
                "timing_runs": None,
                "warmup_runs": None,
                "system": None,
                "mode": None,
                "output_dir": None,
                "backend": None,
                "methods": ["cell_list", "batch_cell_list"],
                "dry_run": False,
                "max_total_atoms": None,
            },
        )()

        merged = merge_common_cli_overrides(config, args)

        assert enabled_method_names(merged) == ["cell_list", "batch_cell_list"]
        assert merged["runtime"]["explicit_methods"] is True

    def test_jax_benchmarks_do_not_jit_zero_arg_array_closures(self):
        """JAX benchmark wrappers pass large arrays as jit arguments."""
        functions = (
            benchmark_electrostatics._benchmark_pme_jax,
            benchmark_electrostatics._benchmark_ewald_jax,
            benchmark_neighborlist._benchmark_nl_jax,
            benchmark_dftd3._benchmark_d3_jax,
        )

        for function in functions:
            assert "jax.jit(run_" not in inspect.getsource(function)

    def test_jax_el_kernel_cache_is_defined(self):
        """EL keeps reusable jitted entrypoints at module scope."""
        assert isinstance(benchmark_electrostatics._JAX_EL_KERNEL_CACHE, dict)

    def test_jax_cleanup_preserves_executable_caches_by_default(self):
        """Successful JAX rows do not clear compiled executables per cutoff."""
        signature = inspect.signature(clean_jax)
        source = inspect.getsource(clean_jax)

        assert signature.parameters["clear_executables"].default is False
        assert "if clear_executables:" in source
        assert "jax.clear_caches()" in source

    def test_jax_nl_timing_falls_back_to_serial_after_oom(self):
        """Large JAX NL rows can time serially when batched dispatch OOMs."""
        fallback_source = inspect.getsource(
            benchmark_neighborlist._jax_nl_timed_with_serial_fallback
        )
        benchmark_source = inspect.getsource(benchmark_neighborlist._benchmark_nl_jax)

        assert 'failure_error_type(e) != "OutOfMemoryError"' in fallback_source
        assert "if prefer_serial:" in fallback_source
        assert "clean_jax(clear_executables=True)" in fallback_source
        assert "jax_timed_serial" in fallback_source
        assert '"jax_wall_block_each"' in fallback_source
        assert "_jax_nl_timed_with_serial_fallback(" in benchmark_source
        assert "prefer_serial=num_runs > 1" not in benchmark_source
        assert "prefer_serial=prefer_serial" in benchmark_source
        assert '"timing_method": timing_method' in benchmark_source

    @pytest.mark.parametrize(
        ("total_atoms", "max_neighbors", "num_runs", "memory_limit", "expected"),
        [
            (128, 64, 10, 80 * 1024**3, False),
            (131072, 9830, 10, 80 * 1024**3, True),
            (131072, 9830, 1, 80 * 1024**3, False),
            (131072, 9830, 10, 0, False),
        ],
    )
    def test_jax_nl_serial_timing_uses_output_capacity(
        self,
        total_atoms,
        max_neighbors,
        num_runs,
        memory_limit,
        expected,
    ):
        """Queued-output timing switches modes from shapes and device capacity."""
        assert (
            benchmark_neighborlist._jax_nl_queued_outputs_exceed_device_memory(
                total_atoms,
                max_neighbors,
                num_runs,
                memory_limit,
            )
            is expected
        )

    def test_warp_nl_timing_declares_warp_backend(self):
        """Direct Warp API timings use the shared CUDA-event path explicitly."""
        source = inspect.getsource(benchmark_neighborlist._benchmark_nl_warp)

        assert 'backend="warp"' in source

    def test_warp_nl_reports_preallocated_buffer_boundary(self):
        """Warp rows identify caller-owned output and cell-workspace buffers."""
        source = inspect.getsource(benchmark_neighborlist._benchmark_nl_warp)

        assert '"configured_max_neighbors": int(maxnb)' in source
        assert '"caller_preallocated_outputs"' in source
        assert '"caller_preallocated_outputs_and_cell_workspace"' in source

    def test_warp_naive_uses_root_launcher_strategy_keyword(self):
        """Warp naive variants use the public root-launcher strategy keyword."""
        source = inspect.getsource(benchmark_neighborlist._benchmark_nl_warp)

        assert "native_strategy=native_strategy" not in source
        assert source.count("strategy=native_strategy") == 2

    def test_suite_sets_jax_allocator_env_before_runner_imports(self):
        """Suite-level JAX env includes the throughput allocator policy."""
        source = inspect.getsource(benchmark_suite.main)
        env_block = source.split("if args.plot_only:", 1)[0]

        assert "configure_jax_environment(need_x64=True" in env_block
        helper_source = inspect.getsource(configure_jax_environment)
        assert 'os.environ.setdefault("JAX_ENABLE_X64", "1")' in helper_source
        assert (
            'os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")'
            in helper_source
        )
        assert "XLA_PYTHON_CLIENT_PREALLOCATE" in helper_source
        assert "on-demand allocation" in helper_source
        assert "TF_GPU_ALLOCATOR" not in helper_source

    def test_suite_rejects_profile_selector(self, monkeypatch, capsys):
        """The reportable suite has no reduced profile selector."""
        monkeypatch.setattr(
            sys,
            "argv",
            ["benchmark_suite.py", "--benchmark", "all", "--profile"],
        )

        with pytest.raises(SystemExit) as exc_info:
            benchmark_suite.parse_args()

        captured = capsys.readouterr()
        assert exc_info.value.code == 2
        assert "unrecognized arguments: --profile" in captured.err

    def test_jax_env_warns_when_preallocation_policy_is_unset(
        self, monkeypatch, capsys
    ):
        """DallasF allocator concern is visible without disabling preallocation."""
        monkeypatch.delenv("JAX_ENABLE_X64", raising=False)
        monkeypatch.delenv("XLA_PYTHON_CLIENT_MEM_FRACTION", raising=False)
        monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
        monkeypatch.delenv("XLA_PYTHON_CLIENT_ALLOCATOR", raising=False)
        monkeypatch.setattr(
            "benchmarks.suite_utils._JAX_ALLOCATOR_WARNING_EMITTED", False
        )

        configure_jax_environment(need_x64=True, context="test context")

        captured = capsys.readouterr()
        assert "XLA_PYTHON_CLIENT_PREALLOCATE is unset" in captured.out
        assert "XLA_PYTHON_CLIENT_MEM_FRACTION=0.95" in captured.out
        assert "on-demand allocation" in captured.out
        assert "XLA_PYTHON_CLIENT_PREALLOCATE" not in captured.err

    def test_jax_naive_benchmark_uses_supported_timing_boundary(self):
        """JAX naive timing follows each public strategy's supported boundary."""
        benchmark_source = inspect.getsource(benchmark_neighborlist._benchmark_nl_jax)

        assert "use_direct_jax_nl = jax_family in" in benchmark_source
        assert "return jax_neighbor_list(" in benchmark_source
        assert (
            'shift_range_per_dimension=nl_kwargs["shift_range_per_dimension"]'
            in benchmark_source
        )
        assert 'num_shifts_per_system=nl_kwargs["num_shifts_per_system"]' in (
            benchmark_source
        )
        assert 'max_shifts_per_system=nl_kwargs["max_shifts_per_system"]' in (
            benchmark_source
        )
        assert "compiled_nl = jax.jit(direct_nl_kernel)" in benchmark_source
        assert (
            'use_eager_naive_tile = method in {"naive_tile", "batch_naive_tile"}'
            in (benchmark_source)
        )
        assert (
            "time_sec, timing_method = _jax_nl_timed_with_serial_fallback("
            in benchmark_source
        )
        assert '"jit_managed_functional_buffers"' in benchmark_source

    @pytest.mark.gpu
    @pytest.mark.parametrize(
        ("method", "batch_size"),
        [("naive_tile", 1), ("batch_naive_tile", 2)],
    )
    def test_jax_naive_tile_benchmark_runs_eagerly(self, method, batch_size):
        """Explicit tile strategies run through their eager public API."""
        import jax
        import jax.numpy as jnp

        if not any(device.platform == "gpu" for device in jax.local_devices()):
            pytest.skip("JAX CUDA device is required")

        base_positions = (
            jnp.arange(96, dtype=jnp.float32).reshape(32, 3)
            * jnp.array([0.173, 0.337, 0.491], dtype=jnp.float32)
        ) % 7.0
        positions = jnp.concatenate([base_positions] * batch_size, axis=0)
        base_cell = jnp.eye(3, dtype=jnp.float32) * 8.0
        cell = jnp.repeat(base_cell[None, :, :], batch_size, axis=0)
        pbc = jnp.ones((batch_size, 3), dtype=jnp.bool_)
        data = {
            "positions": positions,
            "cell": cell,
            "pbc": pbc,
            "batch_idx": jnp.repeat(
                jnp.arange(batch_size, dtype=jnp.int32),
                32,
            ),
            "atoms_per_system": 32,
            "batch_size": batch_size,
            "total_atoms": 32 * batch_size,
        }

        result = benchmark_neighborlist._benchmark_nl_jax(
            data,
            cutoff=1.0,
            method=method,
            num_runs=1,
            warmup_runs=0,
        )

        assert result["time_seconds"] > 0.0
        assert result["timing_method"] == "jax_wall_block_until_ready"

    def test_jax_cell_list_benchmark_uses_direct_batched_timing(self):
        """JAX cell-list timing follows the current public API signatures."""
        benchmark_source = inspect.getsource(benchmark_neighborlist._benchmark_nl_jax)

        assert "use_direct_jax_nl = jax_family in" in benchmark_source
        assert "return jax_neighbor_list(" in benchmark_source
        assert 'max_total_cells=nl_kwargs["max_total_cells"]' in benchmark_source
        assert 'neighbor_search_radius=nl_kwargs["neighbor_search_radius"]' in (
            benchmark_source
        )
        batch_source = benchmark_source.rsplit(
            'elif jax_family == "batch_cell_list":', 1
        )[1].split('elif jax_family == "naive":', 1)[0]
        assert "neighbor_search_radius=" not in batch_source
        assert "min_cells_per_dimension=" not in benchmark_source
        assert "compiled_nl = jax.jit(direct_nl_kernel)" in benchmark_source
        assert (
            "time_sec, timing_method = _jax_nl_timed_with_serial_fallback("
            in benchmark_source
        )

    def test_jax_pme_benchmark_precomputes_reusable_metadata(self):
        """JAX PME timings pass cacheable mesh/cell metadata explicitly."""
        source = inspect.getsource(benchmark_electrostatics._benchmark_pme_jax)
        kernel_source = inspect.getsource(benchmark_electrostatics._get_jax_el_kernels)

        assert (
            'compute_bspline_moduli_1d = jax_api["compute_bspline_moduli_1d"]' in source
        )
        assert "cell_inv_t = jnp.transpose(jnp.linalg.inv(cell_3d)" in source
        assert "volume = jnp.abs(jnp.linalg.det(cell_3d))" in source
        assert "moduli_x = compute_bspline_moduli_1d" in source
        assert "volume=volume" in kernel_source
        assert "cell_inv_t=cell_inv_t" in kernel_source
        assert "moduli_x=moduli_x" in kernel_source
        assert "jax.value_and_grad(" in kernel_source
        assert "argnums=(0, 1)" in kernel_source

    def test_torch_pme_benchmark_precomputes_reusable_metadata(self):
        """Torch PME timings keep fixed-cell metadata outside timed closures."""
        helper_source = inspect.getsource(
            benchmark_electrostatics._torch_pme_static_metadata
        )
        benchmark_source = inspect.getsource(benchmark_electrostatics.benchmark_pme)

        assert "cell_inv_t = torch.linalg.inv(cell_3d).transpose(1, 2)" in helper_source
        assert "volume = torch.abs(torch.linalg.det(cell_3d))" in helper_source
        assert "moduli_x = compute_bspline_moduli_1d" in helper_source
        for name in ("volume", "cell_inv_t", "moduli_x", "moduli_y", "moduli_z"):
            assert benchmark_source.count(f"{name}=static_metadata.{name}") == 2
        assert '"pme_cache_mode": "full_static"' in benchmark_source

    def test_torch_el_benchmarks_report_cuda_event_timing_metadata(self):
        """Torch EL distinguishes full timing from optional component timing."""
        pme_source = inspect.getsource(benchmark_electrostatics.benchmark_pme)
        ewald_source = inspect.getsource(benchmark_electrostatics.benchmark_ewald)

        for source in (pme_source, ewald_source):
            assert '"timing_method": "torch_cuda_events"' in source
            assert 'timing_method_real = "not_measured"' in source
            assert 'timing_method_reciprocal = "not_measured"' in source
            assert 'timing_method_real = "torch_cuda_events"' in source
            assert 'timing_method_reciprocal = "torch_cuda_events"' in source
            assert '"timing_method_real": timing_method_real' in source
            assert '"timing_method_reciprocal": timing_method_reciprocal' in source

    def test_torch_el_uses_energy_autograd_for_both_derivatives(self):
        """Reportable Torch EL never calls the optional direct-output flags."""
        for function in (
            benchmark_electrostatics.benchmark_pme,
            benchmark_electrostatics.benchmark_ewald,
        ):
            source = inspect.getsource(function)
            assert "_torch_energy_forces_charge_gradients" in source
            assert "compute_forces=" not in source
            assert "compute_charge_gradients=" not in source

    def test_torch_el_energy_autograd_returns_forces_and_charge_gradients(self):
        """The shared Torch helper differentiates one energy call by R and q."""
        positions = torch.tensor(
            [[1.0, -2.0, 3.0], [0.5, 1.5, -1.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        charges = torch.tensor(
            [2.0, -3.0],
            dtype=torch.float64,
            requires_grad=True,
        )

        def energy_fn(pos, charge):
            return pos.square().sum(dim=1) + charge.square()

        energies, forces, charge_gradients = (
            benchmark_electrostatics._torch_energy_forces_charge_gradients(
                energy_fn,
                positions,
                charges,
            )
        )

        torch.testing.assert_close(
            energies,
            positions.square().sum(dim=1) + charges.square(),
        )
        torch.testing.assert_close(forces, -2.0 * positions)
        torch.testing.assert_close(charge_gradients, 2.0 * charges)

    def test_torch_el_component_timing_is_opt_in(self, monkeypatch):
        """The reportable EL default times only the complete workload."""
        positions = torch.ones((2, 3), dtype=torch.float64)
        charges = torch.ones(2, dtype=torch.float64)
        inputs = benchmark_electrostatics.ElectrostaticsInputs(
            positions=positions,
            charges=charges,
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.ones(3, dtype=torch.bool),
            batch_idx=torch.zeros(2, dtype=torch.int64),
            nl_data=torch.empty(0, dtype=torch.int64),
            nl_shifts=torch.empty((0, 3), dtype=torch.float64),
            nl_ptr=torch.zeros(3, dtype=torch.int64),
            backend="torch",
        )

        monkeypatch.setattr(
            benchmark_electrostatics,
            "generate_k_vectors_pme",
            lambda cell, mesh_dims: (torch.empty((1, 3)), torch.empty(1)),
        )
        monkeypatch.setattr(
            benchmark_electrostatics,
            "_torch_pme_static_metadata",
            lambda cell, mesh_dims, spline_order: SimpleNamespace(
                volume=torch.ones(1),
                cell_inv_t=torch.eye(3).unsqueeze(0),
                moduli_x=torch.ones(1),
                moduli_y=torch.ones(1),
                moduli_z=torch.ones(1),
            ),
        )

        def energy_fn(*, positions, charges, **kwargs):  # noqa: ARG001
            return positions.square().sum(dim=1) + charges.square()

        monkeypatch.setattr(benchmark_electrostatics, "ewald_real_space", energy_fn)
        monkeypatch.setattr(benchmark_electrostatics, "pme_reciprocal_space", energy_fn)
        monkeypatch.setattr(benchmark_electrostatics, "particle_mesh_ewald", energy_fn)
        monkeypatch.setattr(
            benchmark_electrostatics,
            "measure_memory_torch",
            lambda fn: (None, {"mem_delta_mb": 0.0, "mem_peak_gb": 0.0}),
        )
        timed = []

        def record_timing(fn, num_runs, warmup_runs=3, backend="torch"):  # noqa: ARG001
            timed.append(fn.__name__)
            fn()
            return 0.001

        monkeypatch.setattr(benchmark_electrostatics, "cuda_timed_runs", record_timing)

        result = benchmark_electrostatics.benchmark_pme(
            inputs,
            alpha=torch.ones(1),
            mesh_dims=(4, 4, 4),
            spline_order=4,
            accuracy=1e-4,
            num_runs=1,
            warmup_runs=0,
            profile_components=False,
        )

        assert timed == ["run_pme"]
        assert math.isnan(result["time_real_seconds"])
        assert math.isnan(result["time_reciprocal_seconds"])
        assert result["component_profiled"] is False

    def test_results_readme_documents_el_component_timing_metadata(self):
        """CSV schema docs include EL component timing labels."""
        readme = Path("docs/benchmarks/benchmark_results/README.md").read_text(
            encoding="utf-8"
        )

        assert "`timing_method_real`" in readme
        assert "`timing_method_reciprocal`" in readme
        assert "`pme_cache_mode`" in readme

    def test_d3_timing_excludes_neighbor_list_setup(self):
        """D3 benchmark rows time D3 separately from neighbor-list setup."""
        torch_source = inspect.getsource(benchmark_dftd3.benchmark_d3)
        jax_source = inspect.getsource(benchmark_dftd3._benchmark_d3_jax)
        row_source = inspect.getsource(benchmark_dftd3._d3_run_one_cutoff)

        assert "Neighbor-list setup is performed outside" in torch_source
        assert "time_d3 = cuda_timed_runs(run_d3, num_runs" in torch_source
        assert "def _run_d3_kernel(" in jax_source
        assert "jax.jit(_run_d3_kernel)" in jax_source
        assert "return _run_d3_kernel_jit(" in jax_source
        assert jax_source.count("neighbor_search_radius=") == 1
        assert "time_d3 = cuda_timed_runs(run_d3" in jax_source
        assert '"time_d3_seconds": time_d3' in torch_source
        assert '"time_d3_seconds": time_d3' in jax_source
        assert '"neighbor_setup_method": neighbor_setup_method' in torch_source
        assert '"neighbor_setup_method": neighbor_setup_method' in jax_source
        assert "time_d3_us_per_atom=time_d3_us_per_atom" in row_source
        assert 'neighbor_setup_method=r["neighbor_setup_method"]' in row_source

    @pytest.mark.parametrize(
        ("batch_size", "passes_radius"),
        [(1, True), (2, False)],
    )
    def test_jax_d3_neighbor_setup_matches_public_signature(
        self, monkeypatch, batch_size, passes_radius
    ):
        """Single and batched D3 setup pass only supported NL sizing kwargs."""
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")
        atoms_per_system = 2
        total_atoms = atoms_per_system * batch_size
        calls = []

        def fake_neighbor_list(**kwargs):
            calls.append(kwargs)
            max_neighbors = kwargs["max_neighbors"]
            return (
                jnp.zeros((total_atoms, max_neighbors), dtype=jnp.int32),
                jnp.zeros(total_atoms, dtype=jnp.int32),
                jnp.zeros((total_atoms, max_neighbors, 3), dtype=jnp.int32),
            )

        fake_api = {
            "jax": jax,
            "jnp": jnp,
            "neighbor_list": fake_neighbor_list,
            "estimate_batch_cell_list_sizes": lambda **_kwargs: (
                32,
                None,
                jnp.ones((batch_size, 3), dtype=jnp.int32),
            ),
            "dftd3": lambda **kwargs: jnp.zeros(
                kwargs["num_systems"], dtype=kwargs["positions"].dtype
            ),
        }
        cell = jnp.repeat(
            (jnp.eye(3, dtype=jnp.float32) * 8.0)[None, :, :],
            batch_size,
            axis=0,
        )
        pbc = jnp.ones((batch_size, 3), dtype=jnp.bool_)
        if batch_size == 1:
            cell = cell[0]
            pbc = pbc[0]
        data = {
            "positions": jnp.zeros((total_atoms, 3), dtype=jnp.float32),
            "cell": cell,
            "pbc": pbc,
            "batch_idx": jnp.repeat(
                jnp.arange(batch_size, dtype=jnp.int32), atoms_per_system
            ),
            "atomic_numbers": jnp.ones(total_atoms, dtype=jnp.int32),
            "atoms_per_system": atoms_per_system,
            "batch_size": batch_size,
            "total_atoms": total_atoms,
        }
        monkeypatch.setattr(
            benchmark_dftd3,
            "lazy_import_jax",
            lambda **_kwargs: fake_api,
        )
        monkeypatch.setattr(
            benchmark_dftd3,
            "estimate_max_neighbors",
            lambda *_args, **_kwargs: 4,
        )
        monkeypatch.setattr(
            benchmark_dftd3,
            "measure_memory_jax",
            lambda _run, _jax: (
                None,
                {"mem_delta_mb": math.nan, "mem_peak_gb": math.nan},
            ),
        )
        monkeypatch.setattr(
            benchmark_dftd3,
            "cuda_timed_runs",
            lambda *_args, **_kwargs: 0.001,
        )

        result = benchmark_dftd3._benchmark_d3_jax(
            data,
            cutoff=2.0,
            d3_params=jnp.zeros(1, dtype=jnp.float32),
            d3_func_params={"a1": 0.4, "a2": 4.5, "s8": 1.0},
            num_runs=1,
            warmup_runs=0,
        )

        assert len(calls) == 1
        assert calls[0]["method"] == (
            "cell_list" if batch_size == 1 else "batch_cell_list"
        )
        assert ("neighbor_search_radius" in calls[0]) is passes_radius
        assert result["neighbor_setup_method"] == calls[0]["method"]

    def test_el_setup_uses_cell_list_not_all_pairs_neighbor_setup(self):
        """EL setup avoids all-pairs neighbor-list construction for large grids."""
        source = inspect.getsource(benchmark_electrostatics._el_build_nl)

        assert "batch_cell_list(" in source
        assert "return_neighbor_list=False" in source
        assert "_torch_neighbor_matrix_to_list_chunked" in source
        assert 'method="batch_cell_list_atom_centric"' in source
        assert "batch_naive_neighbor_list" not in source
        assert 'method="batch_naive"' not in source

    def test_el_torch_chunked_matrix_to_list_preserves_coo_order(self):
        """Chunked Torch COO conversion matches row-major neighbor ordering."""
        neighbor_matrix = torch.tensor(
            [
                [1, 3, 4],
                [0, 4, 4],
                [0, 1, 4],
            ],
            dtype=torch.int32,
        )
        num_neighbors = torch.tensor([2, 1, 2], dtype=torch.int32)
        shifts = torch.arange(27, dtype=torch.int32).reshape(3, 3, 3)

        nl, ptr, nl_shifts = _torch_neighbor_matrix_to_list_chunked(
            neighbor_matrix,
            num_neighbors,
            shifts,
            fill_value=4,
        )

        assert nl.tolist() == [[0, 0, 1, 2, 2], [1, 3, 0, 0, 1]]
        assert ptr.tolist() == [0, 2, 3, 5]
        assert nl_shifts.tolist() == [
            shifts[0, 0].tolist(),
            shifts[0, 1].tolist(),
            shifts[1, 0].tolist(),
            shifts[2, 0].tolist(),
            shifts[2, 1].tolist(),
        ]

    def test_jax_el_runs_large_configs_first_and_cleans_jax(self):
        """JAX EL avoids shape-sweep fragmentation where possible."""
        order_source = inspect.getsource(
            benchmark_electrostatics._ordered_configs_for_backend
        )
        run_source = inspect.getsource(benchmark_electrostatics.run_from_config)
        d3_run_source = inspect.getsource(benchmark_dftd3.run_from_config)

        assert 'backend != "jax"' in order_source
        assert "planned_atom_counts(sys_name, cfg)[2]" in order_source
        assert "reverse=True" in order_source
        assert "configs = _ordered_configs_for_backend" in run_source
        assert "clean_jax()" in run_source
        assert "clean_jax()" in d3_run_source

    def test_jax_el_records_failure_stage_and_serial_timing_fallback(self):
        """JAX EL distinguishes setup OOMs from batched timing OOMs."""
        setup_source = inspect.getsource(benchmark_electrostatics._el_setup_config)
        method_source = inspect.getsource(benchmark_electrostatics._el_run_method)
        fallback_source = inspect.getsource(
            benchmark_electrostatics._jax_timed_with_serial_fallback
        )

        assert "neighbor_list_setup" in setup_source
        assert "failure_stage=setup.failure_stage" in inspect.getsource(
            benchmark_electrostatics.run_from_config
        )
        assert 'failure_stage=f"{method}_timing"' in method_source
        assert "jax_timed_serial" in fallback_source
        assert "jax_wall_block_each" in fallback_source
        assert "clean_jax(clear_executables=True)" in setup_source
        assert "clean_jax(clear_executables=True)" in method_source
        assert "clean_jax(clear_executables=True)" in fallback_source

    def test_jax_el_uses_explicit_atom_centric_nl_setup(self):
        """EL setup avoids auto pair-centric JAX FFI during reportable runs."""
        source = inspect.getsource(benchmark_electrostatics._el_build_nl)

        assert 'method="batch_cell_list_atom_centric"' in source
        assert 'strategy="atom_centric"' not in source
        assert 'atom_centric_path="direct"' in source

    def test_d3_el_generic_oom_paths_clean_gpu(self):
        """Generic JAX-style OOM exceptions clean state before continuing."""
        for function in (
            benchmark_dftd3._d3_run_one_cutoff,
            benchmark_electrostatics._el_setup_config,
            benchmark_electrostatics._el_run_method,
        ):
            source = inspect.getsource(function)
            assert "error_type = failure_error_type(e)" in source
            assert 'if error_type == "OutOfMemoryError":' in source
            assert "clean_gpu()" in source

    def test_d3_missing_parameter_file_is_generated(self, monkeypatch, tmp_path):
        """D3 benchmark can populate the configured parameter cache path."""
        utils_module = ModuleType("examples.dispersion.utils")

        def fake_extract_dftd3_parameters():
            return {
                "rcov": torch.ones(1),
                "r4r2": torch.ones(1),
                "c6ab": torch.ones(1),
                "cn_ref": torch.ones(1),
            }

        utils_module.extract_dftd3_parameters = fake_extract_dftd3_parameters
        monkeypatch.setitem(sys.modules, "examples", ModuleType("examples"))
        monkeypatch.setitem(
            sys.modules, "examples.dispersion", ModuleType("examples.dispersion")
        )
        monkeypatch.setitem(sys.modules, "examples.dispersion.utils", utils_module)
        params_path = tmp_path / "cache" / "dftd3_parameters.pt"

        benchmark_dftd3._ensure_d3_parameter_file(params_path)

        assert params_path.exists()
        loaded = torch.load(params_path, map_location="cpu", weights_only=True)
        assert sorted(loaded) == ["c6ab", "cn_ref", "r4r2", "rcov"]

    def test_d3_default_parameter_path_honors_xdg_cache_home(
        self, monkeypatch, tmp_path
    ):
        """Default D3 cache path can be redirected to scratch with XDG_CACHE_HOME."""
        scratch_cache = tmp_path / "scratch-cache"
        monkeypatch.setenv("XDG_CACHE_HOME", str(scratch_cache))

        resolved = benchmark_dftd3._resolve_d3_params_path(
            "~/.cache/nvalchemiops/dftd3_parameters.pt"
        )
        explicit = benchmark_dftd3._resolve_d3_params_path(
            tmp_path / "explicit" / "dftd3_parameters.pt"
        )

        assert resolved == scratch_cache / "nvalchemiops" / "dftd3_parameters.pt"
        assert explicit == tmp_path / "explicit" / "dftd3_parameters.pt"

    def test_d3_params_path_override_only_updates_d3_config(self, tmp_path):
        """Unified suite can point D3 at a pre-seeded scratch parameter file."""
        params_path = tmp_path / "scratch" / "dftd3_parameters.pt"
        args = SimpleNamespace(
            timing_runs=None,
            warmup_runs=None,
            system=None,
            mode=None,
            output_dir=None,
            backend=None,
            methods=None,
            dry_run=False,
            max_total_atoms=None,
            d3_params_path=params_path,
        )

        d3_merged = merge_common_cli_overrides(
            load_yaml_config(benchmark_suite.RUNNERS["d3"]["config"]),
            args,
        )
        nl_merged = merge_common_cli_overrides(
            load_yaml_config(benchmark_suite.RUNNERS["nl"]["config"]),
            args,
        )

        assert d3_merged["params_path"] == str(params_path)
        assert "params_path" not in nl_merged

    def test_jax_cluster_tile_uses_static_host_metadata(self):
        """Only the traceable single-system cluster-tile path is jitted."""
        source = inspect.getsource(benchmark_neighborlist._benchmark_nl_jax)
        cluster_source, batch_source = source.split(
            'elif jax_family == "cluster_tile":', 1
        )[1].split('elif jax_family == "batch_cluster_tile":', 1)

        assert "batch_ptr=batch_ptr" in source
        assert "pbc=pbc" in source
        assert "batch_ptr_static" not in source
        assert 'jax_api["cluster_tile_neighbor_list"]' in cluster_source
        assert "_validate_jax_cluster_tile_pbc(pbc, jax, jnp)" in cluster_source
        assert "compiled_nl = jax.jit(direct_nl_kernel)" in cluster_source
        assert "compiled_nl = jax.jit" not in batch_source

    @pytest.mark.parametrize("pbc", [None, np.array([True, False, True])])
    def test_jax_cluster_tile_rejects_nonperiodic_inputs(self, pbc):
        """The direct timing path preserves the public fully-periodic guard."""
        fake_jax = SimpleNamespace(device_get=np.asarray)
        fake_jnp = SimpleNamespace(all=np.all)

        with pytest.raises(NotImplementedError, match="fully periodic pbc"):
            benchmark_neighborlist._validate_jax_cluster_tile_pbc(
                pbc, fake_jax, fake_jnp
            )

    def test_jax_cluster_tile_accepts_fully_periodic_inputs(self):
        """The direct timing path accepts the same periodic inputs as the API."""
        fake_jax = SimpleNamespace(device_get=np.asarray)
        fake_jnp = SimpleNamespace(all=np.all)

        benchmark_neighborlist._validate_jax_cluster_tile_pbc(
            np.ones(3, dtype=bool), fake_jax, fake_jnp
        )

    @pytest.mark.gpu
    def test_jax_cluster_tile_benchmark_runs_through_public_jit_path(self):
        """The unbatched cluster-tile public API runs through its jitted path."""
        import jax
        import jax.numpy as jnp

        if not any(device.platform == "gpu" for device in jax.local_devices()):
            pytest.skip("JAX CUDA device is required")

        positions = (
            jnp.arange(96, dtype=jnp.float32).reshape(32, 3)
            * jnp.array([0.173, 0.337, 0.491], dtype=jnp.float32)
        ) % 7.0
        data = {
            "positions": positions,
            "cell": jnp.eye(3, dtype=jnp.float32) * 8.0,
            "pbc": jnp.ones((3,), dtype=jnp.bool_),
            "batch_idx": jnp.zeros((32,), dtype=jnp.int32),
            "atoms_per_system": 32,
            "batch_size": 1,
            "total_atoms": 32,
        }

        result = benchmark_neighborlist._benchmark_nl_jax(
            data,
            cutoff=1.0,
            method="cluster_tile",
            num_runs=1,
            warmup_runs=0,
        )

        assert result["time_seconds"] > 0.0
        assert result["timing_method"] == "jax_wall_block_until_ready"

    @pytest.mark.gpu
    def test_jax_batch_cluster_tile_benchmark_keeps_batch_ptr_concrete(self):
        """The reportable batched cluster-tile path traces with concrete sizing."""
        import jax
        import jax.numpy as jnp

        if not any(device.platform == "gpu" for device in jax.local_devices()):
            pytest.skip("JAX CUDA device is required")

        base_positions = (
            jnp.arange(96, dtype=jnp.float32).reshape(32, 3)
            * jnp.array([0.173, 0.337, 0.491], dtype=jnp.float32)
        ) % 7.0
        positions = jnp.concatenate([base_positions, base_positions], axis=0)
        cell = jnp.repeat(
            (jnp.eye(3, dtype=jnp.float32) * 8.0)[None, :, :],
            2,
            axis=0,
        )
        data = {
            "positions": positions,
            "cell": cell,
            "pbc": jnp.ones((2, 3), dtype=jnp.bool_),
            "batch_idx": jnp.repeat(jnp.arange(2, dtype=jnp.int32), 32),
            "atoms_per_system": 32,
            "batch_size": 2,
            "total_atoms": 64,
        }

        result = benchmark_neighborlist._benchmark_nl_jax(
            data,
            cutoff=1.0,
            method="batch_cluster_tile",
            num_runs=1,
            warmup_runs=0,
        )

        assert result["time_seconds"] > 0.0
        assert result["timing_method"] == "jax_wall_block_until_ready"

    @pytest.mark.parametrize(
        ("method", "batch_size", "neighbor_search_radius", "passes_radius"),
        [
            (
                "cell_list_pair_centric",
                1,
                np.array([2, 2, 2], dtype=np.int32),
                True,
            ),
            (
                "batch_cell_list_pair_centric",
                2,
                np.array([[2, 2, 2], [2, 2, 2]], dtype=np.int32),
                False,
            ),
        ],
    )
    def test_jax_pair_centric_benchmark_uses_concrete_sizing_outside_jit(
        self,
        monkeypatch,
        method,
        batch_size,
        neighbor_search_radius,
        passes_radius,
    ):
        """Pair-centric timing follows each public API's sizing contract."""
        calls = []

        def fail_jit(*_args, **_kwargs):
            pytest.fail("explicit pair-centric must not be wrapped in jax.jit")

        def fake_neighbor_list(**kwargs):
            calls.append(kwargs)
            if passes_radius:
                assert isinstance(kwargs["neighbor_search_radius"], np.ndarray)
            else:
                assert "neighbor_search_radius" not in kwargs
            return np.zeros((1,), dtype=np.int32)

        def fake_timing(run_nl, *_args, **_kwargs):
            run_nl()
            return 0.001, "jax_wall_block_until_ready"

        fake_jax = SimpleNamespace(jit=fail_jit, device_get=np.asarray)
        fake_jnp = SimpleNamespace(
            arange=lambda size, dtype: np.arange(size, dtype=dtype),
            int32=np.int32,
        )
        fake_api = {
            "jax": fake_jax,
            "jnp": fake_jnp,
            "neighbor_list": fake_neighbor_list,
            "estimate_cell_list_sizes": lambda **_kwargs: (
                64,
                None,
                neighbor_search_radius,
            ),
            "estimate_batch_cell_list_sizes": lambda **_kwargs: (
                128,
                None,
                neighbor_search_radius,
            ),
            "compute_naive_num_shifts": lambda **_kwargs: pytest.fail(
                "naive sizing is unrelated"
            ),
        }
        total_atoms = 8 * batch_size
        data = {
            "positions": np.zeros((total_atoms, 3), dtype=np.float32),
            "cell": np.repeat(
                (np.eye(3, dtype=np.float32) * 8.0)[None, :, :],
                batch_size,
                axis=0,
            ),
            "pbc": np.ones((batch_size, 3), dtype=bool),
            "batch_idx": np.repeat(np.arange(batch_size, dtype=np.int32), 8),
            "atoms_per_system": 8,
            "batch_size": batch_size,
            "total_atoms": total_atoms,
        }

        monkeypatch.setattr(benchmark_neighborlist, "lazy_import_jax", lambda: fake_api)
        monkeypatch.setattr(
            benchmark_neighborlist,
            "estimate_max_neighbors",
            lambda *_args, **_kwargs: 16,
        )
        monkeypatch.setattr(
            benchmark_neighborlist,
            "measure_memory_jax",
            lambda _run, _jax: (
                None,
                {"mem_delta_mb": math.nan, "mem_peak_gb": math.nan},
            ),
        )
        monkeypatch.setattr(
            benchmark_neighborlist,
            "_jax_nl_timed_with_serial_fallback",
            fake_timing,
        )

        result = benchmark_neighborlist._benchmark_nl_jax(
            data,
            cutoff=2.0,
            method=method,
            num_runs=1,
            warmup_runs=0,
        )

        assert len(calls) == 1
        assert calls[0]["method"] == method
        assert calls[0]["max_total_cells"] in {64, 128}
        assert result["max_neighbors"] == 16

    def test_torch_naive_uses_preallocated_output_and_shift_metadata(self):
        """Torch naive timings match JAX by reusing outputs and PBC shift metadata."""
        source = inspect.getsource(benchmark_neighborlist.benchmark_nl)

        assert "compute_naive_num_shifts" in source
        assert '"neighbor_matrix": torch.empty' in source
        assert '"neighbor_matrix_shifts": torch.empty' in source
        assert '"num_neighbors": torch.empty' in source
        assert '"shift_range_per_dimension": shift_range' in source
        assert '"num_shifts_per_system": num_shifts' in source
        assert '"max_shifts_per_system": int(max_shifts)' in source

    def test_max_total_atoms_override_updates_batch_scaling_grid(self):
        """CLI atom caps also resize batch-scaling config generation."""
        config = {
            "parameters": {},
            "runtime": {},
            "systems": {"cscl": {"enabled": True}},
            "scaling": {
                "system_size": {"enabled": True},
                "batch_scaling": {"enabled": True, "max_total_atoms": 128},
            },
            "methods": [{"name": "cell_list", "enabled": True}],
            "output": {"base_dir": "unused"},
        }
        args = SimpleNamespace(
            timing_runs=None,
            warmup_runs=None,
            system=None,
            mode=None,
            output_dir=None,
            backend=None,
            methods=None,
            dry_run=False,
            max_total_atoms=4096,
        )

        merged = merge_common_cli_overrides(config, args)

        assert merged["parameters"]["max_total_atoms"] == 4096
        assert merged["scaling"]["batch_scaling"]["max_total_atoms"] == 4096
        assert "max_total_atoms" not in merged["scaling"]["system_size"]

    def test_shipped_config_keeps_reportable_grid_without_hidden_overrides(self):
        """Shipped configs stay full/reportable unless CLI filters are explicit."""
        args = SimpleNamespace(
            timing_runs=None,
            warmup_runs=None,
            system=None,
            mode=None,
            output_dir=None,
            backend=None,
            methods=None,
            dry_run=False,
            max_total_atoms=None,
        )

        for runner in benchmark_suite.RUNNERS.values():
            config = load_yaml_config(runner["config"])
            merged = merge_common_cli_overrides(config, args)

            assert "profiles" not in merged
            assert merged.get("active_profile") is None
            assert merged["parameters"]["warmup_runs"] == 3
            assert merged["parameters"]["timing_runs"] == 10
            assert merged["scaling"]["constant_workload"]["target_atoms"] == 131072
            assert merged["scaling"]["batch_scaling"]["max_total_atoms"] == 131072
            assert "max_total_atoms" not in merged["parameters"]

            for system_config in merged["systems"].values():
                if "atom_counts" in system_config:
                    assert system_config["atom_counts"][-1] == 131072
                    assert 131072 in system_config["atom_counts"]

    def test_shipped_configs_use_canonical_methods(self):
        """Shipped YAMLs avoid stale review-era method names."""
        nl_config = load_yaml_config(benchmark_suite.RUNNERS["nl"]["config"])
        d3_config = load_yaml_config(benchmark_suite.RUNNERS["d3"]["config"])
        el_config = load_yaml_config(benchmark_suite.RUNNERS["el"]["config"])

        assert [m["name"] for m in nl_config["methods"]] == [
            "naive_scalar",
            "naive_tile",
            "cell_list_atom_centric",
            "cell_list_pair_centric",
            "cluster_tile",
        ]
        assert [m["name"] for m in d3_config["methods"]] == ["dftd3"]
        assert [m["name"] for m in el_config["methods"]] == ["pme", "ewald"]
        assert "compute_charge_gradients" not in el_config

        shipped_methods = {
            method["name"]
            for config in (nl_config, d3_config, el_config)
            for method in config["methods"]
        }
        assert "cell" not in shipped_methods
        assert not Path(
            "benchmarks/interactions/dispersion/validate_d3_energies.py"
        ).exists()

    def test_reportable_helper_uses_full_protocol_without_atom_cap(self):
        """The reportable helper does not hide a reduced workload."""
        source = Path("benchmarks/run_reportable_suite.sh").read_text()
        run_suite_body = source.split("run_suite() {", 1)[1].split("for backend in", 1)[
            0
        ]

        assert "--timing-runs 10" in run_suite_body
        assert "--warmup-runs 3" in run_suite_body
        assert "--max-total-atoms" not in run_suite_body

    def test_reportable_helper_exposes_hardware_neutral_shard_filters(self):
        """Reportable runs can be sharded without changing benchmark grids."""
        source = Path("benchmarks/run_reportable_suite.sh").read_text()

        assert 'BENCHMARK="all"' in source
        assert 'SYSTEM_FILTER=""' in source
        assert 'MODE_FILTER=""' in source
        assert "--benchmark all|nl|d3|el" in source
        assert "--system SYSTEM" in source
        assert "--mode MODE" in source
        assert '--benchmark "$suite_benchmark"' in source
        assert "warp) BACKENDS=(warp)" in source
        assert "all) BACKENDS=(torch jax warp)" in source
        assert "BACKENDS=(torch jax)" in source
        assert '"$BACKEND" == "all" && "$BENCHMARK" != "all"' in source
        assert 'BACKEND="all"' in source
        assert 'suite_benchmark="nl"' in source
        assert '"$BACKEND" == "warp" && "$BENCHMARK" != "nl"' in source
        assert 'common_args+=(--system "$SYSTEM_FILTER")' in source
        assert 'common_args+=(--mode "$MODE_FILTER")' in source
        assert "full_suite_selection()" in source
        assert "REPORTABLE_CSV_NAMES" in source
        assert "found_names != REPORTABLE_CSV_NAMES" in source
        assert "--validate-only" in source
        assert "validate_reportable_case_matrix" in source
        assert '"nl": {"torch", "jax", "warp"}' in source
        assert '"d3": {"torch", "jax"}' in source
        assert '"el": {"torch", "jax"}' in source
        assert (
            "Skipping full-suite CSV completeness check for selected shard." in source
        )

    def test_reportable_case_matrix_rejects_missing_cases(self, monkeypatch, tmp_path):
        """Final shard validation compares case keys, not only filenames."""
        base_row = {
            "benchmark": "nl",
            "backend": "torch",
            "system": "cscl",
            "scaling_mode": "system_size",
            "method": "naive_scalar",
            "atoms_per_system": 128,
            "batch_size": 1,
            "total_atoms": 128,
            "cutoff": 6.0,
            "timing_runs": 10,
            "warmup_runs": 3,
        }
        planned_rows = [base_row, {**base_row, "cutoff": 15.0}]
        monkeypatch.setattr(
            benchmark_suite,
            "build_reportable_plan",
            lambda _backends, nh3_dir=None: planned_rows,
        )
        csv_path = tmp_path / "nl-cscl-system-size-scaling.csv"
        with csv_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=base_row)
            writer.writeheader()
            writer.writerow(base_row)

        with pytest.raises(ValueError, match="case matrix mismatch; missing"):
            benchmark_suite.validate_reportable_case_matrix([csv_path], {"torch"})

    def test_reportable_case_matrix_requires_full_timing_protocol(
        self, monkeypatch, tmp_path
    ):
        """Publication validation requires the documented 3+10 protocol."""
        row = {
            "benchmark": "nl",
            "backend": "torch",
            "system": "cscl",
            "scaling_mode": "system_size",
            "method": "naive_scalar",
            "atoms_per_system": 128,
            "batch_size": 1,
            "total_atoms": 128,
            "cutoff": 6.0,
            "timing_runs": 5,
            "warmup_runs": 1,
        }
        monkeypatch.setattr(
            benchmark_suite,
            "build_reportable_plan",
            lambda _backends, nh3_dir=None: [row],
        )
        csv_path = tmp_path / "nl-cscl-system-size-scaling.csv"
        with csv_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=row)
            writer.writeheader()
            writer.writerow(row)

        with pytest.raises(ValueError, match="timing protocol mismatch"):
            benchmark_suite.validate_reportable_case_matrix([csv_path], {"torch"})

    def test_reportable_helper_keeps_outputs_and_caches_off_home(self):
        """Reportable cluster runs route writable state to scratch."""
        source = Path("benchmarks/run_reportable_suite.sh").read_text()

        assert 'reject_home_path "BENCHMARK_SCRATCH" "$SCRATCH"' in source
        assert 'reject_home_path "output directory" "$RESULT_DIR"' in source
        assert 'reject_home_path "D3 parameter path" "$D3_PARAMS_PATH"' in source
        assert "/home|/home/*)" not in source
        assert '"$ORIGINAL_HOME"|"$ORIGINAL_HOME"/*)' in source
        assert 'export HOME="$SCRATCH/home"' in source
        assert "BENCHMARK_D3_PARAMS_PATH" in source
        assert "--d3-params-path" in source
        for cache_var in (
            "XDG_CACHE_HOME",
            "UV_CACHE_DIR",
            "PRE_COMMIT_HOME",
            "UV_PROJECT_ENVIRONMENT",
            "WARP_CACHE_PATH",
            "TORCH_EXTENSIONS_DIR",
            "PYTORCH_KERNEL_CACHE_PATH",
            "JAX_COMPILATION_CACHE_DIR",
            "MPLCONFIGDIR",
            "CUDA_CACHE_PATH",
        ):
            assert f"export {cache_var}=" in source

    def test_reportable_helper_uses_compatible_sync_defaults(self):
        """Reportable cluster sync avoids mutually exclusive CUDA extras."""
        source = Path("benchmarks/run_reportable_suite.sh").read_text()

        assert "sync --all-extras" not in source
        assert (
            'UV_SYNC_ARGS="${UV_SYNC_ARGS:---extra torch --extra jax --group docs}"'
            in source
        )
        assert 'read -r -a uv_sync_args <<< "$UV_SYNC_ARGS"' in source
        assert '"$UV_BIN" sync "${uv_sync_args[@]}"' in source
        assert "uv_run_args+=(--no-sync)" in source
        assert 'run "${uv_run_args[@]}" python' in source
        assert '--python "${UV_PROJECT_ENVIRONMENT}/bin/python"' in source
        assert "BENCHMARK_PIP_PACKAGES" in source
        assert "pyyaml>=6.0.3" in source
        assert "nvidia-ml-py==13.590.48" in source
        assert 'echo "uv_sync_args=$UV_SYNC_ARGS"' in source

    def test_reportable_helper_validates_selected_backends_before_plotting(self):
        """Torch+JAX publication does not incorrectly require Warp rows."""
        source = Path("benchmarks/run_reportable_suite.sh").read_text()

        assert '--expected-backends "${BACKENDS[@]}"' in source

    def test_reportable_helper_preflights_nh3_inputs(self):
        """Reportable runs fail early when generated NH3 PDBs are missing."""
        source = Path("benchmarks/run_reportable_suite.sh").read_text()

        assert "CONFIG_PATHS" in source
        assert "ammonia_pbc_{atom_count}.pdb" in source
        assert "Missing NH3 PBC benchmark inputs" in source
        assert "generate_pbc_pdbs.sh" in source

    def test_cluster_tile_resolves_to_batch_api_for_batched_cases(self):
        """Method expansion maps cluster-tile to the batch-shaped API."""
        assert _nl_method_for_case("cluster_tile", batch_size=1, explicit=False) == (
            "cluster_tile"
        )
        assert _nl_method_for_case("cluster_tile", batch_size=4, explicit=False) == (
            "batch_cluster_tile"
        )
        assert _nl_method_for_case("cluster_tile", batch_size=4, explicit=True) == (
            "batch_cluster_tile"
        )

    def test_explicit_nl_methods_still_follow_batch_shape(self):
        """CLI method filters select a family, not a mismatched batch API."""
        assert _nl_method_for_case("batch_cell_list", batch_size=1, explicit=True) == (
            "cell_list"
        )
        assert _nl_method_for_case("cell_list", batch_size=4, explicit=True) == (
            "batch_cell_list"
        )
        assert _nl_method_for_case("naive_scalar", batch_size=4, explicit=True) == (
            "batch_naive_scalar"
        )
        assert _nl_method_for_case(
            "batch_cell_list_pair_centric", batch_size=1, explicit=True
        ) == ("cell_list_pair_centric")


class TestBenchmarkCliSurface:
    """Test that each benchmark CLI advertises only executable options."""

    @pytest.mark.parametrize(
        "runner",
        [benchmark_dftd3, benchmark_electrostatics],
    )
    def test_d3_and_el_reject_unsupported_warp_backend(self, monkeypatch, runner):
        """Standalone D3 and EL runners reject Warp during argument parsing."""
        monkeypatch.setattr(
            sys,
            "argv",
            ["benchmark", "--config", "config.yaml", "--backend", "warp"],
        )

        with pytest.raises(SystemExit, match="2"):
            runner.parse_args()

    def test_neighborlist_accepts_warp_backend(self, monkeypatch):
        """The standalone neighbor-list runner keeps its supported Warp option."""
        monkeypatch.setattr(
            sys,
            "argv",
            ["benchmark", "--config", "config.yaml", "--backend", "warp"],
        )

        assert benchmark_neighborlist.parse_args().backend == "warp"

    @pytest.mark.parametrize(
        "runner",
        [benchmark_neighborlist, benchmark_electrostatics],
    )
    def test_non_d3_runners_reject_d3_parameter_path(self, monkeypatch, runner):
        """D3 cache configuration is absent from unrelated standalone CLIs."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "benchmark",
                "--config",
                "config.yaml",
                "--d3-params-path",
                "d3-params.pt",
            ],
        )

        with pytest.raises(SystemExit, match="2"):
            runner.parse_args()

    @pytest.mark.parametrize(
        "runner_argv",
        [
            ["benchmark", "--config", "config.yaml"],
            ["benchmark-suite", "--benchmark", "d3"],
        ],
    )
    def test_d3_entrypoints_accept_parameter_path(self, monkeypatch, runner_argv):
        """Standalone D3 and the unified suite expose the D3 cache override."""
        params_path = Path("d3-params.pt")
        monkeypatch.setattr(
            sys,
            "argv",
            [*runner_argv, "--d3-params-path", str(params_path)],
        )

        if runner_argv[0] == "benchmark":
            args = benchmark_dftd3.parse_args()
        else:
            args = benchmark_suite.parse_args()

        assert args.d3_params_path == params_path


class TestSuiteBackendSelection:
    """Test suite-level backend compatibility guards."""

    def test_warp_backend_accepts_neighbor_list_only(self):
        """The Warp backend is valid for the NL benchmark."""
        validate_backend_selection("warp", {"nl"})

    def test_warp_backend_rejects_non_neighbor_list_benchmarks(self):
        """The suite rejects Warp for D3/EL before dispatch."""
        with pytest.raises(ValueError, match="not supported"):
            validate_backend_selection("warp", {"nl", "d3", "el"})

    def test_electrostatics_rejects_multipole_methods(self):
        """The unified suite keeps multipole EL outside this benchmark path."""
        with pytest.raises(ValueError, match="Multipole"):
            validate_method_selection(["multipole_ewald"], {"el"})


class TestBenchmarkSuitePlotting:
    """Test suite-level plotting control flow."""

    def test_nl_plot_labels_cluster_tile_family(self):
        """NL plot labels preserve user-facing strategy distinctions."""
        assert plot_benchmarks._nl_method_family("cluster_tile") == "cluster_tile"
        assert plot_benchmarks._nl_method_family("batch_cluster_tile") == "cluster_tile"
        assert plot_benchmarks._nl_method_label("batch_cluster_tile") == "Cluster tile"
        assert plot_benchmarks._nl_method_family("batch_naive_scalar") == (
            "naive_scalar"
        )
        assert plot_benchmarks._nl_method_label("batch_naive_tile") == "Naive tile"
        assert plot_benchmarks._nl_method_family("cell_list_pair_centric") == (
            "cell_list_pair_centric"
        )
        assert plot_benchmarks._nl_method_label("batch_cell_list_atom_centric") == (
            "Cell atom"
        )

    def test_plotting_does_not_assume_h100_vram_reference(self):
        """Memory plot hardware references must come from explicit metadata."""
        assert plot_benchmarks.GPU_VRAM_REFS == {}

    def test_nl_legend_uses_named_visual_dimensions(self):
        """NL legends separate method, backend, cutoff, and system keys."""
        figure, axis = plot_benchmarks.plt.subplots()
        try:
            plot_benchmarks._create_nl_dimension_legend(
                axis,
                ["cell_list_atom_centric", "naive_scalar"],
                [6.0, 15.0, 25.0],
                [128, 1024],
                ["torch", "jax"],
            )
            labels = [text.get_text() for text in axis.get_legend().get_texts()]

            assert {"Method", "Backend", "Cutoff", "System"}.issubset(labels)
            assert {"Cell atom", "Naive scalar", "6Å", "15Å", "25Å"}.issubset(labels)
            assert not any("··" in label for label in labels)
        finally:
            plot_benchmarks.plt.close(figure)

    def test_nl_visual_hierarchy_is_stable_across_cutoffs(self):
        """Family and strategy styling stays fixed as cutoff layers are added."""
        rows = [
            {
                "method": method,
                "cutoff": cutoff,
                "atoms_per_system": atoms,
                "batch_size": 1,
                "total_atoms": atoms,
                "time_us_per_atom": time_value,
                "throughput_atoms_per_sec": 1_000_000.0 / time_value,
                "mem_peak_gb": 1.0,
                "backend": "torch",
            }
            for method, method_scale in (
                ("cell_list_atom_centric", 1.0),
                ("cell_list_pair_centric", 1.1),
                ("naive_tile", 2.0),
            )
            for cutoff, cutoff_scale in ((6.0, 1.0), (25.0, 1.5))
            for atoms, time_value in (
                (128, method_scale * cutoff_scale),
                (256, method_scale * cutoff_scale * 0.8),
            )
        ]
        figure, axis = plot_benchmarks.plt.subplots()
        try:
            plot_benchmarks.render_nl_panel(
                axis,
                rows,
                system="cscl",
                mode="system_size",
                panel="time",
            )
            lines = {line.get_gid(): line for line in axis.lines}
            cell_atom_6 = lines["nl-cutoff-6A-cell-list-atom-centric-time"]
            cell_atom_25 = lines["nl-cutoff-25A-cell-list-atom-centric-time"]
            cell_pair_6 = lines["nl-cutoff-6A-cell-list-pair-centric-time"]
            naive_6 = lines["nl-cutoff-6A-naive-tile-time"]

            assert cell_atom_6.get_color() == cell_atom_25.get_color()
            assert cell_atom_6.get_marker() == cell_atom_25.get_marker()
            assert cell_atom_6.get_linestyle() == cell_atom_25.get_linestyle()
            assert cell_atom_6.get_fillstyle() != cell_atom_25.get_fillstyle()
            assert cell_atom_6.get_linewidth() != cell_atom_25.get_linewidth()

            assert cell_atom_6.get_color() != cell_pair_6.get_color()
            assert cell_atom_6.get_linestyle() != cell_pair_6.get_linestyle()
            assert cell_atom_6.get_marker() != cell_pair_6.get_marker()
            assert cell_atom_6.get_color() != naive_6.get_color()
            assert plot_benchmarks._nl_cutoff_style(15.0)["fillstyle"] == "none"
        finally:
            plot_benchmarks.plt.close(figure)

    def test_el_visual_hierarchy_is_stable_across_accuracies(self):
        """EL method styling stays fixed while accuracy adds a visual layer."""
        rows = [
            {
                "method": method,
                "accuracy": accuracy,
                "atoms_per_system": atoms,
                "batch_size": 1,
                "total_atoms": atoms,
                "time_us_per_atom": method_scale * accuracy_scale,
                "throughput_atoms_per_sec": 1_000_000.0
                / (method_scale * accuracy_scale),
                "mem_peak_gb": 1.0,
                "backend": "torch",
            }
            for method, method_scale in (("pme", 1.0), ("ewald", 2.0))
            for accuracy, accuracy_scale in ((1e-6, 1.0), (1e-4, 1.2))
            for atoms in (128, 256)
        ]
        figure, axis = plot_benchmarks.plt.subplots()
        try:
            plot_benchmarks.render_el_panel(
                axis,
                rows,
                system="cscl",
                mode="system_size",
                panel="time",
            )
            pme_style = plot_benchmarks.EL_METHOD_STYLES["pme"]
            ewald_style = plot_benchmarks.EL_METHOD_STYLES["ewald"]
            pme_lines = [
                line for line in axis.lines if line.get_color() == pme_style["color"]
            ]
            ewald_lines = [
                line for line in axis.lines if line.get_color() == ewald_style["color"]
            ]

            assert len(pme_lines) == len(ewald_lines) == 2
            assert len({line.get_marker() for line in pme_lines}) == 1
            assert len({line.get_linestyle() for line in pme_lines}) == 1
            assert {line.get_fillstyle() for line in pme_lines} == {"none", "full"}
            default_accuracy_line = next(
                line for line in pme_lines if line.get_linewidth() > 1.0
            )
            assert default_accuracy_line.get_fillstyle() == "none"
            assert len({line.get_linewidth() for line in pme_lines}) == 2
            assert pme_lines[0].get_marker() != ewald_lines[0].get_marker()
            assert pme_lines[0].get_linestyle() != ewald_lines[0].get_linestyle()

            labels = [text.get_text() for text in axis.get_legend().get_texts()]
            assert {"Method", "Accuracy", "PME", "Ewald", "e-6", "e-4"}.issubset(labels)
        finally:
            plot_benchmarks.plt.close(figure)

    @pytest.mark.parametrize("module", ["d3", "el"])
    def test_d3_el_batch_system_size_uses_line_and_marker_shape(self, module):
        """D3 and EL batch sizes never rely on subtle marker-size changes."""
        rows = []
        for atoms_per_system in (256, 8192):
            for batch_size in (1, 2):
                row = {
                    "method": "dftd3" if module == "d3" else "pme",
                    "atoms_per_system": atoms_per_system,
                    "batch_size": batch_size,
                    "total_atoms": atoms_per_system * batch_size,
                    "time_us_per_atom": 1.0 / batch_size,
                    "throughput_atoms_per_sec": 1_000_000.0 * batch_size,
                    "mem_peak_gb": 1.0,
                    "backend": "torch",
                }
                if module == "d3":
                    row.update({"cutoff": 15.0, "time_d3_us_per_atom": 0.8})
                else:
                    row["accuracy"] = 1e-6
                rows.append(row)

        figure, axis = plot_benchmarks.plt.subplots()
        try:
            renderer = (
                plot_benchmarks.render_d3_panel
                if module == "d3"
                else plot_benchmarks.render_el_panel
            )
            renderer(
                axis,
                rows,
                system="nh3",
                mode="batch_scaling",
                panel="time",
            )

            assert {line.get_marker() for line in axis.lines} == {"o", "D"}
            assert len({line.get_linestyle() for line in axis.lines}) == 2
            assert {line.get_markersize() for line in axis.lines} == {
                plot_benchmarks.DATA_MARKER_SIZE
            }
        finally:
            plot_benchmarks.plt.close(figure)

    def test_d3_legend_uses_shared_dimension_layout(self):
        """D3 legends use the same named-column layout as NL and EL."""
        assert (
            plot_benchmarks.D3_CUTOFF_STYLES[15.0]["linestyle"]
            != plot_benchmarks.D3_CUTOFF_STYLES[25.0]["linestyle"]
        )
        assert (
            plot_benchmarks.D3_CUTOFF_STYLES[15.0]["marker"]
            != plot_benchmarks.D3_CUTOFF_STYLES[25.0]["marker"]
        )
        assert (
            plot_benchmarks.D3_CUTOFF_COLORS[15.0]
            != plot_benchmarks.D3_CUTOFF_COLORS[25.0]
        )
        assert plot_benchmarks.D3_CUTOFF_STYLES[15.0]["fillstyle"] == "none"
        assert plot_benchmarks.D3_CUTOFF_STYLES[25.0]["fillstyle"] == "full"

        figure, axis = plot_benchmarks.plt.subplots()
        try:
            plot_benchmarks._create_d3_dimension_legend(
                axis,
                [15.0, 25.0],
                [128, 1024],
            )
            labels = [text.get_text() for text in axis.get_legend().get_texts()]

            assert {"Cutoff", "System", "15Å", "25Å", "N=128", "N=1k"}.issubset(labels)
            assert not any("··" in label for label in labels)
        finally:
            plot_benchmarks.plt.close(figure)

    def test_reportable_png_export_meets_4k_pixel_floor(self, tmp_path):
        """Single-panel PNGs retain enough source pixels for high-DPI displays."""
        plot_benchmarks.setup_plot_style()
        figure, axis = plot_benchmarks.plt.subplots(
            figsize=plot_benchmarks.SINGLE_PANEL_SIZE
        )
        output_path = tmp_path / "reportable-panel.png"
        try:
            axis.plot([128, 256], [1.0, 0.5], marker="o")
            axis.set_title("Benchmark | CsCl | System Size Scaling")
            axis.set_xlabel("System Size (atoms)")
            axis.set_ylabel("Time per atom [us]")
            plot_benchmarks._savefig_atomic(figure, output_path)

            height, width = plot_benchmarks.plt.imread(output_path).shape[:2]
            min_width, min_height = plot_styles.SINGLE_PANEL_MIN_PIXEL_SIZE
            assert plot_benchmarks.PNG_EXPORT_DPI >= 360
            assert width >= min_width
            assert height >= min_height
        finally:
            plot_benchmarks.plt.close(figure)

    @pytest.mark.parametrize("mode", ["system_size", "batch_scaling"])
    def test_backend_comparison_uses_nl_base_line_weight(
        self, monkeypatch, tmp_path, mode
    ):
        """EL/D3 comparison lines use the same restrained weight as NL."""
        rows = [
            {
                "success": True,
                "system": "cscl",
                "scaling_mode": mode,
                "method": method,
                "backend": backend,
                "accuracy": 1.0e-6,
                "atoms_per_system": 128,
                "batch_size": 1 if mode == "system_size" else 2,
                "total_atoms": 128 if mode == "system_size" else 256,
                "time_us_per_atom": 1.0,
                "throughput_atoms_per_sec": 1_000_000.0,
                "backend_comparable": True,
                "timing_scope": "backend_comparison",
            }
            for method in ("ewald", "pme")
            for backend in ("torch", "jax")
        ]
        line_widths = []
        original_plot_data_line = plot_benchmarks._plot_data_line

        def record_line_weight(*args, **kwargs):
            line_widths.append(kwargs.get("linewidth", plot_benchmarks.DATA_LINE_WIDTH))
            return original_plot_data_line(*args, **kwargs)

        monkeypatch.setattr(plot_benchmarks, "load_csv", lambda _path: rows)
        monkeypatch.setattr(
            plot_benchmarks,
            "_plot_data_line",
            record_line_weight,
        )
        monkeypatch.setattr(plot_benchmarks, "_savefig_atomic", lambda *_args: None)

        assert plot_benchmarks.plot_comparison_panel(
            tmp_path / f"el-cscl-{mode}.csv",
            "time",
            tmp_path / "comparison.png",
            "el",
        )
        assert line_widths
        assert set(line_widths) == {plot_benchmarks.DATA_LINE_WIDTH}

    def test_nl_backend_comparison_layers_every_cutoff(self, monkeypatch, tmp_path):
        """Backend overlays retain every cutoff for the page-level switches."""
        rows = [
            {
                "success": True,
                "system": "cscl",
                "scaling_mode": "system_size",
                "method": "naive_scalar",
                "backend": backend,
                "cutoff": cutoff,
                "atoms_per_system": atoms,
                "batch_size": 1,
                "total_atoms": atoms,
                "time_us_per_atom": cutoff / atoms,
                "throughput_atoms_per_sec": atoms / cutoff * 1_000_000.0,
                "backend_comparable": True,
                "timing_scope": "backend_comparison",
            }
            for backend in ("torch", "jax")
            for cutoff in (6.0, 15.0, 25.0)
            for atoms in (128, 256)
        ]
        captured_lines = []

        def capture_figure(
            figure,
            output_path,
            *,
            dpi=plot_benchmarks.PNG_EXPORT_DPI,
        ):  # noqa: ARG001
            captured_lines.extend(figure.axes[0].lines)

        monkeypatch.setattr(plot_benchmarks, "load_csv", lambda _path: rows)
        monkeypatch.setattr(plot_benchmarks, "_savefig_atomic", capture_figure)

        assert plot_benchmarks.plot_comparison_panel(
            tmp_path / "nl-cscl-system-size-scaling.csv",
            "time",
            tmp_path / "comparison.svg",
            "nl",
            layer_all_params=True,
        )

        assert len(captured_lines) == 6
        assert {line.get_gid().split("-", 3)[2] for line in captured_lines} == {
            "6A",
            "15A",
            "25A",
        }
        assert {line.get_color() for line in captured_lines} == {
            plot_benchmarks._nl_method_color("naive_scalar")
        }
        assert len({line.get_linestyle() for line in captured_lines}) == 2
        assert {line.get_fillstyle() for line in captured_lines} == {
            "none",
            "left",
            "full",
        }

    @pytest.mark.parametrize(
        ("module", "method", "param_name", "param_value", "expected_color"),
        [
            ("d3", "dftd3", "cutoff", 15.0, plot_benchmarks.D3_CUTOFF_COLORS[15.0]),
            (
                "el",
                "pme",
                "accuracy",
                1.0e-6,
                plot_benchmarks.EL_METHOD_STYLES["pme"]["color"],
            ),
        ],
    )
    def test_backend_comparison_preserves_scientific_identity_color(
        self,
        monkeypatch,
        tmp_path,
        module,
        method,
        param_name,
        param_value,
        expected_color,
    ):
        """D3 cutoff and EL method keep their color while backend uses line style."""
        rows = [
            {
                "success": True,
                "system": "cscl",
                "scaling_mode": "system_size",
                "method": method,
                "backend": backend,
                param_name: param_value,
                "atoms_per_system": atoms,
                "batch_size": 1,
                "total_atoms": atoms,
                "time_us_per_atom": 1.0,
                "time_d3_us_per_atom": 0.8,
                "throughput_atoms_per_sec": 1_000_000.0,
                "backend_comparable": True,
                "timing_scope": "backend_comparison",
            }
            for backend in ("torch", "jax")
            for atoms in (128, 256)
        ]
        captured_lines = []

        def capture_figure(
            figure,
            output_path,
            *,
            dpi=plot_benchmarks.PNG_EXPORT_DPI,
        ):  # noqa: ARG001
            captured_lines.extend(figure.axes[0].lines)

        monkeypatch.setattr(plot_benchmarks, "load_csv", lambda _path: rows)
        monkeypatch.setattr(plot_benchmarks, "_savefig_atomic", capture_figure)

        assert plot_benchmarks.plot_comparison_panel(
            tmp_path / f"{module}-cscl-system-size-scaling.csv",
            "time",
            tmp_path / "comparison.png",
            module,
            fixed_param=param_value,
        )

        assert len(captured_lines) == 2
        assert {line.get_color() for line in captured_lines} == {expected_color}
        assert len({line.get_linestyle() for line in captured_lines}) == 2

    def test_docs_cutoff_selector_does_not_depend_on_object_documents(self):
        """The cutoff controls work for directly opened file:// documentation."""
        source = Path("docs/benchmarks/neighborlist.md").read_text(encoding="utf-8")
        conf = Path("docs/conf.py").read_text(encoding="utf-8")

        assert "contentDocument" not in source
        assert "object.nl-cutoff-plot" not in source
        assert "svg.nl-cutoff-plot" in source
        assert "inline_neighborlist_svgs" in conf

    def test_inline_nl_svg_namespaces_ids_and_marks_cutoff_layers(self, tmp_path):
        """Inlining avoids cross-plot ID collisions and exposes cutoff groups."""
        svg_path = tmp_path / "nl-cscl-system-size-scaling-time.svg"
        svg_path.write_text(
            """<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg">
  <defs><clipPath id="clip"><path /></clipPath></defs>
  <g id="nl-cutoff-6A-cell-list" clip-path="url(#clip)">
    <use href="#clip" />
  </g>
</svg>
""",
            encoding="utf-8",
        )

        markup = docs_benchmark_sphinxext._inline_svg_markup(svg_path, "NL plot")

        namespace = "nl-cscl-system-size-scaling-time"
        assert "<?xml" not in markup
        assert 'class="nl-cutoff-plot"' in markup
        assert 'data-nl-cutoff="6A"' in markup
        assert f'id="{namespace}-clip"' in markup
        assert f"url(#{namespace}-clip)" in markup
        assert f'href="#{namespace}-clip"' in markup

    def test_nl_backend_plots_are_inline_svg_candidates(self):
        """Backend comparisons participate in the same cutoff interaction."""
        assert docs_benchmark_sphinxext._NL_PLOT_RE.fullmatch(
            "nl-backend-cscl-system-size-scaling-time.png"
        )

    def test_nl_backend_generation_writes_png_and_layered_svg(
        self, monkeypatch, tmp_path
    ):
        """Backend comparison generation produces an interactive SVG peer."""
        results_dir = tmp_path / "results"
        output_dir = tmp_path / "plots"
        results_dir.mkdir()
        output_dir.mkdir()
        (results_dir / "nl-cscl-system-size-scaling.csv").touch()
        calls = []

        def record_plot(csv_path, panel, output_path, module, **kwargs):
            calls.append((Path(output_path).suffix, panel, module, kwargs))
            Path(output_path).touch()
            return True

        monkeypatch.setattr(
            docs_generate_plots,
            "_suite_csv_dirs",
            lambda _results_dir: [results_dir],
        )
        monkeypatch.setattr(plot_benchmarks, "plot_comparison_panel", record_plot)

        docs_generate_plots.generate_nl_backend_comparison_plots(
            results_dir, output_dir
        )

        assert {(suffix, panel) for suffix, panel, _, _ in calls} == {
            (".png", "time"),
            (".svg", "time"),
            (".png", "throughput"),
            (".svg", "throughput"),
        }
        assert all(module == "nl" for _, _, module, _ in calls)
        assert all(kwargs["layer_all_params"] is True for *_, kwargs in calls)

    def test_nl_batch_modes_do_not_plot_single_api_baseline_points(self):
        """Batch scaling charts contain connected batch-API series only."""
        rows = [
            {
                "method": "batch_naive_scalar",
                "cutoff": 6.0,
                "atoms_per_system": 128,
                "batch_size": 4,
                "total_atoms": 512,
                "time_us_per_atom": 1.0,
                "throughput_atoms_per_sec": 1_000_000.0,
                "mem_peak_gb": 1.0,
                "backend": "torch",
            },
            {
                "method": "batch_naive_scalar",
                "cutoff": 6.0,
                "atoms_per_system": 256,
                "batch_size": 2,
                "total_atoms": 512,
                "time_us_per_atom": 2.0,
                "throughput_atoms_per_sec": 500_000.0,
                "mem_peak_gb": 1.0,
                "backend": "torch",
            },
            {
                "method": "naive_scalar",
                "cutoff": 6.0,
                "atoms_per_system": 512,
                "batch_size": 1,
                "total_atoms": 512,
                "time_us_per_atom": 3.0,
                "throughput_atoms_per_sec": 333_333.0,
                "mem_peak_gb": 1.0,
                "backend": "torch",
            },
        ]
        figure, axis = plot_benchmarks.plt.subplots()
        try:
            plot_benchmarks.render_nl_panel(
                axis,
                rows,
                system="cscl",
                mode="constant_workload",
                panel="time",
            )
            plotted = [line for line in axis.lines if line.get_gid() is not None]
            assert len(plotted) == 1
            assert list(plotted[0].get_xdata()) == [128, 256]
        finally:
            plot_benchmarks.plt.close(figure)

    def test_nl_batch_multi_cutoff_series_stay_separate(self):
        """Batch plots never connect points from different cutoff radii."""
        rows = [
            {
                "method": "batch_cell_list_atom_centric",
                "cutoff": cutoff,
                "atoms_per_system": 128,
                "batch_size": batch_size,
                "total_atoms": 128 * batch_size,
                "time_us_per_atom": base_time / batch_size,
                "throughput_atoms_per_sec": batch_size * 1_000_000.0,
                "mem_peak_gb": base_time,
                "backend": "torch",
            }
            for cutoff, base_time in ((6.0, 1.0), (25.0, 4.0))
            for batch_size in (1, 2)
        ]

        figure, axis = plot_benchmarks.plt.subplots()
        try:
            plot_benchmarks.render_nl_panel(
                axis,
                rows,
                system="cscl",
                mode="batch_scaling",
                panel="time",
            )

            assert len(axis.lines) == 2
            assert [list(line.get_xdata()) for line in axis.lines] == [
                [128, 256],
                [128, 256],
            ]
            assert axis.get_title() == "Neighbor List | CsCl | Batch Scaling"
        finally:
            plot_benchmarks.plt.close(figure)

    def test_nl_line_breaks_when_jax_timing_boundary_changes(self):
        """Batch and per-call JAX timings are not joined as one curve."""
        rows = [
            {
                "method": "cell_list_atom_centric",
                "cutoff": 25.0,
                "atoms_per_system": atoms,
                "batch_size": 1,
                "total_atoms": atoms,
                "time_us_per_atom": time_value,
                "throughput_atoms_per_sec": 1_000_000.0 / time_value,
                "mem_peak_gb": math.nan,
                "backend": "jax",
                "success": True,
                "timing_method": timing_method,
            }
            for atoms, time_value, timing_method in (
                (128, 1.0, "jax_wall_block_until_ready"),
                (256, 0.8, "jax_wall_block_until_ready"),
                (512, 0.7, "jax_wall_block_each"),
            )
        ]

        figure, axis = plot_benchmarks.plt.subplots()
        try:
            plot_benchmarks.render_nl_panel(
                axis,
                rows,
                system="cscl",
                mode="system_size",
                panel="time",
            )

            x_values = np.asarray(axis.lines[0].get_xdata(), dtype=float)
            assert x_values[:2].tolist() == [128.0, 256.0]
            assert np.isnan(x_values[2])
            assert x_values[3] == 512.0
        finally:
            plot_benchmarks.plt.close(figure)

    def test_el_line_breaks_at_failed_coordinate(self):
        """A failed intermediate case remains a gap in the rendered line."""
        rows = [
            {
                "method": "pme",
                "accuracy": 1e-6,
                "atoms_per_system": atoms,
                "batch_size": 1,
                "total_atoms": atoms,
                "time_us_per_atom": time_value,
                "throughput_atoms_per_sec": throughput,
                "mem_peak_gb": math.nan,
                "backend": "jax",
                "success": success,
                "timing_method": "jax_wall_block_until_ready",
            }
            for atoms, time_value, throughput, success in (
                (432, 1.0, 1_000_000.0, True),
                (686, math.nan, math.nan, False),
                (1024, 0.8, 1_250_000.0, True),
            )
        ]

        figure, axis = plot_benchmarks.plt.subplots()
        try:
            plot_benchmarks.render_el_panel(
                axis,
                rows,
                system="cscl",
                mode="system_size",
                panel="time",
            )

            y_values = np.asarray(axis.lines[0].get_ydata(), dtype=float)
            assert y_values[0] == 1.0
            assert np.isnan(y_values[1])
            assert y_values[2] == 0.8
        finally:
            plot_benchmarks.plt.close(figure)

    def test_el_comparison_uses_consistent_marker_fill(self, tmp_path, monkeypatch):
        """Torch and JAX use the same method marker treatment in EL overlays."""
        csv_path = tmp_path / "el-cscl-system-size-scaling.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "system",
                    "scaling_mode",
                    "method",
                    "backend",
                    "total_atoms",
                    "atoms_per_system",
                    "batch_size",
                    "accuracy",
                    "time_us_per_atom",
                    "throughput_atoms_per_sec",
                    "mem_peak_gb",
                    "success",
                ],
            )
            writer.writeheader()
            for backend, time_value in (("torch", 1.0), ("jax", 1.2)):
                for atoms in (128, 256):
                    writer.writerow(
                        {
                            "system": "cscl",
                            "scaling_mode": "system_size",
                            "method": "pme",
                            "backend": backend,
                            "total_atoms": atoms,
                            "atoms_per_system": atoms,
                            "batch_size": 1,
                            "accuracy": 1e-6,
                            "time_us_per_atom": time_value,
                            "throughput_atoms_per_sec": 1_000_000.0 / time_value,
                            "mem_peak_gb": 1.0,
                            "success": True,
                        }
                    )

        marker_faces = []
        markers = []

        def capture_figure(
            figure,
            output_path,
            *,
            dpi=plot_benchmarks.PNG_EXPORT_DPI,
        ):  # noqa: ARG001
            marker_faces.extend(
                line.get_markerfacecolor() for line in figure.axes[0].lines
            )
            markers.extend(line.get_marker() for line in figure.axes[0].lines)

        monkeypatch.setattr(plot_benchmarks, "_savefig_atomic", capture_figure)
        plot_benchmarks.plot_comparison_panel(
            csv_path,
            "time",
            tmp_path / "comparison.png",
            "el",
            fixed_param=1e-6,
        )

        assert len(marker_faces) == 2
        assert len(set(markers)) == 1
        assert marker_faces == ["none", "none"]

    def test_comparison_plotter_filters_noncomparable_and_serial_rows(self):
        """Backend overlays keep CSV-visible but non-comparable rows out."""
        assert plot_benchmarks._is_backend_comparison_row(
            {
                "success": True,
                "backend_comparable": True,
                "timing_scope": "backend_comparison",
                "timing_method": "jax_wall_block_until_ready",
            }
        )
        assert not plot_benchmarks._is_backend_comparison_row(
            {
                "success": True,
                "backend_comparable": False,
                "timing_scope": "coverage_only_pair_centric",
                "timing_method": "jax_wall_block_until_ready",
            }
        )
        assert not plot_benchmarks._is_backend_comparison_row(
            {
                "success": True,
                "backend_comparable": True,
                "timing_scope": "backend_comparison",
                "timing_method": "jax_wall_block_each",
            }
        )

    def test_gitignore_allows_unified_benchmark_csvs(self):
        """Current unified NL/D3/EL docs CSVs are not ignored."""
        gitignore = Path(".gitignore").read_text(encoding="utf-8")

        assert "!docs/benchmarks/benchmark_results/nl-*.csv" in gitignore
        assert "!docs/benchmarks/benchmark_results/d3-*.csv" in gitignore
        assert "!docs/benchmarks/benchmark_results/el-*.csv" in gitignore

    def test_plot_data_line_skips_empty_memory_series(self):
        """All-missing memory series do not create ghost legend entries."""
        fig, ax = plot_benchmarks.plt.subplots()
        try:
            plotted = plot_benchmarks._plot_data_line(
                ax,
                [128, 256],
                [None, math.nan],
                color="black",
                linestyle="-",
                marker="o",
                label="D3 jax",
            )
            assert plotted is False
            assert list(ax.lines) == []
        finally:
            plot_benchmarks.plt.close(fig)

    def test_plot_only_short_circuits_inside_main(self, monkeypatch, tmp_path):
        """``main`` handles plot-only mode before importing any runners."""

        def fake_parse_args():
            return SimpleNamespace(
                benchmark=["all"],
                backend=None,
                plot_only=tmp_path,
                plots=["time"],
                expected_backends=["torch", "jax"],
            )

        called = {}

        def fake_generate_plots(
            results_dir,
            plots=None,
            *,
            require_complete_suite=False,
            expected_backends=None,
        ):
            called["results_dir"] = Path(results_dir)
            called["plots"] = plots
            called["require_complete_suite"] = require_complete_suite
            called["expected_backends"] = expected_backends
            return True

        def fail_import(_module_name):
            pytest.fail("plot-only mode should not import benchmark runners")

        monkeypatch.setattr(benchmark_suite, "parse_args", fake_parse_args)
        monkeypatch.setattr(benchmark_suite, "_generate_plots", fake_generate_plots)
        monkeypatch.setattr(benchmark_suite.importlib, "import_module", fail_import)

        assert benchmark_suite.main() == 0
        assert called == {
            "results_dir": tmp_path,
            "plots": ["time"],
            "require_complete_suite": True,
            "expected_backends": {"torch", "jax"},
        }

    def test_run_dir_writes_results_without_nested_timestamp(
        self, monkeypatch, tmp_path
    ):
        """``--run-dir`` writes directly into a caller-owned result directory."""
        run_dir = tmp_path / "merged-results"
        calls = {}

        def fake_parse_args():
            return SimpleNamespace(
                benchmark=["nl"],
                backend="jax",
                plot_only=None,
                plots=["time"],
                system=None,
                mode=None,
                output_dir=tmp_path / "unused-base",
                run_dir=run_dir,
                timing_runs=None,
                warmup_runs=None,
                methods=None,
                dry_run=False,
                max_total_atoms=None,
                no_plot=True,
                cutoffs=None,
                accuracies=None,
            )

        fake_runner = SimpleNamespace(
            dry_run_from_config=lambda _config: [{}],
            run_from_config=lambda _config, output_dir=None: (
                calls.setdefault("output_dir", Path(output_dir)) and [{"success": True}]
            ),
        )

        monkeypatch.setattr(benchmark_suite, "parse_args", fake_parse_args)
        monkeypatch.setattr(
            benchmark_suite,
            "load_yaml_config",
            lambda _path: {
                "parameters": {},
                "runtime": {},
                "systems": {},
                "scaling": {},
                "methods": [],
                "output": {"base_dir": str(tmp_path)},
            },
        )
        monkeypatch.setattr(
            benchmark_suite.importlib,
            "import_module",
            lambda _name: fake_runner,
        )

        assert benchmark_suite.main() == 0
        assert calls["output_dir"] == run_dir
        assert run_dir.is_dir()
        assert not list((tmp_path / "unused-base").glob("run_*"))
        assert "- **Backend**: jax" in (run_dir / "RUN_LOG.md").read_text(
            encoding="utf-8"
        )

    def test_runtime_report_rejects_missing_planned_case(self, tmp_path, capsys):
        """A partial row set cannot pass merely because one case succeeded."""
        planned = [
            {
                "benchmark": "nl",
                "backend": "jax",
                "system": "cscl",
                "mode": "system_size",
                "method": method,
                "atoms_per_system": 128,
                "batch_size": 1,
                "total_atoms": 128,
                "cutoff": 6.0,
            }
            for method in ("naive_scalar", "naive_tile")
        ]
        emitted = [
            {
                **planned[0],
                "scaling_mode": planned[0]["mode"],
                "success": True,
            }
        ]
        summary = _SuiteResults()
        summary.record("NL", emitted, plan_only=False)
        summary.validate_coverage("NL", planned, emitted)

        assert _report_run(summary, tmp_path, plot_ok=True) == 1
        assert "naive_tile" in capsys.readouterr().err

    def test_runtime_coverage_matches_el_derivative_workload(self):
        """EL coverage keys include the single derivative workload metadata."""
        planned = {
            "benchmark": "el",
            "backend": "jax",
            "system": "cscl",
            "mode": "system_size",
            "method": "pme",
            "atoms_per_system": 128,
            "batch_size": 1,
            "total_atoms": 128,
            "accuracy": 1e-4,
            "derivative_contract": "energy_autograd",
            "workload": "energy_forces_charge_gradients",
            "compute_forces": True,
            "compute_charge_gradients": True,
        }
        emitted = {
            "backend": "jax",
            "system": "cscl",
            "scaling_mode": "system_size",
            "method": "pme",
            "atoms_per_system": 128,
            "batch_size": 1,
            "total_atoms": 128,
            "accuracy": 1e-4,
            "derivative_contract": "energy_autograd",
            "workload": "energy_forces_charge_gradients",
            "compute_forces": True,
            "compute_charge_gradients": True,
            "success": True,
        }
        summary = _SuiteResults()

        summary.validate_coverage("EL", [planned], [emitted])

        assert summary.coverage_errors == {}

    def test_runtime_coverage_rejects_el_workload_drift(self):
        """An energy-only EL row cannot satisfy the derivative benchmark plan."""
        planned = {
            "benchmark": "el",
            "backend": "torch",
            "system": "cscl",
            "mode": "system_size",
            "method": "ewald",
            "atoms_per_system": 128,
            "batch_size": 1,
            "total_atoms": 128,
            "accuracy": 1e-4,
            "derivative_contract": "energy_autograd",
            "workload": "energy_forces_charge_gradients",
            "compute_forces": True,
            "compute_charge_gradients": True,
        }
        emitted = {
            **planned,
            "scaling_mode": planned["mode"],
            "workload": "energy_only",
            "compute_forces": False,
            "compute_charge_gradients": False,
        }
        summary = _SuiteResults()

        summary.validate_coverage("EL", [planned], [emitted])

        assert "EL" in summary.coverage_errors

    def test_count_mode_uses_dry_plan_without_row_listing(self, monkeypatch, capsys):
        """``--count`` prints row counts through the no-allocation planning path."""

        def fake_parse_args():
            return SimpleNamespace(
                benchmark=["nl"],
                backend="torch",
                plot_only=None,
                plots=["time"],
                system=None,
                mode=None,
                output_dir=None,
                run_dir=None,
                timing_runs=None,
                warmup_runs=None,
                methods=None,
                dry_run=False,
                list_plan=False,
                count_plan=True,
                max_total_atoms=None,
                no_plot=True,
                cutoffs=None,
                accuracies=None,
                d3_params_path=None,
            )

        calls = {}

        def fake_dry_run(config):
            calls["plan_output"] = config["runtime"]["plan_output"]
            return [{"benchmark": "nl"}, {"benchmark": "nl"}]

        fake_runner = SimpleNamespace(
            dry_run_from_config=fake_dry_run,
            run_from_config=lambda *_args, **_kwargs: pytest.fail(
                "count mode must not run benchmarks"
            ),
        )

        monkeypatch.setattr(benchmark_suite, "parse_args", fake_parse_args)
        monkeypatch.setattr(
            benchmark_suite.importlib,
            "import_module",
            lambda _module_name: fake_runner,
        )

        assert benchmark_suite.main() == 0
        out = capsys.readouterr().out
        assert calls == {"plan_output": "count"}
        assert "COUNT COMPLETE: 2 planned row(s)" in out

    def test_generate_plots_fails_when_all_attempts_fail(self, monkeypatch, tmp_path):
        """A plot pass with CSV input returns False when every plot fails."""
        csv_path = tmp_path / "nl-cscl-system-size-scaling.csv"
        csv_path.write_text(
            "success,backend,method,total_atoms\nTrue,torch,cell_list,2\n"
        )

        def fail_plot(*_args, **_kwargs):
            raise RuntimeError("plot boom")

        monkeypatch.setattr(plot_benchmarks, "detect_and_plot", fail_plot)
        monkeypatch.setattr(plot_benchmarks, "plot_single_panel", fail_plot)

        assert benchmark_suite._generate_plots(tmp_path) is False

    def test_generate_plots_honors_panel_filter(self, monkeypatch, tmp_path):
        """``plots=['time']`` skips 3-panel plots and renders only time panels."""
        csv_path = tmp_path / "nl-cscl-system-size-scaling.csv"
        csv_path.write_text(
            "success,backend,method,total_atoms\nTrue,torch,cell_list,2\n"
        )
        panels = []

        def fail_three_panel(*_args, **_kwargs):
            pytest.fail(
                "filtered single-panel plotting should not render 3-panel plots"
            )

        def record_single_panel(_csv_path, panel, output_path, **kwargs):
            Path(output_path).write_text("png", encoding="utf-8")
            panels.append((panel, kwargs.get("filters")))
            return True

        monkeypatch.setattr(plot_benchmarks, "detect_and_plot", fail_three_panel)
        monkeypatch.setattr(plot_benchmarks, "plot_single_panel", record_single_panel)

        assert benchmark_suite._generate_plots(tmp_path, plots=["time"]) is True
        assert panels == [
            ("time", None),
            ("time", {"cutoff": 6.0}),
            ("time", {"cutoff": 15.0}),
            ("time", {"cutoff": 25.0}),
        ]

    def test_generate_plots_ignores_legacy_non_suite_csvs(self, monkeypatch, tmp_path):
        """Suite plot-only skips legacy or unrelated CSVs in docs result dirs."""
        suite_csv = tmp_path / "nl-cscl-system-size-scaling.csv"
        suite_csv.write_text(
            "success,backend,method,total_atoms,cutoff\nTrue,torch,cell_list,2,6.0\n",
            encoding="utf-8",
        )
        legacy_csv = tmp_path / "dftd3_benchmark_torch_h100-80gb-hbm3.csv"
        legacy_csv.write_text(
            "success,backend,method,total_atoms\nTrue,torch,dftd3,2\n",
            encoding="utf-8",
        )
        seen = []

        def record_single_panel(csv_path, panel, output_path, **kwargs):
            seen.append(Path(csv_path).name)
            Path(output_path).write_text("png", encoding="utf-8")
            return True

        monkeypatch.setattr(plot_benchmarks, "plot_single_panel", record_single_panel)

        assert benchmark_suite._generate_plots(tmp_path, plots=["time"]) is True
        assert seen == [
            "nl-cscl-system-size-scaling.csv",
            "nl-cscl-system-size-scaling.csv",
            "nl-cscl-system-size-scaling.csv",
            "nl-cscl-system-size-scaling.csv",
        ]

    def test_plot_only_rejects_incomplete_provenanced_suite(self, tmp_path):
        """Final reportable plotting is also the 18-file completeness gate."""
        (tmp_path / ".benchmark-run-id").write_text(
            "12345678123456781234567812345678\n", encoding="ascii"
        )
        (tmp_path / "nl-cscl-system-size-scaling.csv").write_text(
            "success\nTrue\n", encoding="utf-8"
        )

        assert (
            benchmark_suite._generate_plots(
                tmp_path,
                plots=["time"],
                require_complete_suite=True,
            )
            is False
        )

    def test_plot_only_rejects_empty_unmarked_suite(self, tmp_path):
        """Complete-suite mode cannot treat an empty directory as success."""
        assert (
            benchmark_suite._generate_plots(
                tmp_path,
                plots=["time"],
                require_complete_suite=True,
            )
            is False
        )

    def test_plot_only_rejects_incomplete_unmarked_suite(self, tmp_path):
        """The completeness gate does not depend on a run-ID marker."""
        (tmp_path / "nl-cscl-system-size-scaling.csv").write_text(
            "success\nTrue\n", encoding="utf-8"
        )

        assert (
            benchmark_suite._generate_plots(
                tmp_path,
                plots=["time"],
                require_complete_suite=True,
            )
            is False
        )

    def test_plot_only_validates_complete_case_matrix(self, monkeypatch, tmp_path):
        """A complete filename set is matrix-checked before publication."""
        for name in benchmark_suite.REPORTABLE_CSV_NAMES:
            (tmp_path / name).write_text("success\nTrue\n", encoding="utf-8")
        checked = []
        monkeypatch.setattr(
            "benchmarks.suite_orchestration_plots.validate_result_files",
            lambda paths, expected_run_id=None: None,
        )
        monkeypatch.setattr(
            benchmark_suite,
            "_validate_complete_reportable_suite",
            lambda paths, backends: checked.append(
                (
                    {Path(path).name for path in paths},
                    set(backends),
                )
            ),
        )
        monkeypatch.setattr(
            plot_benchmarks,
            "plot_single_panel",
            lambda *_args, **_kwargs: True,
        )

        assert benchmark_suite._generate_plots(
            tmp_path,
            plots=["time"],
            require_complete_suite=True,
            expected_backends={"torch", "jax"},
        )
        assert checked == [(benchmark_suite.REPORTABLE_CSV_NAMES, {"torch", "jax"})]

    def test_plot_only_fails_when_any_expected_render_fails(
        self, monkeypatch, tmp_path
    ):
        """Publication cannot succeed with a stale or missing plot panel."""
        for name in benchmark_suite.REPORTABLE_CSV_NAMES:
            (tmp_path / name).write_text("success\nTrue\n", encoding="utf-8")
        monkeypatch.setattr(
            "benchmarks.suite_orchestration_plots.validate_result_files",
            lambda paths, expected_run_id=None: None,
        )
        monkeypatch.setattr(
            benchmark_suite,
            "_validate_complete_reportable_suite",
            lambda paths, backends: None,
        )

        def render(_csv_path, _panel, output_path, **_kwargs):
            return Path(output_path).name != "d3-cscl-system-size-scaling-time.png"

        monkeypatch.setattr(plot_benchmarks, "plot_single_panel", render)

        assert not benchmark_suite._generate_plots(
            tmp_path,
            plots=["time"],
            require_complete_suite=True,
            expected_backends={"torch", "jax", "warp"},
        )

    def test_automatic_plotting_allows_selected_suite_outputs(
        self, monkeypatch, tmp_path
    ):
        """A successful selected run is not mistaken for a full reportable suite."""
        calls = []

        def record_orchestration(_results_dir, **kwargs):
            calls.append(
                (
                    kwargs["expected_csv_names"],
                    kwargs["require_all_plots"],
                )
            )
            return True

        monkeypatch.setattr(
            benchmark_suite,
            "_orchestrate_plots",
            record_orchestration,
        )

        assert benchmark_suite._generate_plots(tmp_path, plots=["time"]) is True
        assert calls == [(None, False)]

    def test_docs_backend_comparison_omits_noncomparable_pair_centric(
        self, monkeypatch, tmp_path
    ):
        """Metadata-less NL comparisons retain only the fair scalar family."""
        csv_path = tmp_path / "nl-cscl-system-size-scaling.csv"
        rows = [
            "success,system,backend,method,cutoff,total_atoms,batch_size,time_us_per_atom,throughput_atoms_per_sec",
        ]
        for method in (
            "naive_scalar",
            "naive_tile",
            "cell_list_atom_centric",
            "cell_list_pair_centric",
        ):
            rows.extend(
                f"True,cscl,{backend},{method},15,128,1,1.0,128000000.0"
                for backend in ("torch", "jax")
            )
        csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

        batch_csv_path = tmp_path / "nl-cscl-batch-scaling.csv"
        batch_rows = [
            "success,system,backend,method,cutoff,total_atoms,batch_size,time_us_per_atom,throughput_atoms_per_sec",
        ]
        for method in (
            "batch_naive_scalar",
            "batch_naive_tile",
            "batch_cell_list_atom_centric",
            "batch_cell_list_pair_centric",
        ):
            batch_rows.extend(
                f"True,cscl,{backend},{method},15,128,4,1.0,128000000.0"
                for backend in ("torch", "jax")
            )
        batch_csv_path.write_text("\n".join(batch_rows) + "\n", encoding="utf-8")

        series_ids = []

        def record_line(_axis, _x, _y, **kwargs):
            series_ids.append(kwargs["gid"].lower())
            return True

        monkeypatch.setattr(plot_benchmarks, "_plot_data_line", record_line)
        monkeypatch.setattr(plot_benchmarks, "_savefig_atomic", lambda *_args: None)

        docs_generate_plots.generate_nl_backend_comparison_plots(tmp_path, tmp_path)

        assert any("naive-scalar" in series_id for series_id in series_ids)
        assert not any("naive-tile" in series_id for series_id in series_ids)
        assert not any(
            "cell-list-atom-centric" in series_id for series_id in series_ids
        )
        assert not any(
            "cell-list-pair-centric" in series_id for series_id in series_ids
        )

    def test_docs_backend_comparison_requires_matched_x_values(
        self, monkeypatch, tmp_path
    ):
        """Torch-vs-JAX docs panels only plot x-points where both backends succeeded."""
        csv_path = tmp_path / "nl-cscl-system-size-scaling.csv"
        csv_path.write_text(
            "\n".join(
                [
                    "success,system,backend,method,cutoff,total_atoms,batch_size,time_us_per_atom,throughput_atoms_per_sec,backend_comparable,timing_scope",
                    "True,cscl,torch,naive_scalar,15,128,1,1.0,128000000.0,True,backend_comparison",
                    "True,cscl,torch,naive_scalar,15,256,1,1.0,256000000.0,True,backend_comparison",
                    "True,cscl,jax,naive_scalar,15,128,1,2.0,64000000.0,True,backend_comparison",
                    "True,cscl,jax,naive_scalar,15,512,1,3.0,42666666.0,True,backend_comparison",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        plotted = {}

        def record_line(_axis, x, _y, **kwargs):
            plotted[kwargs["gid"]] = list(x)
            return True

        monkeypatch.setattr(plot_benchmarks, "_plot_data_line", record_line)
        monkeypatch.setattr(plot_benchmarks, "_savefig_atomic", lambda *_args: None)

        docs_generate_plots.generate_nl_backend_comparison_plots(tmp_path, tmp_path)

        assert set(plotted) == {
            "nl-cutoff-15A-naive-scalar-jax-single-time",
            "nl-cutoff-15A-naive-scalar-torch-single-time",
            "nl-cutoff-15A-naive-scalar-jax-single-throughput",
            "nl-cutoff-15A-naive-scalar-torch-single-throughput",
        }
        assert all(x_values == [128.0] for x_values in plotted.values())

    def test_nl_method_metadata_marks_backend_comparable_methods(self):
        """NL rows label comparison scope explicitly in the CSV schema."""
        comparable = benchmark_neighborlist._nl_method_metadata("naive_scalar")
        eager = benchmark_neighborlist._nl_method_metadata("naive_tile")
        cell = benchmark_neighborlist._nl_method_metadata("cell_list_atom_centric")
        batch_cell = benchmark_neighborlist._nl_method_metadata(
            "batch_cell_list_atom_centric"
        )
        pair = benchmark_neighborlist._nl_method_metadata("cell_list_pair_centric")
        cluster = benchmark_neighborlist._nl_method_metadata("batch_cluster_tile")

        assert comparable == {
            "backend_comparable": True,
            "timing_scope": "backend_comparison",
        }
        assert eager == {
            "backend_comparable": False,
            "timing_scope": "coverage_only_eager_jax",
        }
        assert cell == {
            "backend_comparable": False,
            "timing_scope": "coverage_only_backend_specific",
        }
        assert batch_cell == {
            "backend_comparable": False,
            "timing_scope": "coverage_only_backend_specific",
        }
        assert pair == {
            "backend_comparable": False,
            "timing_scope": "coverage_only_pair_centric",
        }
        assert cluster == {
            "backend_comparable": False,
            "timing_scope": "coverage_only_cluster_tile",
        }

    def test_docs_backend_comparison_ignores_legacy_when_unified_csv_is_torch_only(
        self, monkeypatch, tmp_path
    ):
        """Legacy backend CSVs must not mask missing current JAX rows."""
        unified = tmp_path / "nl-cscl-system-size-scaling.csv"
        unified.write_text(
            "\n".join(
                [
                    "success,system,backend,method,cutoff,total_atoms,batch_size,time_us_per_atom,throughput_atoms_per_sec",
                    "True,cscl,torch,naive_scalar,15,128,1,1.0,128000000.0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        legacy = tmp_path / "nl-backend-cscl-system-size-scaling.csv"
        legacy.write_text(
            "\n".join(
                [
                    "success,system,backend,method,cutoff,total_atoms,batch_size,time_us_per_atom,throughput_atoms_per_sec",
                    "True,cscl,torch,naive,15,128,1,1.0,128000000.0",
                    "True,cscl,jax,naive,15,128,1,1.2,106000000.0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        inputs = []

        def record_comparison(csv_path, *_args, **_kwargs):
            inputs.append(Path(csv_path).name)
            return False

        monkeypatch.setattr(
            plot_benchmarks,
            "plot_comparison_panel",
            record_comparison,
        )
        monkeypatch.setattr(
            docs_generate_plots,
            "_write_no_data_placeholder",
            lambda *_args, **_kwargs: None,
        )

        docs_generate_plots.generate_nl_backend_comparison_plots(tmp_path, tmp_path)

        assert inputs == [unified.name] * 4

    def test_docs_cutoff_selector_uses_layered_svg_without_combination_images(
        self, monkeypatch, tmp_path
    ):
        """Docs emit one layered SVG per panel instead of pre-rendered combinations."""
        csv_path = tmp_path / "nl-cscl-constant-workload-scaling.csv"
        csv_path.write_text(
            "\n".join(
                [
                    "success,system,backend,method,cutoff,total_atoms,batch_size,time_us_per_atom,throughput_atoms_per_sec,error_type",
                    "True,cscl,torch,batch_cell_list_atom_centric,25,131072,1,1.0,131072000.0,",
                    "True,cscl,jax,batch_cell_list_atom_centric,25,131072,1,1.2,109226666.7,",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        rendered = []

        def record_single_panel(_csv, panel, output_path, **kwargs):
            rendered.append((panel, Path(output_path).name, kwargs.get("filters")))
            Path(output_path).write_text("image", encoding="utf-8")
            return True

        monkeypatch.setattr(plot_benchmarks, "plot_single_panel", record_single_panel)
        monkeypatch.setattr(
            plot_benchmarks,
            "plot_comparison_panel",
            lambda *_args, **_kwargs: None,
        )

        docs_generate_plots.generate_suite_csv_plots(tmp_path, tmp_path)

        output_names = {name for _, name, _ in rendered}
        assert "nl-cscl-constant-workload-scaling-time.svg" in output_names
        assert "nl-cscl-constant-workload-scaling-jax-time.svg" in output_names
        assert "nl-cscl-constant-workload-scaling-memory.svg" in output_names
        assert not any("-cutoff-" in name for name in output_names)

    def test_docs_d3_el_comparison_generates_memory_panels(self, monkeypatch, tmp_path):
        """D3/EL comparison plots refresh memory panels as well as timing panels."""
        for name in (
            "d3-cscl-system-size-scaling.csv",
            "el-cscl-system-size-scaling.csv",
        ):
            (tmp_path / name).write_text(
                "\n".join(
                    [
                        "success,system,scaling_mode,backend,method,total_atoms,batch_size,cutoff,accuracy,time_us_per_atom,throughput_atoms_per_sec,mem_peak_mb,backend_comparable,timing_scope",
                        "True,cscl,system_size,torch,pme,128,1,15,1e-6,1.0,128000000.0,10.0,True,backend_comparison",
                        "True,cscl,system_size,jax,pme,128,1,15,1e-6,2.0,64000000.0,nan,True,backend_comparison",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

        comparisons = []
        single_panels = []

        def record_comparison(csv_path, panel, output_path, module):
            comparisons.append(
                (Path(csv_path).name, panel, Path(output_path).name, module)
            )

        monkeypatch.setattr(
            plot_benchmarks,
            "plot_single_panel",
            lambda csv_path, panel, output_path, **kwargs: (
                single_panels.append(
                    (
                        Path(csv_path).name,
                        panel,
                        Path(output_path).name,
                        kwargs.get("filters"),
                    )
                )
                or Path(output_path).write_text("png", encoding="utf-8")
                or True
            ),
        )
        monkeypatch.setattr(plot_benchmarks, "plot_comparison_panel", record_comparison)

        docs_generate_plots.generate_suite_csv_plots(tmp_path, tmp_path)

        assert (
            "d3-cscl-system-size-scaling.csv",
            "memory",
            "d3-cscl-system-size-comparison-memory.png",
            "d3",
        ) in comparisons
        assert (
            "el-cscl-system-size-scaling.csv",
            "memory",
            "el-cscl-system-size-comparison-memory.png",
            "el",
        ) in comparisons
        assert (
            "d3-cscl-system-size-scaling.csv",
            "time",
            "d3-cscl-system-size-scaling-jax-time.png",
            {"backend": "jax"},
        ) in single_panels
        assert (
            "el-cscl-system-size-scaling.csv",
            "throughput",
            "el-cscl-system-size-scaling-jax-throughput.png",
            {"backend": "jax"},
        ) in single_panels

    def test_comparison_plotter_requires_matched_x_values(self):
        """Shared comparison panels drop rows missing on either backend."""
        grouped = {
            ("naive_scalar", "torch", None): [
                {"total_atoms": 128, "backend": "torch"},
                {"total_atoms": 256, "backend": "torch"},
            ],
            ("naive_scalar", "jax", None): [
                {"total_atoms": 128, "backend": "jax"},
                {"total_atoms": 512, "backend": "jax"},
            ],
            ("cell_list_atom_centric", "torch", None): [
                {"total_atoms": 128, "backend": "torch"},
            ],
        }

        filtered = plot_benchmarks._filter_grouped_to_matched_backend_x(
            grouped,
            "total_atoms",
        )

        assert set(filtered) == {
            ("naive_scalar", "torch", None),
            ("naive_scalar", "jax", None),
        }
        assert [
            row["total_atoms"] for row in filtered[("naive_scalar", "torch", None)]
        ] == [128]
        assert [
            row["total_atoms"] for row in filtered[("naive_scalar", "jax", None)]
        ] == [128]

    def test_generate_plots_fails_when_single_panel_has_no_data(self, tmp_path):
        """All-failed CSVs are not counted as successfully rendered plots."""
        csv_path = tmp_path / "nl-cscl-system-size-scaling.csv"
        csv_path.write_text(
            "success,backend,method,total_atoms\nFalse,torch,cell_list,2\n"
        )

        assert benchmark_suite._generate_plots(tmp_path, plots=["time"]) is False

    def test_three_panel_detect_returns_false_when_csv_has_no_data(self, tmp_path):
        """The 3-panel plotter reports empty/all-failed CSVs to orchestration."""
        csv_path = tmp_path / "nl-cscl-system-size-scaling.csv"
        csv_path.write_text(
            "success,backend,method,total_atoms\nFalse,torch,cell_list,2\n",
            encoding="utf-8",
        )

        assert plot_benchmarks.detect_and_plot(csv_path, tmp_path) is False

    def test_suite_detects_yaml_selected_jax_backend(self, monkeypatch, tmp_path):
        """Suite-level JAX env setup honors YAML runtime backend, not just CLI."""
        config_path = tmp_path / "benchmark_config.yaml"
        config_path.write_text("runtime:\n  backend: jax\n", encoding="utf-8")
        monkeypatch.setitem(
            benchmark_suite.RUNNERS,
            "nl",
            {
                "label": "NL",
                "config": config_path,
                "module": "benchmarks.neighborlist.benchmark_neighborlist",
            },
        )
        args = SimpleNamespace(backend=None)

        assert benchmark_suite._suite_needs_jax_env(args, {"nl"}) is True

    def test_dry_run_with_no_planned_rows_fails(self, monkeypatch):
        """Dry-run exits nonzero when CLI filters produce an empty plan."""

        def fake_parse_args():
            return SimpleNamespace(
                benchmark=["el"],
                backend="torch",
                plot_only=None,
                plots=None,
                system=["cscl"],
                mode=["system_size"],
                output_dir=None,
                timing_runs=None,
                warmup_runs=None,
                methods=["cell_list"],
                dry_run=True,
                max_total_atoms=512,
                no_plot=True,
                accuracies=None,
            )

        monkeypatch.setattr(benchmark_suite, "parse_args", fake_parse_args)

        assert benchmark_suite.main() == 1

    def test_non_dry_run_with_no_successful_rows_fails(self, monkeypatch, tmp_path):
        """Runtime exits nonzero when every emitted row is failed or skipped."""

        def fake_parse_args():
            return SimpleNamespace(
                benchmark=["nl"],
                backend=None,
                plot_only=None,
                plots=["time"],
                system=None,
                mode=None,
                output_dir=tmp_path,
                timing_runs=None,
                warmup_runs=None,
                methods=None,
                dry_run=False,
                max_total_atoms=None,
                no_plot=True,
                cutoffs=None,
                accuracies=None,
            )

        fake_runner = SimpleNamespace(
            dry_run_from_config=lambda _config: [{}],
            run_from_config=lambda _config, output_dir=None: [
                {"success": False, "error_type": "RuntimeError"}
            ],
        )

        monkeypatch.setattr(benchmark_suite, "parse_args", fake_parse_args)
        monkeypatch.setattr(
            benchmark_suite,
            "load_yaml_config",
            lambda _path: {
                "parameters": {},
                "runtime": {},
                "systems": {},
                "scaling": {},
                "methods": [],
                "output": {"base_dir": str(tmp_path)},
            },
        )
        monkeypatch.setattr(
            benchmark_suite.importlib,
            "import_module",
            lambda _name: fake_runner,
        )

        assert benchmark_suite.main() == 1

    def test_non_dry_run_fails_when_one_requested_module_has_no_success(
        self, monkeypatch, tmp_path
    ):
        """A failing requested module is not masked by another module's success."""

        def fake_parse_args():
            return SimpleNamespace(
                benchmark=["nl", "el"],
                backend=None,
                plot_only=None,
                plots=["time"],
                system=None,
                mode=None,
                output_dir=tmp_path,
                timing_runs=None,
                warmup_runs=None,
                methods=None,
                dry_run=False,
                max_total_atoms=None,
                no_plot=True,
                cutoffs=None,
                accuracies=None,
            )

        def fake_import(module_name):
            if module_name.endswith("benchmark_neighborlist"):
                return SimpleNamespace(
                    dry_run_from_config=lambda _config: [{}],
                    run_from_config=lambda _config, output_dir=None: [
                        {"success": True}
                    ],
                )
            return SimpleNamespace(
                dry_run_from_config=lambda _config: [{}],
                run_from_config=lambda _config, output_dir=None: [
                    {"success": False, "error_type": "RuntimeError"}
                ],
            )

        monkeypatch.setattr(benchmark_suite, "parse_args", fake_parse_args)
        monkeypatch.setattr(
            benchmark_suite,
            "load_yaml_config",
            lambda _path: {
                "parameters": {},
                "runtime": {},
                "systems": {},
                "scaling": {},
                "methods": [],
                "output": {"base_dir": str(tmp_path)},
            },
        )
        monkeypatch.setattr(benchmark_suite.importlib, "import_module", fake_import)

        assert benchmark_suite.main() == 1


class TestBenchmarkAtomPlanning:
    """Test allocation-free atom count planning."""

    def test_nh3_provenance_tracks_only_runtime_pdb_inputs(self, tmp_path):
        """Generated Packmol records do not make provenance path-sensitive."""
        config = {
            "systems": {
                "nh3": {
                    "enabled": True,
                    "pdb_dir": str(tmp_path),
                    "atom_counts": [128, 256],
                    "constant_atoms_sizes": [256],
                }
            }
        }
        (tmp_path / "ammonia_pbc_128.pdb").write_text("pdb-128")
        (tmp_path / "ammonia_pbc_128.inp").write_text(
            f"output {tmp_path / 'ammonia_pbc_128.pdb'}"
        )

        artifacts = configured_nh3_artifacts(config)

        assert artifacts == {
            "nh3_pdb_128": tmp_path / "ammonia_pbc_128.pdb",
            "nh3_pdb_256": tmp_path / "ammonia_pbc_256.pdb",
        }
        assert all(path.suffix == ".pdb" for path in artifacts.values())

    def test_planned_atom_counts_for_cscl_supercell(self):
        """CsCl planning uses the rounded valid supercell atom count."""
        atoms_per_system, batch_size, total_atoms = planned_atom_counts(
            "cscl", {"num_atoms": 100, "batch_size": 4}
        )

        assert atoms_per_system == 128
        assert batch_size == 4
        assert total_atoms == 512

    def test_filter_configs_by_total_atoms_splits_before_allocation(self):
        """Configs over the atom cap are reported as skipped rows."""
        configs = [
            {"num_atoms": 100, "batch_size": 1},
            {"num_atoms": 100, "batch_size": 4},
        ]

        kept, skipped = filter_configs_by_total_atoms(configs, "cscl", 256)

        assert kept == [{"num_atoms": 100, "batch_size": 1}]
        assert skipped == [({"num_atoms": 100, "batch_size": 4}, 512)]

    def test_plan_only_nh3_configs_do_not_require_pdb_files(self, tmp_path):
        """Dry-run NH3 planning uses YAML sizes without generated PDB files."""
        configs = configs_for_mode(
            "system_size",
            {"enabled": True},
            "nh3",
            {"enabled": True, "atom_counts": [128, 256]},
            tmp_path / "missing-nh3",
            plan_only=True,
        )

        assert configs == [
            {"num_atoms": 128, "pdb_path": None, "batch_size": 1},
            {"num_atoms": 256, "pdb_path": None, "batch_size": 1},
        ]

    def test_actual_nh3_missing_pdbs_fall_back_to_planned_configs(self, tmp_path):
        """Actual runs keep row accounting when generated NH3 PDBs are absent."""
        configs = configs_for_mode(
            "system_size",
            {"enabled": True},
            "nh3",
            {"enabled": True, "atom_counts": [128]},
            tmp_path / "missing-nh3",
            plan_only=False,
        )

        assert configs == [{"num_atoms": 128, "pdb_path": None, "batch_size": 1}]

    def test_cscl_constant_workload_uses_yaml_atom_counts(self):
        """Constant-workload CsCl rows are driven by config, not baked-in grids."""
        configs = configs_for_mode(
            "constant_workload",
            {"enabled": True, "target_atoms": 1024},
            "cscl",
            {"enabled": True, "atom_counts": [100, 500]},
        )

        assert configs == [
            {"num_atoms": 100, "pdb_path": None, "batch_size": 8},
            {"num_atoms": 500, "pdb_path": None, "batch_size": 1},
        ]


class TestFailureRows:
    """Test failure row schema used by benchmark CSV output."""

    def test_build_failure_result_sets_success_and_error_fields(self):
        """Failures are written into main result rows with explicit metadata."""
        row = build_failure_result(
            error="boom",
            error_type="RuntimeError",
            benchmark="nl",
            backend="torch",
            system="cscl",
            scaling_mode="system_size",
            method="cell_list",
            atoms_per_system=128,
            batch_size=1,
            total_atoms=128,
            timing_runs=10,
            warmup_runs=3,
        )

        assert row["success"] is False
        assert row["error"] == "boom"
        assert row["error_type"] == "RuntimeError"
        assert row["method"] == "cell_list"
        assert row["timing_runs"] == 10
        assert row["warmup_runs"] == 3
        assert math.isnan(row["time_us_per_atom"])
        assert math.isnan(row["throughput_atoms_per_sec"])

    def test_success_result_uses_stable_error_columns(self):
        """Successful rows still carry empty error columns for append stability."""
        row = build_result(
            benchmark="nl",
            backend="torch",
            system="cscl",
            scaling_mode="system_size",
            method="cell_list",
            atoms_per_system=128,
            batch_size=1,
            total_atoms=128,
            time_seconds=1.0,
            timing_runs=10,
            warmup_runs=3,
            mem_info={"mem_delta_mb": 0.0, "mem_peak_gb": 0.0},
        )

        assert row["success"] is True
        assert row["error"] == ""
        assert row["error_type"] == ""
        assert row["timing_runs"] == 10
        assert row["warmup_runs"] == 3
        assert row["timing_method"] == "torch_cuda_events"
        assert row["compile_policy"] == "warmup_excluded"

    def test_jax_success_result_records_timing_contract(self):
        """JAX rows explicitly record wall-clock block-until-ready timing."""
        row = build_result(
            benchmark="nl",
            backend="jax",
            system="cscl",
            scaling_mode="system_size",
            method="cell_list",
            atoms_per_system=128,
            batch_size=1,
            total_atoms=128,
            time_seconds=1.0,
            timing_runs=10,
            warmup_runs=3,
            mem_info={"mem_delta_mb": math.nan, "mem_peak_gb": math.nan},
        )

        assert row["timing_method"] == "jax_wall_block_until_ready"
        assert row["compile_policy"] == "warmup_excluded"

    def test_result_row_treats_none_timing_metadata_as_backend_default(self):
        """Accidental None values do not erase timing metadata in CSV rows."""
        row = build_result(
            benchmark="el",
            backend="torch",
            system="cscl",
            scaling_mode="system_size",
            method="pme",
            atoms_per_system=128,
            batch_size=1,
            total_atoms=128,
            time_seconds=1.0,
            timing_runs=10,
            warmup_runs=3,
            mem_info={"mem_delta_mb": 0.0, "mem_peak_gb": 0.0},
            timing_method=None,
            compile_policy=None,
        )

        assert row["timing_method"] == "torch_cuda_events"
        assert row["compile_policy"] == "warmup_excluded"

    def test_oom_failure_result_uses_concise_error_message(self):
        """OOM rows keep a stable error type without embedding backend tracebacks."""
        row = build_failure_result(
            error="RESOURCE_EXHAUSTED: Failed to allocate request for 19.20GiB",
            error_type="OutOfMemoryError",
            benchmark="nl",
            backend="jax",
            system="cscl",
            scaling_mode="system_size",
            method="cell_list",
            atoms_per_system=128,
            batch_size=1024,
            total_atoms=131072,
            timing_runs=10,
            warmup_runs=3,
        )

        assert row["error_type"] == "OutOfMemoryError"
        assert row["error"] == (
            "Out of memory during benchmark execution; see run logs for backend details."
        )

    def test_safe_launch_limit_is_an_unsupported_configuration(self):
        """Public API launch guards receive a stable benchmark failure class."""
        error = ValueError(
            "strategy='pair_centric' would require too many logical threads, "
            "exceeding the safe linear launch limit of 2147483647."
        )

        assert failure_error_type(error) == "UnsupportedConfiguration"

    def test_save_results_preserves_same_run_rows_when_schema_expands(self, tmp_path):
        """Appending a same-run failure row expands schema without clobbering data."""
        csv_path = tmp_path / "results.csv"
        first = build_result(
            benchmark="nl",
            backend="torch",
            system="cscl",
            scaling_mode="system_size",
            method="cell_list",
            atoms_per_system=128,
            batch_size=1,
            total_atoms=128,
            time_seconds=1.0,
            timing_runs=10,
            warmup_runs=3,
            mem_info={"mem_delta_mb": 0.0, "mem_peak_gb": 0.0},
        )
        save_results([first], csv_path)
        row = build_failure_result(
            error="boom",
            error_type="RuntimeError",
            failure_stage="timing",
            benchmark="nl",
            backend="torch",
            system="nh3",
            scaling_mode="system_size",
            method="cell_list",
            atoms_per_system=128,
            batch_size=1,
            total_atoms=128,
            timing_runs=10,
            warmup_runs=3,
        )

        save_results([row], csv_path, append=True)

        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert [row["system"] for row in rows] == ["cscl", "nh3"]
        assert rows[1]["error"] == "boom"
        assert rows[1]["failure_stage"] == "timing"

    def test_save_results_replaces_existing_rows_by_default(self, tmp_path):
        """Default writes avoid silently mixing old benchmark generations."""
        csv_path = tmp_path / "results.csv"
        csv_path.write_text("system,success\nold,True\n", encoding="utf-8")
        row = build_result(
            benchmark="nl",
            backend="torch",
            system="new",
            scaling_mode="system_size",
            method="cell_list",
            atoms_per_system=128,
            batch_size=1,
            total_atoms=128,
            time_seconds=1.0,
            timing_runs=10,
            warmup_runs=3,
            mem_info={"mem_delta_mb": 0.0, "mem_peak_gb": 0.0},
        )

        save_results([row], csv_path)

        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert [row["system"] for row in rows] == ["new"]

    def test_save_results_replaces_only_requested_backend(self, tmp_path):
        """Shared docs CSVs keep other backends while refreshing one backend."""
        csv_path = tmp_path / "results.csv"
        torch_old = build_result(
            benchmark="nl",
            backend="torch",
            system="old_torch",
            scaling_mode="system_size",
            method="cell_list",
            atoms_per_system=128,
            batch_size=1,
            total_atoms=128,
            time_seconds=1.0,
            timing_runs=10,
            warmup_runs=3,
            mem_info={"mem_delta_mb": 0.0, "mem_peak_gb": 0.0},
        )
        jax_row = build_result(
            benchmark="nl",
            backend="jax",
            system="jax",
            scaling_mode="system_size",
            method="cell_list",
            atoms_per_system=128,
            batch_size=1,
            total_atoms=128,
            time_seconds=1.0,
            timing_runs=10,
            warmup_runs=3,
            mem_info={"mem_delta_mb": 0.0, "mem_peak_gb": 0.0},
            timing_method="jax_wall_block_last",
        )
        torch_new = build_result(
            benchmark="nl",
            backend="torch",
            system="new_torch",
            scaling_mode="system_size",
            method="cell_list",
            atoms_per_system=256,
            batch_size=1,
            total_atoms=256,
            time_seconds=1.0,
            timing_runs=10,
            warmup_runs=3,
            mem_info={"mem_delta_mb": 0.0, "mem_peak_gb": 0.0},
        )

        save_results([torch_old], csv_path, replace_backend="torch")
        save_results([jax_row], csv_path, replace_backend="jax")
        save_results([torch_new], csv_path, replace_backend="torch")

        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert [(row["backend"], row["system"]) for row in rows] == [
            ("jax", "jax"),
            ("torch", "new_torch"),
        ]

    def test_run_log_describes_jax_timing_contract(self, tmp_path):
        """RUN_LOG.md does not describe JAX as CUDA-event timed."""
        start = benchmark_suite.datetime(2026, 1, 1, 0, 0, 0)

        write_run_log(tmp_path, start)

        text = (tmp_path / "RUN_LOG.md").read_text(encoding="utf-8")
        assert "Torch/Warp CUDA timing pattern" in text
        assert "JAX timing pattern" in text
        assert "block_until_ready(last)" in text
        assert "JAX/XLA Environment" in text
        assert "XLA_PYTHON_CLIENT_MEM_FRACTION" in text
        assert "Runtime Cache Environment" in text
        assert "XDG_CACHE_HOME" in text
        assert "PYTORCH_KERNEL_CACHE_PATH" in text
        assert "Reported timings exclude warm-up/compile/load iterations." in text
        assert "uv run python -m benchmarks.benchmark_suite --benchmark all" in text

    def test_build_skipped_result_sets_policy_error_type(self):
        """Policy skips use the same failure-row schema as runtime failures."""
        row = build_skipped_result(
            reason=">64 max_total_atoms",
            benchmark="nl",
            backend="jax",
            system="cscl",
            scaling_mode="system_size",
            method="cell_list",
            atoms_per_system=128,
            batch_size=1,
            total_atoms=128,
            cutoff=25.0,
            timing_runs=10,
            warmup_runs=3,
        )

        assert row["success"] is False
        assert row["error"] == ">64 max_total_atoms"
        assert row["error_type"] == "SkippedByPolicy"
        assert row["method"] == "cell_list"
        assert row["timing_runs"] == 10
        assert row["warmup_runs"] == 3

    def test_result_rows_require_timing_metadata(self):
        """Every benchmark row must carry timing run-count metadata."""
        with pytest.raises(ValueError, match="timing_runs is required"):
            build_result(
                benchmark="nl",
                backend="torch",
                system="cscl",
                scaling_mode="system_size",
                method="cell_list",
                atoms_per_system=128,
                batch_size=1,
                total_atoms=128,
                time_seconds=1.0,
                timing_runs=None,
                warmup_runs=3,
                mem_info={"mem_delta_mb": 0.0, "mem_peak_gb": 0.0},
            )


class TestDocsPlotParallelism:
    """Test worker-count resolution for docs benchmark plots."""

    def test_plot_workers_spawn_after_gallery_uses_jax(self):
        """Parallel plotting never forks a process with active JAX threads."""
        assert docs_generate_plots._PLOT_PROCESS_CONTEXT.get_start_method() == "spawn"

    def test_docs_and_suite_share_reportable_matrix(self):
        """Docs cannot drift from the suite's CSV names or NL cutoff set."""
        assert (
            docs_generate_plots._EXPECTED_SUITE_CSV_NAMES
            == benchmark_suite.REPORTABLE_CSV_NAMES
        )
        assert tuple(docs_generate_plots.NL_DOC_CUTOFFS) == tuple(
            benchmark_suite.NL_DOC_CUTOFFS
        )

    @pytest.mark.parametrize(
        ("jobs", "task_count", "expected"),
        [
            (1, 18, 1),
            ("4", 18, 4),
            (64, 18, 18),
        ],
    )
    def test_explicit_plot_jobs_are_bounded_by_work(self, jobs, task_count, expected):
        """Explicit worker requests never create more processes than tasks."""
        assert docs_generate_plots._resolve_plot_jobs(jobs, task_count) == expected

    def test_auto_plot_jobs_respects_process_affinity(self, monkeypatch):
        """Auto mode uses scheduler-visible CPUs and remains task-bounded."""
        monkeypatch.setattr(docs_generate_plots, "_available_cpu_count", lambda: 8)

        assert docs_generate_plots._resolve_plot_jobs("auto", 18) == 8
        assert docs_generate_plots._resolve_plot_jobs("auto", 3) == 3

    @pytest.mark.parametrize("jobs", [0, -1, "many"])
    def test_invalid_plot_jobs_are_rejected(self, jobs):
        """Invalid worker settings fail before starting any render processes."""
        with pytest.raises(ValueError, match="plot jobs"):
            docs_generate_plots._resolve_plot_jobs(jobs, 18)

    def test_partial_docs_override_is_rejected_without_bundled_fallback(
        self, monkeypatch, tmp_path
    ):
        """A partial fresh run cannot be mixed with bundled historical CSVs."""
        (tmp_path / "nl-cscl-system-size-scaling.csv").write_text(
            "success\nTrue\n", encoding="utf-8"
        )
        monkeypatch.setenv(docs_generate_plots.SUITE_RESULTS_ENV, str(tmp_path))

        with pytest.raises(RuntimeError, match="complete 18-file suite"):
            docs_generate_plots._suite_csv_dirs(tmp_path / "bundled")

    def test_complete_docs_override_is_the_only_csv_source(self, monkeypatch, tmp_path):
        """A complete override is provenance-checked and never backfilled."""
        for name in docs_generate_plots._EXPECTED_SUITE_CSV_NAMES:
            (tmp_path / name).write_text("success\nTrue\n", encoding="utf-8")
        checked = []
        matrix_checked = []

        def record_validation(paths, expected_run_id=None):
            checked.extend(Path(path).name for path in paths)
            assert expected_run_id is None

        monkeypatch.setattr(
            "benchmarks.suite_utils.validate_result_files", record_validation
        )
        monkeypatch.setattr(
            benchmark_suite,
            "validate_reportable_case_matrix",
            lambda paths, backends: matrix_checked.extend(
                Path(path).name for path in paths
            ),
        )
        monkeypatch.setenv(docs_generate_plots.SUITE_RESULTS_ENV, str(tmp_path))

        assert docs_generate_plots._suite_csv_dirs(tmp_path / "bundled") == [tmp_path]
        assert set(checked) == docs_generate_plots._EXPECTED_SUITE_CSV_NAMES
        assert set(matrix_checked) == docs_generate_plots._EXPECTED_SUITE_CSV_NAMES

    def test_bundled_docs_suite_is_also_complete_and_validated(
        self, monkeypatch, tmp_path
    ):
        """Bundled CSVs receive the same gate as an external override."""
        for name in docs_generate_plots._EXPECTED_SUITE_CSV_NAMES:
            (tmp_path / name).write_text("success\nTrue\n", encoding="utf-8")
        checked = []
        matrix_checked = []

        def record_validation(paths, expected_run_id=None):
            checked.extend(Path(path).name for path in paths)
            assert expected_run_id is None

        monkeypatch.delenv(docs_generate_plots.SUITE_RESULTS_ENV, raising=False)
        monkeypatch.setattr(
            "benchmarks.suite_utils.validate_result_files", record_validation
        )
        monkeypatch.setattr(
            benchmark_suite,
            "validate_reportable_case_matrix",
            lambda paths, backends: matrix_checked.extend(
                Path(path).name for path in paths
            ),
        )

        assert docs_generate_plots._suite_csv_dirs(tmp_path, require_complete=True) == [
            tmp_path
        ]
        assert set(checked) == docs_generate_plots._EXPECTED_SUITE_CSV_NAMES
        assert set(matrix_checked) == docs_generate_plots._EXPECTED_SUITE_CSV_NAMES


class TestDryRunSkipPlanning:
    """Test allocation-free planning of policy-skipped benchmark rows."""

    def test_nl_dry_run_expands_max_atom_skips_per_method_and_cutoff(self):
        """NL max-atom caps are visible for every planned method/cutoff row."""
        config = {
            "parameters": {
                "cutoffs": [6.0, 25.0],
                "max_total_atoms": 64,
            },
            "runtime": {},
            "systems": {"cscl": {"enabled": True, "atom_counts": [128]}},
            "scaling": {"system_size": {"enabled": True}},
            "methods": [
                {"name": "naive_neighbor_list", "enabled": True},
                {"name": "cell_list", "enabled": True},
            ],
        }

        rows = dry_run_nl(config, backend="jax")

        assert len(rows) == 4
        assert {row["method"] for row in rows} == {
            "naive_neighbor_list",
            "cell_list",
        }
        assert {row["cutoff"] for row in rows} == {6.0, 25.0}
        assert {row["reason"] for row in rows} == {">64 max_total_atoms"}

    def test_nl_dry_run_includes_cluster_tile_family(self):
        """Cluster-tile is a first-class planned NL method."""
        config = {
            "parameters": {
                "cutoffs": [15.0],
                "max_total_atoms": 1024,
            },
            "runtime": {},
            "systems": {"cscl": {"enabled": True, "atom_counts": [128]}},
            "scaling": {"system_size": {"enabled": True}},
            "methods": [
                {"name": "cell_list", "enabled": True},
                {"name": "cluster_tile", "enabled": True},
            ],
        }

        rows = dry_run_nl(config, backend="torch")

        assert [row["method"] for row in rows] == ["cell_list", "cluster_tile"]

    def test_nl_warp_default_omits_unsupported_cluster_tile_family(self):
        """Default Warp dry-runs include only runnable Warp NL methods."""
        config = {
            "parameters": {
                "cutoffs": [15.0],
                "max_total_atoms": 1024,
            },
            "runtime": {},
            "systems": {"cscl": {"enabled": True, "atom_counts": [128]}},
            "scaling": {"system_size": {"enabled": True}},
            "methods": [
                {"name": "cell_list", "enabled": True},
                {"name": "cluster_tile", "enabled": True},
            ],
        }

        rows = dry_run_nl(config, backend="warp")

        assert [row["method"] for row in rows] == ["cell_list"]
        assert {row["reason"] for row in rows} == {""}

    def test_nl_jax_default_omits_pair_centric_but_runs_cluster_tile(self):
        """Default JAX dry-runs include supported cluster-tile coverage."""
        config = {
            "parameters": {
                "cutoffs": [15.0],
                "max_total_atoms": 1024,
            },
            "runtime": {},
            "systems": {"cscl": {"enabled": True, "atom_counts": [128]}},
            "scaling": {"system_size": {"enabled": True}},
            "methods": [
                {"name": "cell_list", "enabled": True},
                {"name": "cell_list_pair_centric", "enabled": True},
                {"name": "cluster_tile", "enabled": True},
            ],
        }

        rows = dry_run_nl(config, backend="jax")

        assert [row["method"] for row in rows] == ["cell_list", "cluster_tile"]
        assert {row["reason"] for row in rows} == {""}

    def test_nl_jax_explicit_pair_centric_is_planned(self):
        """Coverage-only JAX pair-centric remains available when requested."""
        config = {
            "parameters": {
                "cutoffs": [15.0],
                "max_total_atoms": 1024,
            },
            "runtime": {
                "explicit_methods": True,
                "selected_methods": ["cell_list_pair_centric"],
            },
            "systems": {"cscl": {"enabled": True, "atom_counts": [128]}},
            "scaling": {"system_size": {"enabled": True}},
            "methods": [{"name": "cell_list_pair_centric", "enabled": True}],
        }

        rows = dry_run_nl(config, backend="jax")

        assert [row["method"] for row in rows] == ["cell_list_pair_centric"]
        assert {row["reason"] for row in rows} == {""}

    def test_nl_jax_explicit_cluster_tile_is_planned_when_supported(self):
        """Generated periodic float32 JAX systems plan cluster-tile normally."""
        config = {
            "parameters": {
                "cutoffs": [15.0],
                "max_total_atoms": 1024,
            },
            "runtime": {
                "explicit_methods": True,
                "selected_methods": ["cluster_tile"],
            },
            "systems": {"cscl": {"enabled": True, "atom_counts": [128]}},
            "scaling": {"system_size": {"enabled": True}},
            "methods": [{"name": "cluster_tile", "enabled": True}],
        }

        rows = dry_run_nl(config, backend="jax")

        assert [row["method"] for row in rows] == ["cluster_tile"]
        assert {row["reason"] for row in rows} == {""}

    @pytest.mark.parametrize("backend", ["torch", "jax"])
    def test_nl_dry_run_batches_cluster_tile_for_default_methods(self, backend):
        """Default method expansion uses batch_cluster_tile for batched inputs."""
        config = {
            "parameters": {
                "cutoffs": [15.0],
                "max_total_atoms": 1024,
            },
            "runtime": {},
            "systems": {"cscl": {"enabled": True, "atom_counts": [128, 500]}},
            "scaling": {"constant_workload": {"enabled": True, "target_atoms": 1024}},
            "methods": [{"name": "cluster_tile", "enabled": True}],
        }

        rows = dry_run_nl(config, backend=backend)

        assert rows
        assert {row["method"] for row in rows} == {
            "cluster_tile",
            "batch_cluster_tile",
        }
        assert {row["reason"] for row in rows} == {""}

    def test_d3_dry_run_respects_method_filter(self):
        """D3 does not plan rows when CLI-selected methods exclude dftd3."""
        config = {
            "parameters": {
                "cutoffs": [15.0],
                "max_total_atoms": 256,
            },
            "runtime": {"explicit_methods": True, "selected_methods": ["pme"]},
            "systems": {"cscl": {"enabled": True, "atom_counts": [128]}},
            "scaling": {"system_size": {"enabled": True}},
            "methods": [{"name": "dftd3", "enabled": False}],
        }

        rows = dry_run_d3(config, backend="jax")

        assert rows == []

    def test_d3_dry_run_respects_disabled_yaml_method(self):
        """D3 YAML method switches are authoritative."""
        config = {
            "parameters": {
                "cutoffs": [15.0],
                "max_total_atoms": 256,
            },
            "systems": {"cscl": {"enabled": True, "atom_counts": [128]}},
            "scaling": {"system_size": {"enabled": True}},
            "methods": [{"name": "dftd3", "enabled": False}],
        }

        rows = dry_run_d3(config, backend="jax")

        assert rows == []

    def test_el_setup_failure_is_written_as_failure_row(self, monkeypatch, tmp_path):
        """EL setup failures emit explicit failure rows instead of disappearing."""
        config = {
            "parameters": {"timing_runs": 1, "warmup_runs": 1, "max_total_atoms": 1024},
            "systems": {"cscl": {"enabled": True, "atom_counts": [128]}},
            "scaling": {"system_size": {"enabled": True}},
            "methods": [{"name": "pme", "enabled": True, "spline_order": 5}],
            "accuracies": [1.0e-4],
            "output": {"base_dir": str(tmp_path)},
        }

        def fail_create_system(*_args, **_kwargs):
            raise TypeError("setup boom")

        monkeypatch.setattr(benchmark_electrostatics, "clean_gpu", lambda: None)
        monkeypatch.setattr(
            benchmark_electrostatics,
            "create_system",
            fail_create_system,
        )

        rows = benchmark_electrostatics.run_from_config(
            config,
            output_dir=tmp_path,
            backend="torch",
        )

        assert len(rows) == 1
        assert rows[0]["success"] is False
        assert rows[0]["error"] == "setup boom"
        assert rows[0]["error_type"] == "TypeError"
        assert rows[0]["method"] == "pme"
        assert rows[0]["timing_runs"] == 1
        assert rows[0]["warmup_runs"] == 1
        assert rows[0]["derivative_contract"] == "energy_autograd"
        assert rows[0]["workload"] == "energy_forces_charge_gradients"
        assert rows[0]["compute_forces"] is True
        assert rows[0]["compute_charge_gradients"] is True

    def test_el_dry_run_has_one_derivative_workload_per_method(self):
        """EL planning does not duplicate energy-plus-forces-only variants."""
        config = {
            "parameters": {"timing_runs": 1, "warmup_runs": 1},
            "runtime": {"plan_output": "count"},
            "systems": {"cscl": {"enabled": True, "atom_counts": [128]}},
            "scaling": {"system_size": {"enabled": True}},
            "methods": [
                {"name": "pme", "enabled": True, "spline_order": 4},
                {"name": "ewald", "enabled": True},
            ],
            "accuracies": [1.0e-4],
            "output": {"base_dir": "unused"},
        }

        rows = benchmark_electrostatics.dry_run_from_config(
            config,
            backend="torch",
        )

        assert [row["method"] for row in rows] == ["pme", "ewald"]
        assert {row["derivative_contract"] for row in rows} == {"energy_autograd"}
        assert {row["workload"] for row in rows} == {"energy_forces_charge_gradients"}
        assert all(row["compute_forces"] is True for row in rows)
        assert all(row["compute_charge_gradients"] is True for row in rows)

    def test_nl_setup_failure_is_written_as_failure_rows(self, monkeypatch, tmp_path):
        """NL setup failures emit one row per planned method/cutoff."""
        config = {
            "parameters": {"timing_runs": 1, "warmup_runs": 1, "cutoffs": [6.0]},
            "runtime": {},
            "systems": {"cscl": {"enabled": True, "atom_counts": [128]}},
            "scaling": {"system_size": {"enabled": True}},
            "methods": [{"name": "cell_list", "enabled": True}],
            "output": {"base_dir": str(tmp_path)},
        }

        monkeypatch.setattr(benchmark_neighborlist, "clean_gpu", lambda: None)
        monkeypatch.setattr(
            benchmark_neighborlist,
            "create_system",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(TypeError("setup boom")),
        )

        rows = benchmark_neighborlist.run_from_config(
            config,
            output_dir=tmp_path,
            backend="torch",
        )

        assert len(rows) == 1
        assert rows[0]["success"] is False
        assert rows[0]["error"] == "setup boom"
        assert rows[0]["error_type"] == "TypeError"
        assert rows[0]["failure_stage"] == "system_setup"

    def test_nl_warp_cluster_tile_is_policy_skipped_before_allocation(
        self, monkeypatch, tmp_path
    ):
        """Warp cluster-tile rows are explicit skips, not unsupported failures."""
        config = {
            "parameters": {"timing_runs": 1, "warmup_runs": 1, "cutoffs": [6.0]},
            "runtime": {},
            "systems": {"cscl": {"enabled": True, "atom_counts": [128]}},
            "scaling": {"system_size": {"enabled": True}},
            "methods": [{"name": "cluster_tile", "enabled": True}],
            "output": {"base_dir": str(tmp_path)},
        }

        def fail_create_system(*_args, **_kwargs):
            pytest.fail("policy-skipped cluster_tile should not allocate")

        monkeypatch.setattr(benchmark_neighborlist, "create_system", fail_create_system)

        rows = benchmark_neighborlist.run_from_config(
            config,
            output_dir=tmp_path,
            backend="warp",
        )

        assert len(rows) == 1
        assert rows[0]["success"] is False
        assert rows[0]["error_type"] == "SkippedByPolicy"
        assert rows[0]["error"] == "warp backend does not support cluster_tile"

    @pytest.mark.parametrize(
        ("dtype", "pbc", "platform", "expected_error"),
        [
            (
                "float64",
                [[True, True, True]],
                "gpu",
                "jax cluster_tile requires float32 positions",
            ),
            (
                "float32",
                [[True, False, True]],
                "gpu",
                "jax cluster_tile requires fully periodic pbc",
            ),
            (
                "float32",
                [[True, True, True]],
                "cpu",
                "jax cluster_tile requires a CUDA device",
            ),
        ],
    )
    def test_nl_jax_cluster_tile_unsupported_inputs_are_policy_skipped(
        self,
        monkeypatch,
        tmp_path,
        dtype,
        pbc,
        platform,
        expected_error,
    ):
        """Known unsupported JAX cluster-tile inputs remain policy rows."""
        config = {
            "parameters": {"timing_runs": 1, "warmup_runs": 0, "cutoffs": [6.0]},
            "runtime": {
                "explicit_methods": True,
                "selected_methods": ["cluster_tile"],
            },
            "systems": {"cscl": {"enabled": True, "atom_counts": [128]}},
            "scaling": {"system_size": {"enabled": True}},
            "methods": [{"name": "cluster_tile", "enabled": True}],
            "output": {"base_dir": str(tmp_path)},
        }
        data = {
            "positions": SimpleNamespace(
                dtype=dtype,
                devices=lambda: (SimpleNamespace(platform=platform),),
            ),
            "pbc": np.asarray(pbc, dtype=bool),
            "cell": np.eye(3, dtype=np.dtype(dtype))[None, :, :],
            "batch_idx": np.zeros(128, dtype=np.int32),
            "atoms_per_system": 128,
            "total_atoms": 128,
            "batch_size": 1,
            "cell_size": 100.0,
        }

        monkeypatch.setattr(benchmark_neighborlist, "lazy_import_jax", lambda: {})
        monkeypatch.setattr(benchmark_neighborlist, "clean_gpu", lambda: None)
        monkeypatch.setattr(
            benchmark_neighborlist, "current_alloc_gb", lambda _backend: 0.0
        )
        monkeypatch.setattr(
            benchmark_neighborlist,
            "create_system",
            lambda *_args, **_kwargs: data,
        )
        monkeypatch.setattr(
            benchmark_neighborlist,
            "_nl_run_one_method",
            lambda *_args, **_kwargs: pytest.fail(
                "unsupported cluster-tile data must not reach the benchmark"
            ),
        )

        rows = benchmark_neighborlist.run_from_config(
            config,
            output_dir=tmp_path,
            backend="jax",
        )

        assert len(rows) == 1
        assert rows[0]["success"] is False
        assert rows[0]["error_type"] == "SkippedByPolicy"
        assert rows[0]["error"] == expected_error

    def test_nl_jax_unexpected_method_failure_remains_failed(self, monkeypatch):
        """Unexpected JAX method errors are failures, not policy skips."""

        def fail_benchmark(*_args, **_kwargs):
            raise RuntimeError("kernel boom")

        monkeypatch.setattr(benchmark_neighborlist, "benchmark_nl", fail_benchmark)
        monkeypatch.setattr(benchmark_neighborlist, "clean_jax", lambda: None)

        row = benchmark_neighborlist._nl_run_one_method(
            data={},
            cutoff=6.0,
            method="cluster_tile",
            num_runs=1,
            warmup_runs=0,
            backend="jax",
            row_meta=benchmark_neighborlist.make_row_meta(
                "cscl",
                "system_size",
                "jax",
                128,
                1,
                128,
            ),
        )

        assert row["success"] is False
        assert row["error"] == "kernel boom"
        assert row["error_type"] == "RuntimeError"

    def test_nl_main_fails_when_selected_methods_only_emit_failures(
        self, monkeypatch, tmp_path
    ):
        """The standalone NL CLI returns nonzero for an all-failed selection."""
        args = SimpleNamespace(
            config=tmp_path / "benchmark_config.yaml",
            backend="jax",
            output_dir=tmp_path,
            dry_run=False,
            list_plan=False,
            count_plan=False,
        )
        config = {
            "runtime": {
                "explicit_methods": True,
                "selected_methods": ["cluster_tile"],
            }
        }

        monkeypatch.setattr(benchmark_neighborlist, "parse_args", lambda: args)
        monkeypatch.setattr(
            benchmark_neighborlist, "load_yaml_config", lambda _path: config
        )
        monkeypatch.setattr(
            benchmark_neighborlist,
            "merge_cli_overrides",
            lambda loaded, _args: loaded,
        )
        monkeypatch.setattr(
            benchmark_neighborlist, "ensure_jax_available", lambda: None
        )
        monkeypatch.setattr(
            benchmark_neighborlist,
            "run_from_config",
            lambda *_args, **_kwargs: [
                {
                    "method": "cluster_tile",
                    "success": False,
                    "error_type": "RuntimeError",
                }
            ],
        )

        assert benchmark_neighborlist.main() == 1

    def test_d3_setup_failure_is_written_as_failure_rows(self, monkeypatch, tmp_path):
        """D3 setup failures emit one row per planned cutoff."""
        params_path = tmp_path / "d3_params.pt"
        torch.save({"rcov": torch.tensor([1.0])}, params_path)
        config = {
            "params_path": str(params_path),
            "parameters": {
                "timing_runs": 1,
                "warmup_runs": 1,
                "cutoffs": [6.0, 15.0],
            },
            "runtime": {},
            "systems": {"cscl": {"enabled": True, "atom_counts": [128]}},
            "scaling": {"system_size": {"enabled": True}},
            "methods": [{"name": "dftd3", "enabled": True}],
            "dftd3_parameters": {"a1": 0.4289, "a2": 4.4407, "s8": 0.7875},
            "output": {"base_dir": str(tmp_path)},
        }

        monkeypatch.setattr(
            benchmark_dftd3,
            "_torch_d3_params_to_device",
            lambda params, _device: params,
        )
        monkeypatch.setattr(
            benchmark_dftd3,
            "create_system",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(TypeError("setup boom")),
        )

        rows = benchmark_dftd3.run_from_config(
            config,
            output_dir=tmp_path,
            backend="torch",
        )

        assert len(rows) == 2
        assert {row["cutoff"] for row in rows} == {6.0, 15.0}
        assert {row["error"] for row in rows} == {"setup boom"}
        assert {row["error_type"] for row in rows} == {"TypeError"}
        assert {row["failure_stage"] for row in rows} == {"system_setup"}

    def test_d3_parameter_setup_failure_is_written_as_failure_rows(
        self, monkeypatch, tmp_path
    ):
        """Malformed or incompatible D3 parameter setup is visible in every row."""
        params_path = tmp_path / "d3_params.pt"
        torch.save({"rcov": torch.tensor([1.0])}, params_path)
        config = {
            "params_path": str(params_path),
            "parameters": {
                "timing_runs": 1,
                "warmup_runs": 1,
                "cutoffs": [6.0, 15.0],
            },
            "runtime": {},
            "systems": {"cscl": {"enabled": True, "atom_counts": [128]}},
            "scaling": {"system_size": {"enabled": True}},
            "methods": [{"name": "dftd3", "enabled": True}],
            "dftd3_parameters": {"a1": 0.4289, "a2": 4.4407, "s8": 0.7875},
            "output": {"base_dir": str(tmp_path)},
        }
        monkeypatch.setattr(
            benchmark_dftd3,
            "_torch_d3_params_to_device",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                TypeError("parameter conversion boom")
            ),
        )

        rows = benchmark_dftd3.run_from_config(
            config,
            output_dir=tmp_path,
            backend="torch",
        )

        assert len(rows) == 2
        assert {row["success"] for row in rows} == {False}
        assert {row["error_type"] for row in rows} == {"TypeError"}
        assert {row["error"] for row in rows} == {"parameter conversion boom"}
        assert {row["failure_stage"] for row in rows} == {"parameter_setup"}

    def test_d3_missing_parameters_are_written_as_failure_rows(self, tmp_path):
        """Missing D3 parameter files emit planned CSV failure rows."""
        config = {
            "params_path": str(tmp_path / "missing_d3_params.pt"),
            "parameters": {
                "timing_runs": 1,
                "warmup_runs": 1,
                "cutoffs": [6.0, 15.0],
            },
            "runtime": {},
            "systems": {"cscl": {"enabled": True, "atom_counts": [128]}},
            "scaling": {"system_size": {"enabled": True}},
            "methods": [{"name": "dftd3", "enabled": True}],
            "dftd3_parameters": {"a1": 0.4289, "a2": 4.4407, "s8": 0.7875},
            "output": {"base_dir": str(tmp_path)},
        }

        rows = benchmark_dftd3.run_from_config(
            config,
            output_dir=tmp_path,
            backend="torch",
        )

        assert len(rows) == 2
        assert {row["cutoff"] for row in rows} == {6.0, 15.0}
        assert {row["success"] for row in rows} == {False}
        assert {row["error_type"] for row in rows} == {"FileNotFoundError"}
        assert (tmp_path / "d3-cscl-system-size-scaling.csv").exists()


class TestJaxMemoryContract:
    """Test JAX memory metadata behavior."""

    def test_jax_timed_batch_compiles_before_timing_when_zero_warmups(
        self, monkeypatch
    ):
        """JAX batch timing excludes compilation even when zero warmups are requested."""
        calls = []
        blocked = []

        fake_jax = SimpleNamespace(
            block_until_ready=lambda state: blocked.append(state),
        )
        monkeypatch.setitem(sys.modules, "jax", fake_jax)

        def step():
            calls.append(len(calls) + 1)
            return calls[-1]

        elapsed = jax_timed_batch(step, num_runs=2, warmup_runs=0)

        assert elapsed >= 0.0
        assert calls == [1, 2, 3]
        assert blocked == [1, 3]

    def test_jax_timed_stateful_threads_state(self, monkeypatch):
        """Stateful JAX timing carries donated buffers through every call."""
        seen_states = []
        blocked_states = []

        fake_jax = SimpleNamespace(
            block_until_ready=lambda state: blocked_states.append(state),
        )
        monkeypatch.setitem(sys.modules, "jax", fake_jax)

        def step(state):
            seen_states.append(state)
            return state + 1

        elapsed, final_state = jax_timed_stateful(
            step,
            state=0,
            num_runs=3,
            warmup_runs=2,
        )

        assert elapsed >= 0.0
        assert final_state == 5
        assert seen_states == [0, 1, 2, 3, 4]
        assert blocked_states == [1, 2, 5]

    def test_jax_timed_stateful_compiles_before_timing_when_zero_warmups(
        self, monkeypatch
    ):
        """Stateful JAX timing also excludes compile when zero warmups are requested."""
        seen_states = []
        blocked_states = []

        fake_jax = SimpleNamespace(
            block_until_ready=lambda state: blocked_states.append(state),
        )
        monkeypatch.setitem(sys.modules, "jax", fake_jax)

        def step(state):
            seen_states.append(state)
            return state + 1

        elapsed, final_state = jax_timed_stateful(
            step,
            state=0,
            num_runs=2,
            warmup_runs=0,
        )

        assert elapsed >= 0.0
        assert final_state == 3
        assert seen_states == [0, 1, 2]
        assert blocked_states == [1, 3]

    def test_measure_memory_jax_reports_nan_without_running_function(self):
        """JAX memory is unavailable and must not trigger an extra benchmark run."""

        class FakeJax:
            @staticmethod
            def block_until_ready(result):
                """Return the already-computed fake result."""
                return result

        def fail_if_called():
            raise AssertionError("JAX memory probe should not execute benchmarks")

        result, mem_info = measure_memory_jax(fail_if_called, FakeJax)

        assert result is None
        assert math.isnan(mem_info["mem_delta_mb"])
        assert math.isnan(mem_info["mem_peak_gb"])

    def test_plot_memory_suppresses_jax_rows(self):
        """JAX memory rows are not plotted even if stale CSVs contain values."""
        values = plot_benchmarks._get_memory_y(
            [
                {
                    "backend": "jax",
                    "mem_delta_mb": 123.0,
                    "mem_peak_gb": 80.0,
                },
            ]
        )

        assert values == [None]


class TestElectrostaticsParameterUnpack:
    """Test electrostatics benchmark parameter shape handling."""

    def test_torch_alpha_keeps_per_system_shape(self):
        """Torch component timing receives vector alpha, not a collapsed scalar."""
        pme_params = SimpleNamespace(
            alpha=torch.tensor([0.1, 0.2], dtype=torch.float64),
            real_space_cutoff=torch.tensor([8.0, 8.0], dtype=torch.float64),
            mesh_dimensions=(16, 16, 16),
        )
        ewald_params = SimpleNamespace(
            reciprocal_space_cutoff=torch.tensor([4.0, 4.0], dtype=torch.float64)
        )

        alpha, real_cutoff, mesh_dims, k_cutoff = _el_unpack_params(
            pme_params, ewald_params, "torch"
        )

        assert alpha.shape == (2,)
        assert torch.allclose(alpha, pme_params.alpha)
        assert real_cutoff == 8.0
        assert mesh_dims == (16, 16, 16)
        assert k_cutoff == 4.0

    def test_jax_alpha_object_is_preserved(self):
        """JAX component timing receives the original array-like alpha."""
        alpha_value = object()
        pme_params = SimpleNamespace(
            alpha=alpha_value,
            real_space_cutoff=[8.0],
            mesh_dimensions=(16, 16, 16),
        )
        ewald_params = SimpleNamespace(
            reciprocal_space_cutoff=SimpleNamespace(max=lambda: 4.0)
        )

        alpha, real_cutoff, mesh_dims, k_cutoff = _el_unpack_params(
            pme_params, ewald_params, "jax"
        )

        assert alpha is alpha_value
        assert real_cutoff == 8.0
        assert mesh_dims == (16, 16, 16)
        assert k_cutoff == 4.0


class TestBenchmarkCsvLoading:
    """Test benchmark CSV type conversion used by plotting."""

    def test_load_csv_parses_nan_tokens(self, tmp_path):
        """JAX memory NaN fields are parsed as floats, not strings."""
        csv_path = tmp_path / "results.csv"
        csv_path.write_text(
            "backend,success,mem_delta_mb,mem_peak_gb,time_us_per_atom\n"
            "jax,True,nan,nan,1.25\n",
            encoding="utf-8",
        )

        rows = load_csv(csv_path)

        assert len(rows) == 1
        assert isinstance(rows[0]["mem_delta_mb"], float)
        assert math.isnan(rows[0]["mem_delta_mb"])
        assert isinstance(rows[0]["mem_peak_gb"], float)
        assert math.isnan(rows[0]["mem_peak_gb"])

    def test_load_csv_retains_failed_coordinates_as_nan_gaps(self, tmp_path):
        """Plot loading preserves failure positions without treating them as data."""
        csv_path = tmp_path / "results.csv"
        csv_path.write_text(
            "backend,success,total_atoms,time_us_per_atom,error,error_type\n"
            "jax,True,128,1.25,,\n"
            "jax,False,256,nan,Out of memory,OutOfMemoryError\n",
            encoding="utf-8",
        )

        rows = load_csv(csv_path)

        assert len(rows) == 2
        assert rows[1]["success"] is False
        assert math.isnan(rows[1]["time_us_per_atom"])

    def test_load_csv_parses_empty_failed_measurements_as_nan(self, tmp_path):
        """Blank failed measurements become plot gaps rather than strings."""
        csv_path = tmp_path / "results.csv"
        csv_path.write_text(
            "backend,success,total_atoms,time_us_per_atom,"
            "throughput_atoms_per_sec,mem_delta_mb,mem_peak_gb,error\n"
            "jax,False,256,,,,,\n",
            encoding="utf-8",
        )

        row = load_csv(csv_path)[0]

        for field in (
            "time_us_per_atom",
            "throughput_atoms_per_sec",
            "mem_delta_mb",
            "mem_peak_gb",
        ):
            assert math.isnan(row[field])
        assert row["error"] == ""
