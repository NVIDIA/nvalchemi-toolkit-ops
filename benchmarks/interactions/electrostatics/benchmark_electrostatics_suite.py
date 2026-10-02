#!/usr/bin/env python3
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

"""Electrostatics Benchmark (Ewald + PME).

CRITICAL: Electrostatics requires float64 for positions and cells.
K-vectors are pre-computed ONCE outside the timing loop.

Usage (run from the repository root):
    python -m benchmarks.interactions.electrostatics.benchmark_electrostatics_suite \
        --config benchmarks/interactions/electrostatics/benchmark_config.yaml
    python -m benchmarks.interactions.electrostatics.benchmark_electrostatics_suite \
        --config benchmarks/interactions/electrostatics/benchmark_config.yaml \
        --output-dir docs/benchmarks/benchmark_results

    # JAX backend (the runner sets JAX env defaults before importing JAX)
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
        python -m benchmarks.interactions.electrostatics.benchmark_electrostatics_suite \
        --config benchmarks/interactions/electrostatics/benchmark_config.yaml --backend jax

Backends
--------
``--backend torch`` (default) uses the warp-based torch kernels and CUDA
events for timing. ``--backend jax`` uses the JAX wrappers in
``nvalchemiops.jax.interactions.electrostatics`` and wall-clock timing.

Each reportable row measures one complete differentiable workload: evaluate
the public energy API, then derive forces and charge gradients through the
framework's automatic differentiation machinery. The timed result therefore
contains energy, ``-dE/dR``, and ``dE/dq`` for the same call.

JAX caveats:

- The runner enables ``jax_enable_x64`` before importing electrostatics;
  electrostatics requires float64 positions and cells.
- JAX EL rows use ``jax.value_and_grad`` with both positions and charges as
  differentiated arguments. Torch rows use ``torch.autograd.grad`` over the
  same two inputs.

Environment variables for ``--backend jax``:

- ``XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`` — set by the runner before
  importing JAX unless the user already configured JAX memory behavior. JAX's
  normal preallocator avoids the fragmentation seen with on-demand allocation
  in large benchmark sweeps.
- ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` — optional user override. If unset,
  the runner prints a one-line note before JAX import and keeps the default
  preallocator capped by ``XLA_PYTHON_CLIENT_MEM_FRACTION``.
- ``JAX_ENABLE_X64=True`` — set by the suite/runner for electrostatics;
  exporting it explicitly is also safe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch

__all__ = [
    "ElectrostaticsInputs",
    "benchmark_ewald",
    "benchmark_pme",
    "main",
    "merge_cli_overrides",
    "parse_args",
    "run_from_config",
]

from benchmarks.config import (
    add_common_cli_args,
    load_yaml_config,
    merge_common_cli_overrides,
)
from benchmarks.constants import DEFAULT_NL_SAFETY_FACTOR
from benchmarks.suite_systems import (
    benchmark_system_metadata,
    compute_atomic_density,
    configs_for_mode,
    configured_nh3_artifacts,
    create_system,
    filter_configs_by_total_atoms,
    planned_atom_counts,
    resolve_nh3_dir,
)
from benchmarks.suite_utils import (
    build_failure_result,
    build_result,
    build_skipped_result,
    clean_gpu,
    clean_jax,
    configure_input_provenance,
    create_run_directory,
    current_alloc_gb,
    failure_error_type,
    fixed_total_metadata,
    format_num,
    jax_timed_batch,
    jax_timed_serial,
    lazy_import_jax,
    make_csv_name,
    make_row_meta,
    measure_memory_jax,
    measure_memory_torch,
    measure_timing_batches,
    neighbor_count_metadata,
    save_results,
    summarize_timing_batches,
)
from nvalchemiops.interactions.electrostatics._pme_mesh import (
    _DEFAULT_PME_FFT_PADDING_FRACTION,
)
from nvalchemiops.neighbors import estimate_max_neighbors
from nvalchemiops.torch.interactions.electrostatics import (
    compute_bspline_moduli_1d,
    estimate_ewald_parameters,
    estimate_pme_parameters,
    ewald_real_space,
    ewald_reciprocal_space,
    ewald_summation,
    generate_k_squared_pme,
    generate_k_vectors_ewald_summation,
    particle_mesh_ewald,
    pme_reciprocal_space,
)
from nvalchemiops.torch.neighbors import batch_cell_list

# =============================================================================
# Config Loading
# =============================================================================


def merge_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply CLI overrides on top of YAML config.

    Adds EL-specific ``--accuracies`` on top of the shared flags.
    """
    config = merge_common_cli_overrides(config, args)
    if args.accuracies is not None:
        config["accuracies"] = args.accuracies
    if getattr(args, "profile_components", None) is not None:
        config["parameters"]["profile_components"] = args.profile_components
    return config


def _configured_pme_cutoff(parameters: dict[str, Any]) -> float:
    """Read the positive PME cutoff limit from the versioned configuration."""
    cutoff = float(parameters["max_real_space_cutoff"])
    if not math.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("PME max_real_space_cutoff must be positive and finite")
    return cutoff


def _prepare_el_families(
    config: dict, *, plan_only: bool = False
) -> dict[tuple[str, str], tuple[dict, ...]]:
    """List configured physical cases without invoking a parameter search."""
    families = {}
    for sys_name, sys_config in config["systems"].items():
        if not sys_config.get("enabled", True):
            continue
        nh3_dir = resolve_nh3_dir(sys_config)
        for mode_name, mode_config in config["scaling"].items():
            if not isinstance(mode_config, dict) or not mode_config.get(
                "enabled", True
            ):
                continue
            families[(sys_name, mode_name)] = tuple(
                configs_for_mode(
                    mode_name,
                    mode_config,
                    sys_name,
                    sys_config,
                    nh3_dir,
                    plan_only=plan_only,
                )
            )
    return families


def _canonical_json(value: Any) -> str:
    """Serialize one benchmark-planning value without platform-dependent spaces."""
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_pme_input_sha256(
    positions: np.ndarray,
    charges: np.ndarray,
    cell: np.ndarray,
    batch_idx: np.ndarray,
    pbc: np.ndarray,
) -> str:
    """Hash exact canonical arrays with stable dtypes, shapes, and byte order."""
    arrays = (
        ("positions", np.asarray(positions, dtype="<f8")),
        ("charges", np.asarray(charges, dtype="<f8")),
        ("cell", np.asarray(cell, dtype="<f8")),
        ("batch_idx", np.asarray(batch_idx, dtype="<i8")),
        ("pbc", np.asarray(pbc, dtype=np.bool_)),
    )
    digest = hashlib.sha256()
    digest.update(b"nvalchemiops-benchmark-pme-canonical-input-v1\0")
    for name, array in arrays:
        contiguous = np.ascontiguousarray(array)
        header = _canonical_json(
            {
                "name": name,
                "dtype": contiguous.dtype.str,
                "shape": contiguous.shape,
            }
        ).encode("ascii")
        digest.update(len(header).to_bytes(8, byteorder="big"))
        digest.update(header)
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


# =============================================================================
# Inputs Bundle
# =============================================================================


@dataclass(frozen=True)
class ElectrostaticsInputs:
    """Per-config tensor bundle passed to the benchmark kernels.

    Bundles the system state (positions/charges/cell/pbc/batch_idx) with the
    precomputed neighbor list (nl_data/nl_shifts/nl_ptr) so the kernel-level
    benchmark functions take one object instead of eight positional args.

    Frozen — the runner builds one per (system, accuracy) config and passes
    it through. Field types are ``Any`` because the same shape applies to
    both torch tensors and jax arrays; the ``backend`` field disambiguates.
    """

    positions: Any
    charges: Any
    cell: Any
    pbc: Any
    batch_idx: Any
    nl_data: Any
    nl_shifts: Any
    nl_ptr: Any
    backend: str
    # max_atoms_per_system is needed by the JAX ewald_summation API under
    # jax.jit (shape inference can't query batch_idx.max() inside a trace).
    # Our systems are uniform, so this equals the per-system atom count.
    max_atoms_per_system: int = 0


_JAX_EL_KERNEL_CACHE: dict[int, dict[str, Any]] = {}


class _TorchPmeStaticMetadata(NamedTuple):
    """Fixed-cell PME tensors that remain valid across timed calls."""

    volume: torch.Tensor
    cell_inv_t: torch.Tensor
    moduli_x: torch.Tensor
    moduli_y: torch.Tensor
    moduli_z: torch.Tensor


def _torch_pme_static_metadata(
    cell: torch.Tensor,
    mesh_dims: tuple[int, int, int],
    spline_order: int,
) -> _TorchPmeStaticMetadata:
    """Precompute reusable Torch PME cell and spline metadata."""
    cell_3d = cell if cell.dim() == 3 else cell.unsqueeze(0)
    cell_inv_t = torch.linalg.inv(cell_3d).transpose(1, 2)
    volume = torch.abs(torch.linalg.det(cell_3d))
    mesh_nx, mesh_ny, mesh_nz = mesh_dims
    fft_kwargs = {"device": cell.device, "dtype": cell.dtype}
    miller_x = torch.fft.fftfreq(mesh_nx, d=1.0 / mesh_nx, **fft_kwargs)
    miller_y = torch.fft.fftfreq(mesh_ny, d=1.0 / mesh_ny, **fft_kwargs)
    miller_z = torch.fft.rfftfreq(mesh_nz, d=1.0 / mesh_nz, **fft_kwargs)
    moduli_x = compute_bspline_moduli_1d(miller_x, mesh_nx, spline_order)
    moduli_y = compute_bspline_moduli_1d(miller_y, mesh_ny, spline_order)
    moduli_z = compute_bspline_moduli_1d(miller_z, mesh_nz, spline_order)
    return _TorchPmeStaticMetadata(
        volume=volume,
        cell_inv_t=cell_inv_t,
        moduli_x=moduli_x,
        moduli_y=moduli_y,
        moduli_z=moduli_z,
    )


def _torch_pme_k_squared_metadata(
    cell: torch.Tensor,
    mesh_dims: tuple[int, int, int],
) -> tuple[torch.Tensor, str]:
    """Precompute PME magnitudes without retaining unused Cartesian vectors."""
    cell_3d = cell if cell.dim() == 3 else cell.unsqueeze(0)
    repeated_cells = cell_3d.shape[0] > 1 and torch.equal(
        cell_3d, cell_3d[:1].expand_as(cell_3d)
    )
    metadata_cell = cell_3d[:1] if repeated_cells else cell_3d
    k_squared = generate_k_squared_pme(metadata_cell, mesh_dims)
    if repeated_cells:
        k_squared = k_squared.unsqueeze(0).expand(cell_3d.shape[0], -1, -1, -1)
        k_squared = k_squared.contiguous()
    cache_mode = "shared_cell_k_squared" if repeated_cells else "k_squared_only"
    return k_squared, cache_mode


def _get_jax_el_kernels(jax_api: dict) -> dict[str, Any]:
    """Return cached JAX energy-and-gradient kernels."""
    jax = jax_api["jax"]
    jnp = jax_api["jnp"]
    cache_key = id(jax)
    cached = _JAX_EL_KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    jax_pme = jax_api["particle_mesh_ewald"]
    jax_pme_reciprocal = jax_api["pme_reciprocal_space"]
    jax_ewald = jax_api["ewald_summation"]
    jax_real = jax_api["ewald_real_space"]
    jax_reciprocal = jax_api["ewald_reciprocal_space"]

    def value_forces_charge_gradients(energy_fn, positions, charges):
        """Return per-atom energy, forces, and charge gradients."""

        def total_energy(pos, charge):
            energies = energy_fn(pos, charge)
            return jnp.sum(energies), energies

        (_, energies), (position_grad, charge_grad) = jax.value_and_grad(
            total_energy,
            argnums=(0, 1),
            has_aux=True,
        )(positions, charges)
        return energies, -position_grad, charge_grad

    def pme_real_kernel(
        positions,
        charges,
        cell,
        alpha,
        batch_idx,
        nl_data,
        nl_ptr,
        nl_shifts,
    ):
        def energy_fn(pos, charge):
            return jax_real(
                positions=pos,
                charges=charge,
                cell=cell,
                alpha=alpha,
                batch_idx=batch_idx,
                neighbor_list=nl_data,
                neighbor_ptr=nl_ptr,
                neighbor_shifts=nl_shifts,
            )

        return value_forces_charge_gradients(
            energy_fn,
            positions,
            charges,
        )

    def pme_reciprocal_kernel(
        positions,
        charges,
        cell,
        alpha,
        batch_idx,
        k_vectors,
        k_squared,
        volume,
        cell_inv_t,
        moduli_x,
        moduli_y,
        moduli_z,
        *,
        mesh_dims,
        spline_order,
    ):
        def energy_fn(pos, charge):
            return jax_pme_reciprocal(
                positions=pos,
                charges=charge,
                cell=cell,
                alpha=alpha,
                mesh_dimensions=mesh_dims,
                spline_order=spline_order,
                batch_idx=batch_idx,
                k_vectors=k_vectors,
                k_squared=k_squared,
                volume=volume,
                cell_inv_t=cell_inv_t,
                moduli_x=moduli_x,
                moduli_y=moduli_y,
                moduli_z=moduli_z,
            )

        return value_forces_charge_gradients(
            energy_fn,
            positions,
            charges,
        )

    def pme_kernel(
        positions,
        charges,
        cell,
        alpha,
        batch_idx,
        k_vectors,
        k_squared,
        nl_data,
        nl_ptr,
        nl_shifts,
        volume,
        cell_inv_t,
        moduli_x,
        moduli_y,
        moduli_z,
        *,
        mesh_dims,
        spline_order,
        accuracy,
    ):
        def energy_fn(pos, charge):
            return jax_pme(
                positions=pos,
                charges=charge,
                cell=cell,
                alpha=alpha,
                mesh_dimensions=mesh_dims,
                spline_order=spline_order,
                batch_idx=batch_idx,
                k_vectors=k_vectors,
                k_squared=k_squared,
                neighbor_list=nl_data,
                neighbor_ptr=nl_ptr,
                neighbor_shifts=nl_shifts,
                volume=volume,
                cell_inv_t=cell_inv_t,
                moduli_x=moduli_x,
                moduli_y=moduli_y,
                moduli_z=moduli_z,
                accuracy=accuracy,
            )

        return value_forces_charge_gradients(
            energy_fn,
            positions,
            charges,
        )

    def ewald_real_kernel(
        positions,
        charges,
        cell,
        alpha,
        batch_idx,
        nl_data,
        nl_ptr,
        nl_shifts,
    ):
        def energy_fn(pos, charge):
            return jax_real(
                positions=pos,
                charges=charge,
                cell=cell,
                alpha=alpha,
                batch_idx=batch_idx,
                neighbor_list=nl_data,
                neighbor_ptr=nl_ptr,
                neighbor_shifts=nl_shifts,
            )

        return value_forces_charge_gradients(
            energy_fn,
            positions,
            charges,
        )

    def ewald_reciprocal_kernel(
        positions,
        charges,
        cell,
        alpha,
        batch_idx,
        k_vectors,
        *,
        max_atoms_per_system,
    ):
        def energy_fn(pos, charge):
            return jax_reciprocal(
                positions=pos,
                charges=charge,
                cell=cell,
                alpha=alpha,
                batch_idx=batch_idx,
                max_atoms_per_system=max_atoms_per_system,
                k_vectors=k_vectors,
            )

        return value_forces_charge_gradients(
            energy_fn,
            positions,
            charges,
        )

    def ewald_kernel(
        positions,
        charges,
        cell,
        alpha,
        batch_idx,
        k_vectors,
        nl_data,
        nl_ptr,
        nl_shifts,
        *,
        k_cutoff,
        max_atoms_per_system,
        accuracy,
    ):
        def energy_fn(pos, charge):
            return jax_ewald(
                positions=pos,
                charges=charge,
                cell=cell,
                alpha=alpha,
                k_cutoff=k_cutoff,
                batch_idx=batch_idx,
                max_atoms_per_system=max_atoms_per_system,
                neighbor_list=nl_data,
                neighbor_ptr=nl_ptr,
                neighbor_shifts=nl_shifts,
                k_vectors=k_vectors,
                accuracy=accuracy,
            )

        return value_forces_charge_gradients(
            energy_fn,
            positions,
            charges,
        )

    kernels = {
        "pme_real": jax.jit(pme_real_kernel),
        "pme_reciprocal": jax.jit(
            pme_reciprocal_kernel,
            static_argnames=("mesh_dims", "spline_order"),
        ),
        "pme": jax.jit(
            pme_kernel,
            static_argnames=("mesh_dims", "spline_order", "accuracy"),
        ),
        "ewald_real": jax.jit(ewald_real_kernel),
        "ewald_reciprocal": jax.jit(
            ewald_reciprocal_kernel,
            static_argnames=("max_atoms_per_system",),
        ),
        "ewald": jax.jit(
            ewald_kernel,
            static_argnames=(
                "k_cutoff",
                "max_atoms_per_system",
                "accuracy",
            ),
        ),
    }
    _JAX_EL_KERNEL_CACHE[cache_key] = kernels
    return kernels


# =============================================================================
# Core Benchmarks
# =============================================================================


def _energy_derivative_metadata(
    profile_components: bool = False,
) -> dict[str, str | bool]:
    """Describe the single reportable electrostatics workload."""
    return {
        "derivative_contract": "energy_autograd",
        "workload": "energy_forces_charge_gradients",
        "compute_forces": True,
        "compute_charge_gradients": True,
        "component_profiled": profile_components,
    }


def _el_row_meta(
    sys_name: str,
    mode_name: str,
    mode_config: dict,
    backend: str,
    atoms_per_system: int,
    batch_size: int,
    total_atoms: int,
) -> dict:
    """Build common EL row identity plus fixed-N planning metadata."""
    row_meta = make_row_meta(
        sys_name,
        mode_name,
        backend,
        atoms_per_system,
        batch_size,
        total_atoms,
        mode_config=mode_config,
    )
    return row_meta


def _torch_energy_forces_charge_gradients(
    energy_fn,
    positions: torch.Tensor,
    charges: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate energy and derive forces and charge gradients with autograd."""
    energies = energy_fn(positions, charges)
    position_grad, charge_grad = torch.autograd.grad(
        energies.sum(),
        (positions, charges),
    )
    return energies, -position_grad, charge_grad


def benchmark_pme(
    inputs: ElectrostaticsInputs,
    alpha: Any,
    mesh_dims: tuple[int, int, int],
    spline_order: int,
    accuracy: float,
    num_runs: int,
    warmup_runs: int,
    jax_api: dict | None = None,
    *,
    profile_components: bool = False,
    timing_batches: int,
) -> dict:
    """Benchmark Particle Mesh Ewald for one config.

    Dispatches on ``inputs.backend``. Accepts pre-converted f64 tensors/arrays
    on ``inputs`` to avoid redundant GPU copies.
    """
    if timing_batches <= 0:
        raise ValueError("timing_batches must be positive")
    if inputs.backend == "jax":
        return _benchmark_pme_jax(
            inputs,
            alpha,
            mesh_dims,
            spline_order,
            accuracy,
            num_runs,
            warmup_runs,
            jax_api,
            profile_components=profile_components,
            timing_batches=timing_batches,
        )

    positions = inputs.positions.detach().requires_grad_(True)
    charges = inputs.charges.detach().requires_grad_(True)
    k_squared, pme_cache_mode = _torch_pme_k_squared_metadata(inputs.cell, mesh_dims)
    static_metadata = _torch_pme_static_metadata(
        inputs.cell,
        mesh_dims,
        spline_order,
    )

    def real_energy(pos, charge):
        return ewald_real_space(
            positions=pos,
            charges=charge,
            cell=inputs.cell,
            alpha=alpha,
            batch_idx=inputs.batch_idx,
            neighbor_list=inputs.nl_data,
            neighbor_ptr=inputs.nl_ptr,
            neighbor_shifts=inputs.nl_shifts,
        )

    def reciprocal_energy(pos, charge):
        return pme_reciprocal_space(
            positions=pos,
            charges=charge,
            cell=inputs.cell,
            alpha=alpha,
            mesh_dimensions=mesh_dims,
            spline_order=spline_order,
            batch_idx=inputs.batch_idx,
            k_vectors=None,
            k_squared=k_squared,
            volume=static_metadata.volume,
            cell_inv_t=static_metadata.cell_inv_t,
            moduli_x=static_metadata.moduli_x,
            moduli_y=static_metadata.moduli_y,
            moduli_z=static_metadata.moduli_z,
        )

    def pme_energy(pos, charge):
        return particle_mesh_ewald(
            positions=pos,
            charges=charge,
            cell=inputs.cell,
            alpha=alpha,
            mesh_dimensions=mesh_dims,
            spline_order=spline_order,
            batch_idx=inputs.batch_idx,
            k_vectors=None,
            k_squared=k_squared,
            volume=static_metadata.volume,
            cell_inv_t=static_metadata.cell_inv_t,
            moduli_x=static_metadata.moduli_x,
            moduli_y=static_metadata.moduli_y,
            moduli_z=static_metadata.moduli_z,
            neighbor_list=inputs.nl_data,
            neighbor_ptr=inputs.nl_ptr,
            neighbor_shifts=inputs.nl_shifts,
            accuracy=accuracy,
        )

    def run_real():
        return _torch_energy_forces_charge_gradients(real_energy, positions, charges)

    def run_reciprocal():
        return _torch_energy_forces_charge_gradients(
            reciprocal_energy,
            positions,
            charges,
        )

    def run_pme():
        return _torch_energy_forces_charge_gradients(pme_energy, positions, charges)

    memory_result, mem_info = measure_memory_torch(run_pme)
    del memory_result
    time_real_sec = float("nan")
    time_reciprocal_sec = float("nan")
    timing_method_real = "not_measured"
    timing_method_reciprocal = "not_measured"
    real_max_min_ratio: float | str = ""
    real_samples_json = ""
    reciprocal_max_min_ratio: float | str = ""
    reciprocal_samples_json = ""
    if profile_components:
        real_samples = measure_timing_batches(
            run_real,
            num_runs,
            warmup_runs,
            timing_batches,
        )
        reciprocal_samples = measure_timing_batches(
            run_reciprocal,
            num_runs,
            warmup_runs,
            timing_batches,
        )
        time_real_sec, real_max_min_ratio, real_samples_json = summarize_timing_batches(
            real_samples
        )
        time_reciprocal_sec, reciprocal_max_min_ratio, reciprocal_samples_json = (
            summarize_timing_batches(reciprocal_samples)
        )
        timing_method_real = "torch_cuda_events"
        timing_method_reciprocal = "torch_cuda_events"
    timing_samples = measure_timing_batches(
        run_pme,
        num_runs,
        warmup_runs,
        timing_batches,
    )
    time_sec, max_min_ratio, samples_json = summarize_timing_batches(timing_samples)
    return {
        "time_seconds": time_sec,
        "time_real_seconds": time_real_sec,
        "time_reciprocal_seconds": time_reciprocal_sec,
        "mem_info": mem_info,
        "timing_batches": timing_batches,
        "timing_batch_aggregation": "arithmetic_mean",
        "timing_batch_max_min_ratio": max_min_ratio,
        "timing_batch_seconds_json": samples_json,
        "timing_batch_real_max_min_ratio": real_max_min_ratio,
        "timing_batch_real_seconds_json": real_samples_json,
        "timing_batch_reciprocal_max_min_ratio": reciprocal_max_min_ratio,
        "timing_batch_reciprocal_seconds_json": reciprocal_samples_json,
        "timing_method": "torch_cuda_events",
        "timing_method_real": timing_method_real,
        "timing_method_reciprocal": timing_method_reciprocal,
        "pme_cache_mode": pme_cache_mode,
        "component_profiled": profile_components,
    }


def benchmark_ewald(
    inputs: ElectrostaticsInputs,
    alpha: Any,
    k_cutoff: float,
    accuracy: float,
    num_runs: int,
    warmup_runs: int,
    jax_api: dict | None = None,
    *,
    profile_components: bool = False,
    timing_batches: int,
) -> dict:
    """Benchmark Ewald summation for one config via the unified API."""
    if timing_batches <= 0:
        raise ValueError("timing_batches must be positive")
    if inputs.backend == "jax":
        return _benchmark_ewald_jax(
            inputs,
            alpha,
            k_cutoff,
            accuracy,
            num_runs,
            warmup_runs,
            jax_api,
            profile_components=profile_components,
            timing_batches=timing_batches,
        )

    positions = inputs.positions.detach().requires_grad_(True)
    charges = inputs.charges.detach().requires_grad_(True)
    k_vectors = generate_k_vectors_ewald_summation(inputs.cell, k_cutoff)
    if k_vectors.ndim == 2:
        k_vectors = k_vectors.unsqueeze(0)
    num_k_vectors_per_system = int(k_vectors.shape[-2])

    def real_energy(pos, charge):
        return ewald_real_space(
            positions=pos,
            charges=charge,
            cell=inputs.cell,
            alpha=alpha,
            batch_idx=inputs.batch_idx,
            neighbor_list=inputs.nl_data,
            neighbor_ptr=inputs.nl_ptr,
            neighbor_shifts=inputs.nl_shifts,
        )

    def reciprocal_energy(pos, charge):
        return ewald_reciprocal_space(
            positions=pos,
            charges=charge,
            cell=inputs.cell,
            alpha=alpha,
            batch_idx=inputs.batch_idx,
            k_vectors=k_vectors,
            max_atoms_per_system=inputs.max_atoms_per_system,
        )

    def ewald_energy(pos, charge):
        return ewald_summation(
            positions=pos,
            charges=charge,
            cell=inputs.cell,
            alpha=alpha,
            k_cutoff=k_cutoff,
            batch_idx=inputs.batch_idx,
            neighbor_list=inputs.nl_data,
            neighbor_ptr=inputs.nl_ptr,
            neighbor_shifts=inputs.nl_shifts,
            k_vectors=k_vectors,
            accuracy=accuracy,
            max_atoms_per_system=inputs.max_atoms_per_system,
        )

    def run_real():
        return _torch_energy_forces_charge_gradients(real_energy, positions, charges)

    def run_reciprocal():
        return _torch_energy_forces_charge_gradients(
            reciprocal_energy,
            positions,
            charges,
        )

    def run_ewald():
        return _torch_energy_forces_charge_gradients(ewald_energy, positions, charges)

    memory_result, mem_info = measure_memory_torch(run_ewald)
    del memory_result
    time_real_sec = float("nan")
    time_reciprocal_sec = float("nan")
    timing_method_real = "not_measured"
    timing_method_reciprocal = "not_measured"
    real_max_min_ratio: float | str = ""
    real_samples_json = ""
    reciprocal_max_min_ratio: float | str = ""
    reciprocal_samples_json = ""
    if profile_components:
        real_samples = measure_timing_batches(
            run_real,
            num_runs,
            warmup_runs,
            timing_batches,
        )
        reciprocal_samples = measure_timing_batches(
            run_reciprocal,
            num_runs,
            warmup_runs,
            timing_batches,
        )
        time_real_sec, real_max_min_ratio, real_samples_json = summarize_timing_batches(
            real_samples
        )
        time_reciprocal_sec, reciprocal_max_min_ratio, reciprocal_samples_json = (
            summarize_timing_batches(reciprocal_samples)
        )
        timing_method_real = "torch_cuda_events"
        timing_method_reciprocal = "torch_cuda_events"
    timing_samples = measure_timing_batches(
        run_ewald,
        num_runs,
        warmup_runs,
        timing_batches,
    )
    time_sec, max_min_ratio, samples_json = summarize_timing_batches(timing_samples)
    return {
        "time_seconds": time_sec,
        "time_real_seconds": time_real_sec,
        "time_reciprocal_seconds": time_reciprocal_sec,
        "mem_info": mem_info,
        "timing_batches": timing_batches,
        "timing_batch_aggregation": "arithmetic_mean",
        "timing_batch_max_min_ratio": max_min_ratio,
        "timing_batch_seconds_json": samples_json,
        "timing_batch_real_max_min_ratio": real_max_min_ratio,
        "timing_batch_real_seconds_json": real_samples_json,
        "timing_batch_reciprocal_max_min_ratio": reciprocal_max_min_ratio,
        "timing_batch_reciprocal_seconds_json": reciprocal_samples_json,
        "timing_method": "torch_cuda_events",
        "timing_method_real": timing_method_real,
        "timing_method_reciprocal": timing_method_reciprocal,
        "component_profiled": profile_components,
        "num_k_vectors_per_system": num_k_vectors_per_system,
    }


def _jax_timing_batches(
    fn,
    num_runs: int,
    warmup_runs: int,
    timing_batches: int,
    *,
    queued_timer: Callable[..., float] = jax_timed_batch,
    serial_timer: Callable[..., float] = jax_timed_serial,
    oom_cleanup: Callable[..., None] = clean_jax,
) -> tuple[tuple[float, ...], str]:
    """Measure one JAX row with a single timing method across every batch."""
    if timing_batches <= 0:
        raise ValueError("timing_batches must be positive")

    def measure(timer: Callable[..., float]) -> tuple[float, ...]:
        return tuple(
            timer(
                fn,
                num_runs,
                warmup_runs=warmup_runs if batch_index == 0 else None,
            )
            for batch_index in range(timing_batches)
        )

    try:
        return measure(queued_timer), "jax_wall_block_until_ready"
    except Exception as error:
        if failure_error_type(error) != "OutOfMemoryError":
            raise

    oom_cleanup(clear_executables=True)
    return measure(serial_timer), "jax_wall_block_each"


def _benchmark_pme_jax(
    inputs,
    alpha,
    mesh_dims,
    spline_order,
    accuracy,
    num_runs,
    warmup_runs,
    jax_api,
    *,
    profile_components=False,
    timing_batches,
):
    """JAX backend implementation of :func:`benchmark_pme`."""
    jax = jax_api["jax"]
    jnp = jax_api["jnp"]
    k_squared_pme = jax_api["generate_k_squared_pme"]
    compute_bspline_moduli_1d = jax_api["compute_bspline_moduli_1d"]
    kernels = _get_jax_el_kernels(jax_api)

    cell_3d = inputs.cell if inputs.cell.ndim == 3 else jnp.expand_dims(inputs.cell, 0)
    repeated_cells = cell_3d.shape[0] > 1 and bool(
        jax.device_get(jnp.all(cell_3d == cell_3d[:1]))
    )
    metadata_cell = cell_3d[:1] if repeated_cells else cell_3d
    k_squared = k_squared_pme(metadata_cell, mesh_dims)
    jax.block_until_ready(k_squared)
    k_vectors = None
    pme_cache_mode = "shared_cell_k_squared" if repeated_cells else "k_squared_only"
    mesh_nx, mesh_ny, mesh_nz = mesh_dims
    dtype = inputs.positions.dtype
    cell_inv_t = jnp.transpose(jnp.linalg.inv(cell_3d), (0, 2, 1)).astype(dtype)
    volume = jnp.abs(jnp.linalg.det(cell_3d)).astype(dtype)
    miller_x = jnp.fft.fftfreq(mesh_nx, d=1.0 / mesh_nx).astype(dtype)
    miller_y = jnp.fft.fftfreq(mesh_ny, d=1.0 / mesh_ny).astype(dtype)
    miller_z = jnp.fft.rfftfreq(mesh_nz, d=1.0 / mesh_nz).astype(dtype)
    moduli_x = compute_bspline_moduli_1d(miller_x, mesh_nx, spline_order)
    moduli_y = compute_bspline_moduli_1d(miller_y, mesh_ny, spline_order)
    moduli_z = compute_bspline_moduli_1d(miller_z, mesh_nz, spline_order)

    def run_real():
        return kernels["pme_real"](
            inputs.positions,
            inputs.charges,
            inputs.cell,
            alpha,
            inputs.batch_idx,
            inputs.nl_data,
            inputs.nl_ptr,
            inputs.nl_shifts,
        )

    def run_reciprocal():
        return kernels["pme_reciprocal"](
            inputs.positions,
            inputs.charges,
            inputs.cell,
            alpha,
            inputs.batch_idx,
            k_vectors,
            k_squared,
            volume,
            cell_inv_t,
            moduli_x,
            moduli_y,
            moduli_z,
            mesh_dims=mesh_dims,
            spline_order=spline_order,
        )

    def run_pme():
        return kernels["pme"](
            inputs.positions,
            inputs.charges,
            inputs.cell,
            alpha,
            inputs.batch_idx,
            k_vectors,
            k_squared,
            inputs.nl_data,
            inputs.nl_ptr,
            inputs.nl_shifts,
            volume,
            cell_inv_t,
            moduli_x,
            moduli_y,
            moduli_z,
            mesh_dims=mesh_dims,
            spline_order=spline_order,
            accuracy=accuracy,
        )

    # JAX memory is unavailable in this suite; avoid an extra allocation-heavy
    # execution that would only produce NaN memory fields.
    _, mem_info = measure_memory_jax(run_pme, jax)
    time_real_sec = float("nan")
    time_reciprocal_sec = float("nan")
    timing_method_real = "not_measured"
    timing_method_reciprocal = "not_measured"
    real_max_min_ratio: float | str = ""
    real_samples_json = ""
    reciprocal_max_min_ratio: float | str = ""
    reciprocal_samples_json = ""
    if profile_components:
        real_samples, timing_method_real = _jax_timing_batches(
            run_real,
            num_runs,
            warmup_runs,
            timing_batches,
        )
        reciprocal_samples, timing_method_reciprocal = _jax_timing_batches(
            run_reciprocal,
            num_runs,
            warmup_runs,
            timing_batches,
        )
        time_real_sec, real_max_min_ratio, real_samples_json = summarize_timing_batches(
            real_samples
        )
        time_reciprocal_sec, reciprocal_max_min_ratio, reciprocal_samples_json = (
            summarize_timing_batches(reciprocal_samples)
        )
    timing_samples, timing_method = _jax_timing_batches(
        run_pme,
        num_runs,
        warmup_runs,
        timing_batches,
    )
    time_sec, max_min_ratio, samples_json = summarize_timing_batches(timing_samples)
    return {
        "time_seconds": time_sec,
        "time_real_seconds": time_real_sec,
        "time_reciprocal_seconds": time_reciprocal_sec,
        "timing_method": timing_method,
        "timing_method_real": timing_method_real,
        "timing_method_reciprocal": timing_method_reciprocal,
        "pme_cache_mode": pme_cache_mode,
        "component_profiled": profile_components,
        "mem_info": mem_info,
        "timing_batches": timing_batches,
        "timing_batch_aggregation": "arithmetic_mean",
        "timing_batch_max_min_ratio": max_min_ratio,
        "timing_batch_seconds_json": samples_json,
        "timing_batch_real_max_min_ratio": real_max_min_ratio,
        "timing_batch_real_seconds_json": real_samples_json,
        "timing_batch_reciprocal_max_min_ratio": reciprocal_max_min_ratio,
        "timing_batch_reciprocal_seconds_json": reciprocal_samples_json,
    }


def _benchmark_ewald_jax(
    inputs,
    alpha,
    k_cutoff,
    accuracy,
    num_runs,
    warmup_runs,
    jax_api,
    *,
    profile_components=False,
    timing_batches,
):
    """JAX backend implementation of :func:`benchmark_ewald`."""

    jax = jax_api["jax"]
    k_ewald = jax_api["generate_k_vectors_ewald_summation"]
    kernels = _get_jax_el_kernels(jax_api)

    k_vectors = k_ewald(inputs.cell, k_cutoff)
    if k_vectors.ndim == 2:
        k_vectors = jax_api["jnp"].expand_dims(k_vectors, 0)
    num_k_vectors_per_system = int(k_vectors.shape[-2])

    def run_real():
        return kernels["ewald_real"](
            inputs.positions,
            inputs.charges,
            inputs.cell,
            alpha,
            inputs.batch_idx,
            inputs.nl_data,
            inputs.nl_ptr,
            inputs.nl_shifts,
        )

    def run_reciprocal():
        return kernels["ewald_reciprocal"](
            inputs.positions,
            inputs.charges,
            inputs.cell,
            alpha,
            inputs.batch_idx,
            k_vectors,
            max_atoms_per_system=inputs.max_atoms_per_system,
        )

    def run_ewald():
        return kernels["ewald"](
            inputs.positions,
            inputs.charges,
            inputs.cell,
            alpha,
            inputs.batch_idx,
            k_vectors,
            inputs.nl_data,
            inputs.nl_ptr,
            inputs.nl_shifts,
            k_cutoff=k_cutoff,
            max_atoms_per_system=inputs.max_atoms_per_system,
            accuracy=accuracy,
        )

    # JAX memory is unavailable in this suite; avoid an extra allocation-heavy
    # execution that would only produce NaN memory fields.
    _, mem_info = measure_memory_jax(run_ewald, jax)
    time_real_sec = float("nan")
    time_reciprocal_sec = float("nan")
    timing_method_real = "not_measured"
    timing_method_reciprocal = "not_measured"
    real_max_min_ratio: float | str = ""
    real_samples_json = ""
    reciprocal_max_min_ratio: float | str = ""
    reciprocal_samples_json = ""
    if profile_components:
        real_samples, timing_method_real = _jax_timing_batches(
            run_real,
            num_runs,
            warmup_runs,
            timing_batches,
        )
        reciprocal_samples, timing_method_reciprocal = _jax_timing_batches(
            run_reciprocal,
            num_runs,
            warmup_runs,
            timing_batches,
        )
        time_real_sec, real_max_min_ratio, real_samples_json = summarize_timing_batches(
            real_samples
        )
        time_reciprocal_sec, reciprocal_max_min_ratio, reciprocal_samples_json = (
            summarize_timing_batches(reciprocal_samples)
        )
    timing_samples, timing_method = _jax_timing_batches(
        run_ewald,
        num_runs,
        warmup_runs,
        timing_batches,
    )
    time_sec, max_min_ratio, samples_json = summarize_timing_batches(timing_samples)
    return {
        "time_seconds": time_sec,
        "time_real_seconds": time_real_sec,
        "time_reciprocal_seconds": time_reciprocal_sec,
        "timing_method": timing_method,
        "timing_method_real": timing_method_real,
        "timing_method_reciprocal": timing_method_reciprocal,
        "component_profiled": profile_components,
        "num_k_vectors_per_system": num_k_vectors_per_system,
        "mem_info": mem_info,
        "timing_batches": timing_batches,
        "timing_batch_aggregation": "arithmetic_mean",
        "timing_batch_max_min_ratio": max_min_ratio,
        "timing_batch_seconds_json": samples_json,
        "timing_batch_real_max_min_ratio": real_max_min_ratio,
        "timing_batch_real_seconds_json": real_samples_json,
        "timing_batch_reciprocal_max_min_ratio": reciprocal_max_min_ratio,
        "timing_batch_reciprocal_seconds_json": reciprocal_samples_json,
    }


# =============================================================================
# run_from_config helpers
# =============================================================================


def _el_tensors_from_data(data, backend):
    """Convert a ``create_system`` dict to f64 tensors/arrays for electrostatics.

    Returns ``(positions, charges, cell, pbc, batch_idx)`` as a tuple. The
    caller drops its reference to ``data`` afterward so the f32 originals can
    be released.
    """
    pbc = data["pbc"]
    batch_idx = data["batch_idx"]
    if backend == "torch":
        positions = data["positions"].to(torch.float64)
        charges = data["charges"].to(torch.float64)
        cell = data["cell"].to(torch.float64)
    else:
        import jax.numpy as jnp

        positions = data["positions"].astype(jnp.float64)
        charges = data["charges"].astype(jnp.float64)
        cell = data["cell"].astype(jnp.float64)
    return positions, charges, cell, pbc, batch_idx


def _el_runtime_pme_input_metadata(
    positions: Any,
    charges: Any,
    cell: Any,
    batch_idx: Any,
    pbc: Any,
    *,
    backend: str,
    jax_api: dict | None,
) -> dict[str, str]:
    """Hash post-conversion PME inputs and require one normalized dtype layout."""
    if backend == "torch":
        arrays = {
            "positions": positions.detach().cpu().numpy(),
            "charges": charges.detach().cpu().numpy(),
            "cell": cell.detach().cpu().numpy(),
            "batch_idx": batch_idx.detach().cpu().numpy(),
            "pbc": pbc.detach().cpu().numpy(),
        }
    elif backend == "jax":
        if jax_api is None:
            raise RuntimeError("JAX PME input validation requires the JAX API")
        device_get = jax_api["jax"].device_get
        arrays = {
            "positions": np.asarray(device_get(positions)),
            "charges": np.asarray(device_get(charges)),
            "cell": np.asarray(device_get(cell)),
            "batch_idx": np.asarray(device_get(batch_idx)),
            "pbc": np.asarray(device_get(pbc)),
        }
    else:
        raise ValueError(f"Unsupported electrostatics backend: {backend}")

    dtype_signature = {name: array.dtype.name for name, array in arrays.items()}
    expected_dtypes = {
        "positions": "float64",
        "charges": "float64",
        "cell": "float64",
        "batch_idx": "int32",
        "pbc": "bool",
    }
    if dtype_signature != expected_dtypes:
        raise TypeError(
            "runtime PME input dtypes do not match the reportable layout: "
            f"expected {expected_dtypes}, found {dtype_signature}"
        )
    return {
        "pme_runtime_input_sha256": _canonical_pme_input_sha256(
            arrays["positions"],
            arrays["charges"],
            arrays["cell"],
            arrays["batch_idx"],
            arrays["pbc"],
        ),
        "pme_runtime_dtype_signature": _canonical_json(dtype_signature),
    }


def _el_estimate_pme_params(
    positions,
    cell,
    batch_idx,
    backend,
    accuracy,
    jax_api,
    max_pme_real_space_cutoff,
    spline_order,
):
    """Preserve upstream cutoff/alpha setup and use the shared PME mesh selector.

    The configured PME cutoff limit and spline order are benchmark methodology.
    """
    if backend == "torch":
        pme_estimator = estimate_pme_parameters
    else:
        pme_estimator = jax_api["estimate_pme_parameters"]

    pme_params = pme_estimator(
        positions,
        cell,
        batch_idx=batch_idx,
        accuracy=accuracy,
        spline_order=spline_order,
    )
    estimated_cutoff = float(pme_params.real_space_cutoff.max())
    if estimated_cutoff > max_pme_real_space_cutoff:
        pme_params = pme_estimator(
            positions,
            cell,
            batch_idx=batch_idx,
            accuracy=accuracy,
            real_space_cutoff=max_pme_real_space_cutoff,
            spline_order=spline_order,
        )
    return pme_params


def _el_estimate_ewald_params(
    positions: Any,
    cell: Any,
    batch_idx: Any,
    backend: str,
    accuracy: float,
    jax_api: dict | None,
) -> tuple[Any, float, float]:
    """Resolve only direct-Ewald parameters on the selected backend."""
    if backend == "torch":
        parameters = estimate_ewald_parameters(
            positions,
            cell,
            batch_idx=batch_idx,
            accuracy=accuracy,
        )
        return (
            parameters.alpha.clone(),
            float(parameters.real_space_cutoff.max().item()),
            float(parameters.reciprocal_space_cutoff.max().item()),
        )
    if backend == "jax":
        if jax_api is None:
            raise RuntimeError("JAX Ewald parameter setup requires the JAX API")
        parameters = jax_api["estimate_ewald_parameters"](
            positions,
            cell,
            batch_idx=batch_idx,
            accuracy=accuracy,
        )
        return (
            parameters.alpha,
            float(parameters.real_space_cutoff.max()),
            float(parameters.reciprocal_space_cutoff.max()),
        )
    raise ValueError(f"Unsupported electrostatics backend: {backend}")


def _el_parameter_metadata(
    method: str,
    alpha: Any,
    real_cutoff: float,
    mesh_dims: tuple[int, int, int],
    *,
    reciprocal_space_cutoff: float | None = None,
) -> dict[str, float | int | str | None]:
    """Return the selected setup parameters needed to audit an EL row."""
    alpha_mean = alpha.mean() if hasattr(alpha, "mean") else alpha
    if hasattr(alpha_mean, "item"):
        alpha_mean = alpha_mean.item()
    metadata: dict[str, float | int | str | None] = {
        "alpha": float(alpha_mean),
        "real_space_cutoff": float(real_cutoff),
    }
    if method == "pme":
        mesh_nx, mesh_ny, mesh_nz = (int(value) for value in mesh_dims)
        metadata.update(
            {
                "mesh_nx": mesh_nx,
                "mesh_ny": mesh_ny,
                "mesh_nz": mesh_nz,
                "reciprocal_grid_points": mesh_nx * mesh_ny * (mesh_nz // 2 + 1),
            }
        )
    elif method == "ewald":
        if reciprocal_space_cutoff is None:
            raise ValueError("Ewald metadata requires reciprocal_space_cutoff")
        metadata["reciprocal_space_cutoff"] = float(reciprocal_space_cutoff)
    else:
        raise ValueError(f"Unsupported electrostatics method: {method}")
    return metadata


def _el_neighbor_capacity(positions, cell, batch_idx, real_cutoff) -> int:
    """Estimate the per-atom neighbor limit used during EL setup."""
    atomic_density = compute_atomic_density(
        {
            "positions": positions,
            "cell": cell,
            "batch_idx": batch_idx,
        }
    )
    return int(
        estimate_max_neighbors(
            real_cutoff,
            atomic_density=atomic_density * DEFAULT_NL_SAFETY_FACTOR,
        )
    )


def _torch_neighbor_conversion_chunk_rows(width: int) -> int:
    """Reuse the existing bounded Torch conversion chunks for both passes.

    The retained int32 conversion cap bounds temporary indexing arrays. It
    affects untimed copies; neighbor selection and physical cutoffs are fixed.
    """
    max_chunk_elements = torch.iinfo(torch.int32).max // 8
    return max(1, max_chunk_elements // max(width, 1))


def _torch_neighbor_matrix_indices_chunked(
    neighbor_matrix: torch.Tensor,
    num_neighbors: torch.Tensor,
    *,
    fill_value: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert Torch neighbor indices to COO/CSR in row chunks.

    PyTorch's ``nonzero``/``where`` rejects tensors with more than INT_MAX
    elements. Large electrostatics reportable rows can exceed that matrix size
    even when the actual neighbor count fits in memory. Chunking avoids that
    conversion API ceiling without skipping atoms, reducing the benchmark grid,
    or changing the neighbor-list semantics.
    """
    if num_neighbors.shape[0] == 0:
        neighbor_list = torch.zeros(
            2,
            0,
            dtype=neighbor_matrix.dtype,
            device=neighbor_matrix.device,
        )
        neighbor_ptr = torch.zeros(1, dtype=torch.int32, device=neighbor_matrix.device)
        return neighbor_list, neighbor_ptr

    max_found = int(num_neighbors.max().item())
    if max_found > neighbor_matrix.shape[1]:
        raise ValueError(
            f"neighbor matrix width {neighbor_matrix.shape[1]} is smaller than "
            f"observed max neighbors {max_found}"
        )

    neighbor_ptr = torch.zeros(
        num_neighbors.shape[0] + 1,
        dtype=torch.int32,
        device=neighbor_matrix.device,
    )
    torch.cumsum(num_neighbors, dim=0, out=neighbor_ptr[1:])
    total_pairs = int(neighbor_ptr[-1].item())
    neighbor_list = torch.empty(
        (2, total_pairs),
        dtype=neighbor_matrix.dtype,
        device=neighbor_matrix.device,
    )
    chunk_rows = _torch_neighbor_conversion_chunk_rows(int(neighbor_matrix.shape[1]))

    for start in range(0, neighbor_matrix.shape[0], chunk_rows):
        end = min(start + chunk_rows, neighbor_matrix.shape[0])
        out_start = int(neighbor_ptr[start].item())
        out_end = int(neighbor_ptr[end].item())
        if out_start == out_end:
            continue

        matrix_chunk = neighbor_matrix[start:end]
        active = matrix_chunk != fill_value
        rows, _ = torch.where(active)
        neighbor_list[0, out_start:out_end] = rows.to(neighbor_matrix.dtype) + start
        neighbor_list[1, out_start:out_end] = matrix_chunk[active].to(
            neighbor_matrix.dtype
        )
    return neighbor_list, neighbor_ptr


def _torch_neighbor_shift_matrix_to_list_chunked(
    neighbor_shift_matrix: torch.Tensor,
    num_neighbors: torch.Tensor,
    neighbor_ptr: torch.Tensor,
) -> torch.Tensor:
    """Copy packed atom-centric row shifts after dense indices are released.

    The public atom-centric builder writes every accepted pair to the next row
    slot. Its count identifies the complete active prefix, including repeated
    periodic images, in the same order used by the compact indices.
    """
    total_pairs = int(neighbor_ptr[-1].item())
    shifts = torch.empty(
        (total_pairs, 3),
        dtype=neighbor_shift_matrix.dtype,
        device=neighbor_shift_matrix.device,
    )
    width = int(neighbor_shift_matrix.shape[1])
    chunk_rows = _torch_neighbor_conversion_chunk_rows(width)
    columns = torch.arange(width, device=num_neighbors.device, dtype=torch.int32)
    for start in range(0, len(num_neighbors), chunk_rows):
        end = min(start + chunk_rows, len(num_neighbors))
        out_start = int(neighbor_ptr[start].item())
        out_end = int(neighbor_ptr[end].item())
        if out_start != out_end:
            active = columns[None, :] < num_neighbors[start:end, None]
            shifts[out_start:out_end] = neighbor_shift_matrix[start:end][active]
    return shifts


def _el_build_nl(
    positions,
    cell,
    pbc,
    batch_idx,
    real_cutoff,
    backend,
    jax_api,
    configured_max_neighbors=None,
):
    """Build the real-space neighbor list in LIST (COO + ptr) format."""
    maxnb = (
        _el_neighbor_capacity(positions, cell, batch_idx, real_cutoff)
        if configured_max_neighbors is None
        else int(configured_max_neighbors)
    )
    if backend == "torch":
        neighbor_matrix, num_neighbors, neighbor_matrix_shifts = batch_cell_list(
            positions=positions,
            cutoff=real_cutoff,
            batch_idx=batch_idx,
            pbc=pbc,
            cell=cell,
            max_neighbors=maxnb,
            return_neighbor_list=False,
            strategy="atom_centric",
            atom_centric_path="direct",
        )
        neighbor_list, neighbor_ptr = _torch_neighbor_matrix_indices_chunked(
            neighbor_matrix,
            num_neighbors,
            fill_value=positions.shape[0],
        )
        del neighbor_matrix
        neighbor_shifts = _torch_neighbor_shift_matrix_to_list_chunked(
            neighbor_matrix_shifts, num_neighbors, neighbor_ptr
        )
        return neighbor_list, neighbor_ptr, neighbor_shifts
    jnp = jax_api["jnp"]
    num_systems = int(cell.shape[0]) if cell.ndim == 3 else 1
    atoms_per_system = positions.shape[0] // max(num_systems, 1)
    batch_ptr = jnp.arange(num_systems + 1, dtype=jnp.int32) * int(atoms_per_system)
    max_total_cells, _, _ = jax_api["estimate_batch_cell_list_sizes"](
        positions=positions,
        batch_ptr=batch_ptr,
        cell=cell,
        cutoff=float(real_cutoff),
        pbc=pbc,
    )

    def build_nl_matrix(pos, cell_, pbc_, batch_idx_, batch_ptr_):
        return jax_api["neighbor_list"](
            positions=pos,
            cutoff=float(real_cutoff),
            cell=cell_,
            pbc=pbc_,
            batch_idx=batch_idx_,
            batch_ptr=batch_ptr_,
            method="batch_cell_list_atom_centric",
            return_neighbor_list=False,
            atom_centric_path="direct",
            max_neighbors=int(maxnb),
            max_total_cells=int(max_total_cells),
        )

    neighbor_matrix, num_neighbors, neighbor_matrix_shifts = jax_api["jax"].jit(
        build_nl_matrix
    )(
        positions,
        cell,
        pbc,
        batch_idx,
        batch_ptr,
    )
    jax_api["jax"].block_until_ready(neighbor_matrix)
    from nvalchemiops.jax.neighbors.neighbor_utils import (
        get_neighbor_list_from_neighbor_matrix,
    )

    nl = get_neighbor_list_from_neighbor_matrix(
        neighbor_matrix,
        num_neighbors,
        neighbor_shift_matrix=neighbor_matrix_shifts,
        fill_value=positions.shape[0],
    )
    jax_api["jax"].block_until_ready(nl[0])
    return nl


def _el_csr_neighbor_counts(
    nl_data,
    nl_ptr,
    backend,
    jax_api,
    expected_atoms: int,
) -> list[int]:
    """Validate a compact neighbor list and return per-atom counts."""
    if len(nl_data.shape) != 2 or int(nl_data.shape[0]) != 2:
        raise ValueError(f"neighbor_list must have shape (2, P), got {nl_data.shape}")
    if len(nl_ptr.shape) != 1:
        raise ValueError(f"neighbor_ptr must be one-dimensional, got {nl_ptr.shape}")
    if backend == "torch":
        ptr_values = nl_ptr.detach().cpu().tolist()
    else:
        ptr_values = jax_api["jax"].device_get(nl_ptr).tolist()
    ptr_values = [int(value) for value in ptr_values]

    if not ptr_values or ptr_values[0] != 0:
        raise ValueError("neighbor_ptr must start at zero")
    if len(ptr_values) != int(expected_atoms) + 1:
        raise ValueError(
            "neighbor_ptr must contain one boundary per atom: "
            f"got {len(ptr_values)}, expected {int(expected_atoms) + 1}"
        )
    if any(stop < start for start, stop in zip(ptr_values, ptr_values[1:])):
        raise ValueError("neighbor_ptr must be nondecreasing")

    observed_pairs = int(nl_data.shape[1])
    if ptr_values[-1] != observed_pairs:
        raise ValueError(
            "neighbor_ptr must end at the compact neighbor-list size: "
            f"ptr[-1]={ptr_values[-1]}, pairs={observed_pairs}"
        )
    return [stop - start for start, stop in zip(ptr_values, ptr_values[1:])]


# =============================================================================
# Config-Driven Runner
# =============================================================================


class ElConfigSetup(NamedTuple):
    """Method-specific return shape of :func:`_el_setup_config`.

    Named over positional unpacking. The fields are a mix of
    backend-polymorphic tensors (``inputs``, ``alpha``) and plain scalars
    that the method loop consumes.
    """

    inputs: ElectrostaticsInputs
    alpha: Any  # torch 0-dim tensor (torch) or python float (jax)
    real_cutoff: float
    mesh_dims: tuple[int, int, int]
    k_cutoff: float
    spline_order: int | None
    atoms_per_system: int
    batch_size: int
    actual_total: int
    row_metadata: dict[str, Any]


class ElConfigFailure(NamedTuple):
    """Expected setup failure that still needs a CSV row."""

    error: str
    error_type: str
    failure_stage: str
    row_metadata: dict[str, Any]


def _ordered_configs_for_backend(configs: list[dict], sys_name: str, backend: str):
    """Order configs for stable memory behavior without changing planned rows."""
    if backend != "jax":
        return configs
    return sorted(
        configs,
        key=lambda cfg: planned_atom_counts(sys_name, cfg)[2],
        reverse=True,
    )


def _el_setup_config(
    cfg: dict,
    sys_name: str,
    accuracy: float,
    method: str,
    backend: str,
    jax_api: dict | None,
    max_pme_real_space_cutoff: float,
    spline_order: int,
) -> ElConfigSetup | ElConfigFailure:
    """Build backend inputs using upstream cutoff/alpha setup and the PME mesh selector.

    Returns an ``ElConfigFailure`` on expected failures (create_system error,
    params estimation failure, NL build failure) after printing a diagnostic
    line so callers can still emit ``success=False`` CSV rows.
    """
    n, bs = cfg["num_atoms"], cfg["batch_size"]
    try:
        data = create_system(
            sys_name,
            num_atoms=n,
            pdb_path=cfg.get("pdb_path"),
            batch_size=bs,
            backend=backend,
            cscl_replication_factors=cfg.get("cscl_replication_factors"),
        )
        row_metadata = benchmark_system_metadata(data, sys_name)
    except Exception as e:
        error_type = failure_error_type(e)
        if error_type == "OutOfMemoryError":
            clean_gpu()
            if backend == "jax":
                clean_jax(clear_executables=True)
        print(f"    FAILED: {e}")
        return ElConfigFailure(str(e), error_type, "system_setup", {})

    atoms_per_system = data["atoms_per_system"]
    actual_total = data.get("total_atoms", atoms_per_system)
    batch_size = data.get("batch_size", 1)
    print(
        f"\n  {format_num(atoms_per_system)} atoms × {batch_size} batch = "
        f"{format_num(actual_total)} total  [GPU: {current_alloc_gb(backend):.1f} GB allocated]"
    )

    positions, charges, cell, pbc, batch_idx = _el_tensors_from_data(data, backend)
    del data

    runtime_pme_metadata: dict[str, str] = {}
    try:
        if method == "pme":
            pme_params = _el_estimate_pme_params(
                positions,
                cell,
                batch_idx,
                backend,
                accuracy,
                jax_api,
                max_pme_real_space_cutoff,
                spline_order,
            )
            alpha = pme_params.alpha
            real_cutoff = float(pme_params.real_space_cutoff.max())
            mesh_dims = tuple(int(value) for value in pme_params.mesh_dimensions)
            resolved_spline_order = spline_order
            k_cutoff = 0.0
            runtime_pme_metadata = _el_runtime_pme_input_metadata(
                positions,
                charges,
                cell,
                batch_idx,
                pbc,
                backend=backend,
                jax_api=jax_api,
            )
        else:
            alpha, real_cutoff, k_cutoff = _el_estimate_ewald_params(
                positions,
                cell,
                batch_idx,
                backend,
                accuracy,
                jax_api,
            )
            mesh_dims = (0, 0, 0)
            resolved_spline_order = None
    except Exception as e:
        error_type = failure_error_type(e)
        if error_type == "OutOfMemoryError":
            clean_gpu()
            if backend == "jax":
                clean_jax(clear_executables=True)
        print(f"    FAILED (params): {e}")
        return ElConfigFailure(str(e), error_type, "parameter_setup", row_metadata)

    row_metadata.update(
        _el_parameter_metadata(
            method,
            alpha,
            real_cutoff,
            mesh_dims,
            reciprocal_space_cutoff=k_cutoff,
        )
    )
    if method == "pme":
        row_metadata["spline_order"] = resolved_spline_order
        row_metadata.update(runtime_pme_metadata)
        row_metadata["pme_cutoff_policy"] = "capped_upstream_estimate"
        row_metadata["pme_mesh_policy"] = "bounded_power_two_then_four_fft_snap"
        row_metadata["pme_fft_padding_fraction"] = _DEFAULT_PME_FFT_PADDING_FRACTION
        row_metadata["pme_max_real_space_cutoff"] = max_pme_real_space_cutoff

    try:
        configured_max_neighbors = _el_neighbor_capacity(
            positions, cell, batch_idx, real_cutoff
        )
        row_metadata["configured_max_neighbors"] = configured_max_neighbors
        row_metadata["neighbor_capacity_safety_factor"] = DEFAULT_NL_SAFETY_FACTOR
        nl_data, nl_ptr, nl_shifts = _el_build_nl(
            positions,
            cell,
            pbc,
            batch_idx,
            real_cutoff,
            backend,
            jax_api,
            configured_max_neighbors,
        )
        neighbor_counts = _el_csr_neighbor_counts(
            nl_data,
            nl_ptr,
            backend,
            jax_api,
            expected_atoms=int(positions.shape[0]),
        )
        row_metadata.update(
            neighbor_count_metadata(neighbor_counts, configured_max_neighbors)
        )
    except Exception as e:
        error_type = failure_error_type(e)
        if error_type == "OutOfMemoryError":
            clean_gpu()
            if backend == "jax":
                clean_jax(clear_executables=True)
        print(f"    FAILED (NL): {e}")
        return ElConfigFailure(
            str(e),
            error_type,
            "neighbor_list_setup",
            row_metadata,
        )

    inputs = ElectrostaticsInputs(
        positions=positions,
        charges=charges,
        cell=cell,
        pbc=pbc,
        batch_idx=batch_idx,
        nl_data=nl_data,
        nl_shifts=nl_shifts,
        nl_ptr=nl_ptr,
        backend=backend,
        max_atoms_per_system=int(atoms_per_system),
    )
    return ElConfigSetup(
        inputs=inputs,
        alpha=alpha,
        real_cutoff=real_cutoff,
        mesh_dims=mesh_dims,
        k_cutoff=k_cutoff,
        spline_order=resolved_spline_order,
        atoms_per_system=atoms_per_system,
        batch_size=batch_size,
        actual_total=actual_total,
        row_metadata=row_metadata,
    )


def _el_run_method(
    method,
    inputs,
    alpha,
    mesh_dims,
    k_cutoff,
    spline_order,
    accuracy,
    num_runs,
    warmup_runs,
    timing_batches,
    profile_components,
    jax_api,
    row_meta,
):
    """Run one method's energy, force, and charge-gradient workload.

    Catches OOM, ``NotImplementedError``, and other exceptions as
    ``success=False`` rows for CSV visibility and failed-case plot markers.
    ``row_meta`` carries the identity fields for :func:`build_result`.
    """
    label = method.upper()
    if timing_batches <= 0:
        raise ValueError("EL timing_batches must be positive")
    try:
        if method == "pme":
            r = benchmark_pme(
                inputs,
                alpha,
                mesh_dims,
                spline_order,
                accuracy,
                num_runs,
                warmup_runs,
                jax_api=jax_api,
                profile_components=profile_components,
                timing_batches=timing_batches,
            )
        else:
            r = benchmark_ewald(
                inputs,
                alpha,
                k_cutoff,
                accuracy,
                num_runs,
                warmup_runs,
                jax_api=jax_api,
                profile_components=profile_components,
                timing_batches=timing_batches,
            )
        runtime_parameter_metadata = {}
        if method == "ewald":
            runtime_parameter_metadata["num_k_vectors_per_system"] = int(
                r["num_k_vectors_per_system"]
            )
        result = build_result(
            method=method,
            time_seconds=r["time_seconds"],
            mem_info=r["mem_info"],
            accuracy=accuracy,
            timing_runs=num_runs,
            warmup_runs=warmup_runs,
            timing_batches=r["timing_batches"],
            timing_batch_aggregation=r["timing_batch_aggregation"],
            timing_batch_max_min_ratio=r["timing_batch_max_min_ratio"],
            timing_batch_seconds_json=r["timing_batch_seconds_json"],
            timing_batch_real_max_min_ratio=r.get(
                "timing_batch_real_max_min_ratio", ""
            ),
            timing_batch_real_seconds_json=r.get("timing_batch_real_seconds_json", ""),
            timing_batch_reciprocal_max_min_ratio=r.get(
                "timing_batch_reciprocal_max_min_ratio", ""
            ),
            timing_batch_reciprocal_seconds_json=r.get(
                "timing_batch_reciprocal_seconds_json", ""
            ),
            timing_method=r.get("timing_method"),
            timing_method_real=r.get("timing_method_real"),
            timing_method_reciprocal=r.get("timing_method_reciprocal"),
            pme_cache_mode=r.get("pme_cache_mode", ""),
            time_real_us_per_atom=(
                (r["time_real_seconds"] * 1e6) / row_meta["total_atoms"]
                if row_meta["total_atoms"] > 0
                else 0.0
            ),
            time_reciprocal_us_per_atom=(
                (r["time_reciprocal_seconds"] * 1e6) / row_meta["total_atoms"]
                if row_meta["total_atoms"] > 0
                else 0.0
            ),
            **runtime_parameter_metadata,
            **_energy_derivative_metadata(profile_components),
            **row_meta,
        )
        mem_suffix = (
            f" | {result['mem_peak_gb']:.1f} GB" if inputs.backend == "torch" else ""
        )
        print(f"    {label:10s}: {result['time_us_per_atom']:.3f} μs/atom{mem_suffix}")
        return result
    except NotImplementedError as e:
        print(f"    {label:10s}: UNSUPPORTED - {e}")
        return build_failure_result(
            method=method,
            accuracy=accuracy,
            timing_runs=num_runs,
            warmup_runs=warmup_runs,
            timing_batches=timing_batches,
            timing_batch_aggregation="arithmetic_mean",
            error=str(e),
            error_type=type(e).__name__,
            failure_stage="method_unsupported",
            **_energy_derivative_metadata(profile_components),
            **row_meta,
        )
    except torch.cuda.OutOfMemoryError as e:
        print(f"    {label:10s}: OOM - {e}")
        clean_gpu()
        return build_failure_result(
            method=method,
            accuracy=accuracy,
            error="CUDA out of memory",
            error_type="OutOfMemoryError",
            failure_stage=f"{method}_timing",
            timing_runs=num_runs,
            warmup_runs=warmup_runs,
            timing_batches=timing_batches,
            timing_batch_aggregation="arithmetic_mean",
            **_energy_derivative_metadata(profile_components),
            **row_meta,
        )
    except Exception as e:
        error_type = failure_error_type(e)
        if error_type == "OutOfMemoryError":
            clean_gpu()
            if inputs.backend == "jax":
                clean_jax(clear_executables=True)
        print(f"    {label:10s}: FAILED - {e}")
        return build_failure_result(
            method=method,
            accuracy=accuracy,
            error=str(e),
            error_type=error_type,
            failure_stage=f"{method}_timing",
            timing_runs=num_runs,
            warmup_runs=warmup_runs,
            timing_batches=timing_batches,
            timing_batch_aggregation="arithmetic_mean",
            **_energy_derivative_metadata(profile_components),
            **row_meta,
        )


def dry_run_from_config(config: dict, backend: str | None = None) -> list[dict]:
    """List every configured electrostatics case without GPU setup."""
    params = config["parameters"]
    max_total_atoms = params.get("max_total_atoms")
    profile_components = bool(params.get("profile_components", False))
    accuracies = [float(value) for value in config["accuracies"]]
    method_names = [m["name"] for m in config["methods"] if m.get("enabled", True)]
    plan_output = config.get("runtime", {}).get("plan_output", "dry_run")
    if backend is None:
        backend = config.get("runtime", {}).get("backend", "torch")
    prepared = _prepare_el_families(config, plan_only=True)
    rows = []
    for sys_name, sys_config in config["systems"].items():
        if not sys_config.get("enabled", True):
            continue
        for mode_name, mode_config in config["scaling"].items():
            if not isinstance(mode_config, dict) or not mode_config.get(
                "enabled", True
            ):
                continue
            configs = list(prepared[(sys_name, mode_name)])
            configs, skipped = filter_configs_by_total_atoms(
                configs, sys_name, max_total_atoms
            )
            planned_configs = [
                (cfg, f">{max_total_atoms} max_total_atoms") for cfg, _ in skipped
            ] + [(cfg, "") for cfg in configs]
            for cfg, reason in planned_configs:
                atoms_per_system, batch_size, total_atoms = planned_atom_counts(
                    sys_name, cfg
                )
                fixed_total_row = fixed_total_metadata(
                    mode_name,
                    mode_config,
                    total_atoms,
                )
                rows.extend(
                    {
                        "benchmark": "el",
                        "backend": backend,
                        "system": sys_name,
                        "mode": mode_name,
                        "atoms_per_system": atoms_per_system,
                        "batch_size": batch_size,
                        "total_atoms": total_atoms,
                        **fixed_total_row,
                        "method": method,
                        "accuracy": accuracy,
                        **_energy_derivative_metadata(profile_components),
                        "reason": reason,
                    }
                    for accuracy in accuracies
                    for method in method_names
                )
    if plan_output != "count":
        print("EL dry-run plan")
        for row in rows:
            suffix = f" SKIP {row['reason']}" if row["reason"] else ""
            print(
                "  {system}/{mode} backend={backend} method={method} "
                "accuracy={accuracy} workload={workload} N={atoms_per_system} "
                "batch={batch_size} total={total_atoms}{suffix}".format(
                    **row, suffix=suffix
                )
            )
    print(f"EL dry-run rows: {len(rows)}")
    return rows


def run_from_config(
    config: dict,
    output_dir: Path | str | None = None,
    backend: str | None = None,
    config_dir: Path | str | None = None,
) -> list[dict]:
    """Run electrostatics benchmarks driven by YAML config.

    Parameters
    ----------
    backend : str, optional
        ``'torch'`` or ``'jax'``. Pulled from ``config['runtime']['backend']``
        when None. Default is ``'torch'``.
    config_dir : Path or str, optional
        Retained for compatibility with the unified-suite runner.
    """
    params = config["parameters"]
    num_runs = params["timing_runs"]
    warmup_runs = params["warmup_runs"]
    timing_batches = int(params["timing_batches"])
    if timing_batches <= 0:
        raise ValueError("timing_batches must be positive")
    profile_components = bool(params.get("profile_components", False))
    accuracies = config["accuracies"]
    max_total_atoms = params.get("max_total_atoms")
    methods_config = config["methods"]
    method_names = [m["name"] for m in methods_config if m.get("enabled", True)]
    replace_methods = (
        method_names if config.get("runtime", {}).get("explicit_methods") else None
    )

    if backend is None:
        backend = config.get("runtime", {}).get("backend", "torch")
    if backend == "warp":
        raise ValueError(
            "Electrostatics benchmark supports torch and jax backends, not warp."
        )
    if config.get("runtime", {}).get("dry_run", False):
        return dry_run_from_config(config, backend=backend)

    prepared = _prepare_el_families(config)
    max_pme_real_space_cutoff = _configured_pme_cutoff(params)
    pme_spline_order = next(
        (
            method["spline_order"]
            for method in methods_config
            if method["name"] == "pme"
        ),
        None,
    )

    if output_dir is None:
        output_dir = create_run_directory(config["output"]["base_dir"], prefix="el")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_input_provenance(
        configured_nh3_artifacts(config),
        metadata_values={"benchmark": "el"},
    )
    jax_api = lazy_import_jax(need_electrostatics=True) if backend == "jax" else None

    try:
        gpu_name = torch.cuda.get_device_name(0)
    except (AssertionError, RuntimeError):
        gpu_name = "N/A (no CUDA)"
    print(f"Electrostatics Benchmark | GPU: {gpu_name}")
    print(f"Backend: {backend}")
    print(f"Methods: {method_names} | Accuracies: {accuracies}")
    print(f"Maximum PME real-space cutoff: {max_pme_real_space_cutoff:g} Angstrom")
    print("Direct Ewald parameters: automatic")
    print(
        f"Timing: {timing_batches} batches × {num_runs} calls | "
        f"component profiling: {profile_components}"
    )
    print(f"Output: {output_dir}")

    all_results = []

    for sys_name, sys_config in config["systems"].items():
        if not sys_config.get("enabled", True):
            continue

        for mode_name, mode_config in config["scaling"].items():
            if not isinstance(mode_config, dict) or not mode_config.get(
                "enabled", True
            ):
                continue
            print(f"\n{'=' * 70}")
            print(f"ELECTROSTATICS: {sys_name.upper()} / {mode_name}")
            print(f"{'=' * 70}")

            configs = list(prepared[(sys_name, mode_name)])
            configs, skipped = filter_configs_by_total_atoms(
                configs, sys_name, max_total_atoms
            )
            configs = _ordered_configs_for_backend(configs, sys_name, backend)
            results = []
            csv_name = make_csv_name("el", sys_name, mode_name)
            csv_path = output_dir / csv_name

            def save_progress() -> None:
                """Atomically preserve rows completed in this CSV shard."""
                if results:
                    save_results(
                        results,
                        csv_path,
                        replace_backend=backend,
                        replace_methods=replace_methods,
                    )

            for cfg, skipped_total in skipped:
                print(
                    f"  SKIP total atoms {format_num(skipped_total)} "
                    f"(>{format_num(max_total_atoms)})"
                )
                atoms_per_system, batch_size, total_atoms = planned_atom_counts(
                    sys_name, cfg
                )
                row_meta = _el_row_meta(
                    sys_name,
                    mode_name,
                    mode_config,
                    backend,
                    atoms_per_system,
                    batch_size,
                    total_atoms,
                )
                reason = f">{max_total_atoms} max_total_atoms"
                results.extend(
                    build_skipped_result(
                        method=method,
                        accuracy=accuracy,
                        reason=reason,
                        timing_runs=num_runs,
                        warmup_runs=warmup_runs,
                        timing_batches=timing_batches,
                        timing_batch_aggregation="arithmetic_mean",
                        **_energy_derivative_metadata(profile_components),
                        **row_meta,
                    )
                    for accuracy in accuracies
                    for method in method_names
                )
            save_progress()
            if not configs:
                if results:
                    save_progress()
                    all_results.extend(results)
                continue
            for accuracy in accuracies:
                print(f"\n  --- Accuracy: {accuracy:.0e} ---")

                for cfg in configs:
                    atoms_per_system, batch_size, total_atoms = planned_atom_counts(
                        sys_name, cfg
                    )
                    row_meta = _el_row_meta(
                        sys_name,
                        mode_name,
                        mode_config,
                        backend,
                        atoms_per_system,
                        batch_size,
                        total_atoms,
                    )
                    for idx, method in enumerate(method_names):
                        if backend == "jax":
                            clean_jax()
                        clean_gpu()
                        setup = _el_setup_config(
                            cfg,
                            sys_name,
                            accuracy,
                            method,
                            backend,
                            jax_api,
                            max_pme_real_space_cutoff,
                            pme_spline_order,
                        )
                        if isinstance(setup, ElConfigFailure):
                            result = build_failure_result(
                                method=method,
                                accuracy=accuracy,
                                error=setup.error,
                                error_type=setup.error_type,
                                failure_stage=setup.failure_stage,
                                timing_runs=num_runs,
                                warmup_runs=warmup_runs,
                                timing_batches=timing_batches,
                                timing_batch_aggregation="arithmetic_mean",
                                **_energy_derivative_metadata(profile_components),
                                **setup.row_metadata,
                                **row_meta,
                            )
                            results.append(result)
                            save_progress()
                            if backend == "jax":
                                clean_jax(
                                    clear_executables=(
                                        setup.error_type == "OutOfMemoryError"
                                    )
                                )
                            if (
                                backend == "jax"
                                and setup.error_type == "OutOfMemoryError"
                            ):
                                results.extend(
                                    build_failure_result(
                                        method=next_method,
                                        accuracy=accuracy,
                                        error=(
                                            "Previous JAX method OOM for this "
                                            "configuration; remaining methods "
                                            "were not executed in the same process."
                                        ),
                                        error_type="SkippedAfterOOM",
                                        failure_stage="post_oom_containment",
                                        timing_runs=num_runs,
                                        warmup_runs=warmup_runs,
                                        timing_batches=timing_batches,
                                        timing_batch_aggregation="arithmetic_mean",
                                        **_energy_derivative_metadata(
                                            profile_components
                                        ),
                                        **row_meta,
                                    )
                                    for next_method in method_names[idx + 1 :]
                                )
                                save_progress()
                                break
                            continue

                        print(
                            f"    {method.upper()} params: "
                            f"alpha={float(setup.alpha.mean()):.4f}, "
                            f"r_cut={setup.real_cutoff:.2f}Å, "
                            f"NL pairs={setup.inputs.nl_data.shape[1]:,}"
                        )
                        row_meta = _el_row_meta(
                            sys_name,
                            mode_name,
                            mode_config,
                            backend,
                            setup.atoms_per_system,
                            setup.batch_size,
                            setup.actual_total,
                        )
                        method_row_meta = {**row_meta, **setup.row_metadata}
                        result = _el_run_method(
                            method,
                            setup.inputs,
                            setup.alpha,
                            setup.mesh_dims,
                            setup.k_cutoff,
                            setup.spline_order,
                            accuracy,
                            num_runs,
                            warmup_runs,
                            timing_batches,
                            profile_components,
                            jax_api,
                            method_row_meta,
                        )
                        if result is not None:
                            results.append(result)
                            save_progress()
                        del setup
                        if (
                            backend == "jax"
                            and result is not None
                            and result.get("success") is False
                            and result.get("error_type") == "OutOfMemoryError"
                        ):
                            clean_jax(clear_executables=True)
                            results.extend(
                                build_failure_result(
                                    method=next_method,
                                    accuracy=accuracy,
                                    error=(
                                        "Previous JAX method OOM for this "
                                        "configuration; remaining methods "
                                        "were not executed in the same process."
                                    ),
                                    error_type="SkippedAfterOOM",
                                    failure_stage="post_oom_containment",
                                    timing_runs=num_runs,
                                    warmup_runs=warmup_runs,
                                    timing_batches=timing_batches,
                                    timing_batch_aggregation="arithmetic_mean",
                                    **_energy_derivative_metadata(profile_components),
                                    **row_meta,
                                )
                                for next_method in method_names[idx + 1 :]
                            )
                            save_progress()
                            break
                        if backend == "jax":
                            clean_jax()

            if results:
                save_progress()
                all_results.extend(results)

    print(f"\nCOMPLETE: {len(all_results)} results in {output_dir}")
    return all_results


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    """Parse command-line arguments for electrostatics benchmarks."""
    parser = argparse.ArgumentParser(
        description="Electrostatics Benchmark (2 systems × 3 modes)",
        allow_abbrev=False,
    )
    parser.add_argument("--config", type=Path, required=True)
    add_common_cli_args(parser, backends=("torch", "jax"))
    parser.add_argument(
        "--accuracies",
        "-a",
        type=float,
        nargs="+",
        default=None,
        help="Override the configured per-method accuracy targets",
    )
    parser.add_argument(
        "--profile-components",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Also time real- and reciprocal-space components separately. "
            "Disabled by default; the reportable timing is the full workload."
        ),
    )
    return parser.parse_args()


def main():
    """Run electrostatics benchmarks."""
    args = parse_args()
    config = load_yaml_config(args.config)
    config = merge_cli_overrides(config, args)

    backend = args.backend or config.get("runtime", {}).get("backend", "torch")
    results = run_from_config(
        config,
        output_dir=args.output_dir,
        backend=backend,
        config_dir=args.config.parent,
    )
    if not results:
        return 1
    if not any(row.get("success", True) is not False for row in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
