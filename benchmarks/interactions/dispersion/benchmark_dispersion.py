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

"""Benchmark dispersion PME (LJ-PME), modeled on ``benchmark_electrostatics.py``.

Runs a size-scaling sweep and a batched sweep for the dispersion reciprocal-space
term and the full dispersion PME (real + reciprocal), on the torch and/or JAX
backends, and writes CSV results.

Examples
--------
::

    python benchmark_dispersion.py --backend torch --sizes 1000 4000 16000
    python benchmark_dispersion.py --backend jax --output results.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from benchmarks.systems import create_random_system
from benchmarks.utils import BenchmarkTimer, save_benchmark_results

DEFAULT_SIZES = [500, 1000, 2000, 4000, 8000, 16000]
DEFAULT_BATCH_SIZES = [2, 4, 8, 16]
DENSITY = 0.1  # atoms / Angstrom^3 (sparse enough to bound neighbor counts)
MAX_CUTOFF = 6.0  # cap real-space cutoff so neighbor lists stay bounded


def _assign_lj_params(num_atoms, device, dtype, seed=0):
    """Random per-atom LJ sigma/epsilon (Ar-like ranges)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    sigma = (3.0 + 0.6 * torch.rand(num_atoms, generator=g, dtype=torch.float64)).to(
        device=device, dtype=dtype
    )
    epsilon = (0.2 + 0.3 * torch.rand(num_atoms, generator=g, dtype=torch.float64)).to(
        device=device, dtype=dtype
    )
    return sigma, epsilon


def _build_torch_inputs(num_atoms, device, dtype, n_systems=1):
    """Build positions/sigma/epsilon/cell/neighbors for a (batched) system."""
    from nvalchemiops.torch.interactions.dispersion import (
        estimate_dispersion_pme_parameters,
    )
    from nvalchemiops.torch.neighbors import neighbor_list as neighbor_list_fn

    systems = []
    for s in range(n_systems):
        sysd = create_random_system(
            num_atoms, density=DENSITY, device=device, dtype=dtype
        )
        systems.append(sysd)

    positions = torch.cat([s["positions"] for s in systems], dim=0)
    cells = torch.cat([s["cell"] for s in systems], dim=0)
    sigma, epsilon = _assign_lj_params(num_atoms * n_systems, device, dtype)
    batch_idx = None
    if n_systems > 1:
        batch_idx = torch.cat(
            [
                torch.full((num_atoms,), s, dtype=torch.int32, device=device)
                for s in range(n_systems)
            ]
        )

    # Cap the real-space cutoff so neighbor counts stay bounded, then derive a
    # matching alpha/mesh from that cutoff.
    box = float(systems[0]["box_size"])
    rc = min(MAX_CUTOFF, 0.45 * box)
    params = estimate_dispersion_pme_parameters(
        positions, cells, batch_idx=batch_idx, accuracy=1e-5, real_space_cutoff=rc
    )
    pbc = torch.tensor([True, True, True], device=device)
    nl, nptr, nsh = neighbor_list_fn(
        positions,
        cutoff=rc,
        cell=cells,
        pbc=pbc,
        batch_idx=batch_idx,
        return_neighbor_list=True,
    )
    return {
        "positions": positions,
        "sigma": sigma,
        "epsilon": epsilon,
        "cell": cells,
        "alpha": params.alpha,
        "mesh_dimensions": tuple(params.mesh_dimensions),
        "real_space_cutoff": rc,
        "batch_idx": batch_idx,
        "neighbor_list": nl,
        "neighbor_ptr": nptr,
        "neighbor_shifts": nsh,
    }


def _make_torch_callables(inp):
    from nvalchemiops.torch.interactions.dispersion import (
        dispersion_pme,
        dispersion_reciprocal_space,
    )

    def recip():
        return dispersion_reciprocal_space(
            inp["positions"],
            inp["sigma"],
            inp["epsilon"],
            inp["cell"],
            alpha=inp["alpha"],
            mesh_dimensions=inp["mesh_dimensions"],
            batch_idx=inp["batch_idx"],
            compute_forces=True,
        )

    def pme():
        return dispersion_pme(
            inp["positions"],
            inp["sigma"],
            inp["epsilon"],
            inp["cell"],
            alpha=inp["alpha"],
            mesh_dimensions=inp["mesh_dimensions"],
            batch_idx=inp["batch_idx"],
            real_space_cutoff=inp["real_space_cutoff"],
            neighbor_list=inp["neighbor_list"],
            neighbor_ptr=inp["neighbor_ptr"],
            neighbor_shifts=inp["neighbor_shifts"],
            compute_forces=True,
        )

    return {"dispersion_reciprocal": recip, "dispersion_pme": pme}


def _make_jax_callables(inp):
    import jax.numpy as jnp

    from nvalchemiops.jax.interactions.dispersion import (
        dispersion_pme,
        dispersion_reciprocal_space,
    )

    def _j(t):
        return None if t is None else jnp.asarray(t.cpu().numpy())

    jp, jsig, jeps, jcell = (
        _j(inp["positions"]),
        _j(inp["sigma"]),
        _j(inp["epsilon"]),
        _j(inp["cell"]),
    )
    jalpha = _j(inp["alpha"])
    jbidx = _j(inp["batch_idx"])
    jnl, jnptr, jnsh = (
        _j(inp["neighbor_list"]),
        _j(inp["neighbor_ptr"]),
        _j(inp["neighbor_shifts"]),
    )

    def recip():
        return dispersion_reciprocal_space(
            jp,
            jsig,
            jeps,
            jcell,
            alpha=jalpha,
            mesh_dimensions=inp["mesh_dimensions"],
            batch_idx=jbidx,
            compute_forces=True,
        )

    def pme():
        return dispersion_pme(
            jp,
            jsig,
            jeps,
            jcell,
            alpha=jalpha,
            mesh_dimensions=inp["mesh_dimensions"],
            batch_idx=jbidx,
            real_space_cutoff=inp["real_space_cutoff"],
            neighbor_list=jnl,
            neighbor_ptr=jnptr,
            neighbor_shifts=jnsh,
            compute_forces=True,
        )

    return {"dispersion_reciprocal": recip, "dispersion_pme": pme}


def run(backend, sizes, batch_sizes, dtype, device, output):
    timer = BenchmarkTimer(backend=backend, device=str(device))
    results = []

    def _record(func_name, n_atoms, n_systems, callable_):
        stats = timer.time_function(callable_)
        row = {
            "function": func_name,
            "backend": backend,
            "num_atoms_per_system": n_atoms,
            "num_systems": n_systems,
            "total_atoms": n_atoms * n_systems,
            "median_ms": stats.get("median"),
            "peak_memory_mb": stats.get("peak_memory_mb"),
            "success": stats.get("success"),
        }
        results.append(row)
        status = "ok" if stats.get("success") else f"FAIL ({stats.get('error_type')})"
        med = stats.get("median")
        med_s = f"{med:.3f} ms" if med is not None else "n/a"
        print(
            f"  [{backend}] {func_name:24s} N={n_atoms:>6} B={n_systems:>2} -> {med_s} [{status}]"
        )

    print(f"=== Scaling sweep ({backend}) ===")
    for n in sizes:
        inp = _build_torch_inputs(n, device, dtype, n_systems=1)
        calls = (
            _make_torch_callables(inp)
            if backend == "torch"
            else _make_jax_callables(inp)
        )
        for fname, fn in calls.items():
            _record(fname, n, 1, fn)

    print(f"=== Batched sweep ({backend}, N={sizes[0]} per system) ===")
    for b in batch_sizes:
        inp = _build_torch_inputs(sizes[0], device, dtype, n_systems=b)
        calls = (
            _make_torch_callables(inp)
            if backend == "torch"
            else _make_jax_callables(inp)
        )
        for fname, fn in calls.items():
            _record(fname, sizes[0], b, fn)

    save_benchmark_results(results, output, f"dispersion_{backend}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark dispersion PME (LJ-PME).")
    parser.add_argument("--backend", choices=["torch", "jax"], default="torch")
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES
    )
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output", default=str(Path(__file__).parent / "dispersion_benchmark.csv")
    )
    args = parser.parse_args()

    if args.backend == "jax":
        import jax

        jax.config.update("jax_enable_x64", args.dtype == "float64")

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    np.random.seed(0)
    run(args.backend, args.sizes, args.batch_sizes, dtype, device, args.output)


if __name__ == "__main__":
    main()
