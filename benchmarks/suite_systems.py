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

"""Chemical system generation and loading for benchmarks.

Two systems supported:
- CsCl (cesium chloride): primitive-cubic B2 crystal, 2 atoms/cell, programmatic
- NH3 (ammonia): Packmol-packed PBC boxes, loaded from PDB files

Each system provides: positions, atomic_numbers, cell, pbc, charges (optional),
and batching support via tiling/replication.

Supports both 'torch' and 'jax' backends via the ``backend`` parameter on
:func:`create_system`. The underlying system generation runs on numpy and is
converted to the requested framework at the end. Returned dictionaries have
the same keys for both backends; only the array types differ.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

__all__ = [
    "configs_for_mode",
    "configured_nh3_artifacts",
    "cscl_actual_atoms",
    "create_cscl_batch",
    "create_cscl_system",
    "create_nh3_batch",
    "create_system",
    "find_nh3_pdbs",
    "filter_configs_by_total_atoms",
    "get_constant_atoms_configs",
    "get_constant_total_configs",
    "get_system_size_configs",
    "planned_atom_counts",
    "load_nh3_system",
    "parse_pdb",
    "resolve_nh3_dir",
]

# =============================================================================
# Constants
# =============================================================================

# Element atomic numbers
ELEMENT_Z = {"H": 1, "C": 6, "N": 7, "O": 8, "Cs": 55, "Cl": 17}

# Partial charges for NH3 (neutral molecule: 3×0.3 + 1×(-0.9) = 0)
NH3_PARTIAL_CHARGES = {"H": 0.3, "N": -0.9}

# CsCl charges (ionic crystal)
CSCL_CHARGES = {"Cs": 1.0, "Cl": -1.0}

# CsCl lattice constant in Angstroms (Cs at corner, Cl at body center)
CSCL_LATTICE_CONSTANT = 4.119


def cscl_actual_atoms(n):
    """Return actual CsCl atom count for a target of n atoms.

    CsCl has 2 atoms per cubic unit cell, so valid sizes are 2*k^3.
    """
    n_cells = max(1, int(np.ceil((n / 2) ** (1 / 3))))
    return 2 * n_cells**3


# Default paths (relative to benchmarks/ directory)
SCRIPT_DIR = Path(__file__).parent
DEFAULT_NH3_DIR = SCRIPT_DIR / "nh3"


def configured_nh3_artifacts(config: dict) -> dict[str, Path]:
    """Return the canonical NH3 PDB inputs used by an enabled suite config.

    Fingerprint only the PDBs consumed by the benchmark. Packmol ``.inp`` and
    log files can contain scratch-specific paths and are generation records,
    not runtime inputs.
    """
    nh3_config = config.get("systems", {}).get("nh3")
    if not isinstance(nh3_config, dict) or not nh3_config.get("enabled", True):
        return {}
    nh3_dir = resolve_nh3_dir(nh3_config) or DEFAULT_NH3_DIR
    atom_counts = {
        int(value)
        for key in ("atom_counts", "constant_atoms_sizes")
        for value in nh3_config.get(key, [])
    }
    if not atom_counts:
        return {"nh3_pdb_directory": nh3_dir}
    return {
        f"nh3_pdb_{atom_count}": nh3_dir / f"ammonia_pbc_{atom_count}.pdb"
        for atom_count in sorted(atom_counts)
    }


# =============================================================================
# PDB Parsing (NH3 systems)
# =============================================================================


def parse_pdb(path):
    """Parse PDB file with CRYST1 support.

    Parameters
    ----------
    path : str or Path
        Path to PDB file.

    Returns
    -------
    coords : np.ndarray, shape (N, 3), float32
        Atomic coordinates in Angstroms.
    atomic_numbers : np.ndarray, shape (N,), int32
        Atomic numbers.
    elements : list[str]
        Element symbols.
    cell : np.ndarray, shape (3, 3), float32
        Unit cell matrix (diagonal for cubic cells).
    """
    lines = Path(path).read_text().splitlines()
    coords, numbers, elements = [], [], []
    cell = None

    for line in lines:
        if line.startswith("CRYST1"):
            parts = line.split()
            cell = np.diag([float(parts[1]), float(parts[2]), float(parts[3])]).astype(
                np.float32
            )
        if line.startswith(("HETATM", "ATOM")):
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            # Element symbol: columns 77-78 or infer from atom name
            el = line[76:78].strip() if len(line) >= 78 else line[12:14].strip()[0]
            elements.append(el)
            numbers.append(ELEMENT_Z.get(el, 1))

    if cell is None:
        cell = np.eye(3, dtype=np.float32) * 10.0  # default fallback

    return np.asarray(coords, np.float32), np.asarray(numbers, np.int32), elements, cell


def find_nh3_pdbs(nh3_dir=None):
    """Find all NH3 PDB files sorted by atom count.

    Parameters
    ----------
    nh3_dir : str or Path, optional
        Directory containing ammonia_pbc_*.pdb files.

    Returns
    -------
    list[Path]
        Sorted PDB file paths (by atom count extracted from filename).
    """
    import re

    nh3_dir = Path(nh3_dir or DEFAULT_NH3_DIR)
    pdb_files = list(nh3_dir.glob("ammonia_pbc_*.pdb"))

    if not pdb_files:
        raise FileNotFoundError(
            f"No NH3 PDB files in {nh3_dir}. Run: cd nh3 && bash generate_pbc_pdbs.sh"
        )

    # Sort by atom count in filename (natural sort without natsort dependency)
    def _extract_count(p):
        m = re.search(r"ammonia_pbc_(\d+)\.pdb", p.name)
        return int(m.group(1)) if m else 0

    return sorted(pdb_files, key=_extract_count)


# =============================================================================
# Backend Converters
# =============================================================================


def _to_torch(np_data, device="cuda", dtype=torch.float32):
    """Convert a numpy system dict to torch tensors on device.

    Floating arrays become ``dtype`` (default float32); charges always float64;
    integer arrays int32; boolean stays bool. Non-array metadata is passed
    through unchanged.
    """
    out = {}
    for k, v in np_data.items():
        if not isinstance(v, np.ndarray):
            out[k] = v
            continue
        if v.dtype == bool:
            out[k] = torch.tensor(v, dtype=torch.bool, device=device)
        elif k == "charges":
            out[k] = torch.tensor(v, dtype=torch.float64, device=device)
        elif np.issubdtype(v.dtype, np.integer):
            out[k] = torch.tensor(v, dtype=torch.int32, device=device)
        else:
            out[k] = torch.tensor(v, dtype=dtype, device=device)
    return out


def _to_jax(np_data, dtype=None):
    """Convert a numpy system dict to jax arrays on the default device.

    Floating arrays become ``dtype`` if given (default float32); charges use
    float64 only when JAX x64 is enabled; integers int32; bool stays bool.
    Imports JAX lazily so torch-only paths do not require it.
    """
    import jax
    import jax.numpy as jnp

    float_dtype = dtype if dtype is not None else jnp.float32
    out = {}
    for k, v in np_data.items():
        if not isinstance(v, np.ndarray):
            out[k] = v
            continue
        if v.dtype == bool:
            out[k] = jnp.asarray(v, dtype=jnp.bool_)
        elif k == "charges":
            charge_dtype = jnp.float64 if jax.config.x64_enabled else float_dtype
            out[k] = jnp.asarray(v, dtype=charge_dtype)
        elif np.issubdtype(v.dtype, np.integer):
            out[k] = jnp.asarray(v, dtype=jnp.int32)
        else:
            out[k] = jnp.asarray(v, dtype=float_dtype)
    return out


def _dispatch_backend(np_data, backend, device, dtype):
    """Dispatch a numpy system dict to the requested backend."""
    if backend == "torch":
        return _to_torch(np_data, device=device, dtype=dtype)
    elif backend == "jax":
        # For JAX, convert torch dtype tokens to jax equivalents
        import jax.numpy as jnp

        if dtype is None or dtype is torch.float32:
            jdt = jnp.float32
        elif dtype is torch.float64:
            jdt = jnp.float64
        else:
            jdt = dtype  # assume already a jax dtype
        return _to_jax(np_data, dtype=jdt)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'torch' or 'jax'.")


# =============================================================================
# NH3 System Creation
# =============================================================================


def _build_nh3_single_numpy(pdb_path):
    """Build a single-system NH3 numpy dict (backend-agnostic)."""
    coords, numbers, elements, cell = parse_pdb(pdb_path)
    charges = np.array(
        [NH3_PARTIAL_CHARGES.get(el, 0.0) for el in elements], dtype=np.float64
    )
    return {
        "positions": coords.astype(np.float32),
        "atomic_numbers": numbers.astype(np.int32),
        "charges": charges,
        "cell": cell.astype(np.float32)[None],  # [1, 3, 3]
        "pbc": np.array([[True, True, True]], dtype=bool),
        "batch_idx": np.zeros(len(numbers), dtype=np.int32),
        "elements": elements,
        "atoms_per_system": len(numbers),
        "cell_size": float(np.diag(cell)[0]),
    }


def _build_nh3_batch_numpy(pdb_path, batch_size):
    """Build a batched NH3 numpy dict (backend-agnostic)."""
    coords, numbers, elements, cell = parse_pdb(pdb_path)
    n = len(numbers)
    charges = np.array(
        [NH3_PARTIAL_CHARGES.get(el, 0.0) for el in elements], dtype=np.float64
    )
    return {
        "positions": np.tile(coords, (batch_size, 1)).astype(np.float32),
        "atomic_numbers": np.tile(numbers, batch_size).astype(np.int32),
        "charges": np.tile(charges, batch_size),
        "cell": np.tile(cell[None], (batch_size, 1, 1)).astype(np.float32),
        "pbc": np.ones((batch_size, 3), dtype=bool),
        "batch_idx": np.repeat(np.arange(batch_size, dtype=np.int32), n),
        "elements": elements,
        "atoms_per_system": n,
        "total_atoms": n * batch_size,
        "batch_size": batch_size,
        "cell_size": float(np.diag(cell)[0]),
    }


def load_nh3_system(pdb_path, device="cuda", dtype=torch.float32, backend="torch"):
    """Load a single NH3 system from PDB file.

    Parameters
    ----------
    pdb_path : str or Path
        Path to PDB file.
    device : str, default='cuda'
        Torch device (only used for ``backend='torch'``).
    dtype : torch.dtype, default=torch.float32
        Floating-point precision.
    backend : str, default='torch'
        Framework backend: ``'torch'`` or ``'jax'``.

    Returns
    -------
    dict
        Keys: positions, atomic_numbers, charges, cell, pbc,
              elements, atoms_per_system, cell_size.
    """
    np_data = _build_nh3_single_numpy(pdb_path)
    return _dispatch_backend(np_data, backend, device, dtype)


def create_nh3_batch(
    pdb_path, batch_size, device="cuda", dtype=torch.float32, backend="torch"
):
    """Create a batched NH3 system by replicating a single PDB.

    Parameters
    ----------
    pdb_path : str or Path
        Path to NH3 PDB file.
    batch_size : int
        Number of replicas.
    device : str, default='cuda'
        Torch device (only used for ``backend='torch'``).
    dtype : torch.dtype, default=torch.float32
        Floating-point precision.
    backend : str, default='torch'
        Framework backend: ``'torch'`` or ``'jax'``.

    Returns
    -------
    dict
        Batched system with concatenated positions, tiled cells, batch_idx, etc.
    """
    np_data = _build_nh3_batch_numpy(pdb_path, batch_size)
    return _dispatch_backend(np_data, backend, device, dtype)


# =============================================================================
# CsCl System Creation
# =============================================================================


def _build_cscl_single_numpy(num_atoms):
    """Build a single-system CsCl numpy dict (backend-agnostic)."""
    a = CSCL_LATTICE_CONSTANT
    atoms_per_cell = 2  # Cs + Cl

    n_cells = max(1, int(np.ceil((num_atoms / atoms_per_cell) ** (1 / 3))))
    actual_atoms = cscl_actual_atoms(num_atoms)

    positions = []
    atomic_numbers = []
    charges = []

    for ix in range(n_cells):
        for iy in range(n_cells):
            for iz in range(n_cells):
                origin = np.array([ix, iy, iz], dtype=np.float32) * a
                # Cs at corner
                positions.append(origin)
                atomic_numbers.append(ELEMENT_Z["Cs"])
                charges.append(CSCL_CHARGES["Cs"])
                # Cl at body center
                positions.append(
                    origin + np.array([0.5, 0.5, 0.5], dtype=np.float32) * a
                )
                atomic_numbers.append(ELEMENT_Z["Cl"])
                charges.append(CSCL_CHARGES["Cl"])

    cell_size = n_cells * a
    cell = np.eye(3, dtype=np.float32) * cell_size

    return {
        "positions": np.asarray(positions[:actual_atoms], dtype=np.float32),
        "atomic_numbers": np.asarray(atomic_numbers[:actual_atoms], dtype=np.int32),
        "charges": np.asarray(charges[:actual_atoms], dtype=np.float64),
        "cell": cell[None],  # [1, 3, 3]
        "pbc": np.array([[True, True, True]], dtype=bool),
        "batch_idx": np.zeros(actual_atoms, dtype=np.int32),
        "atoms_per_system": actual_atoms,
        "total_atoms": actual_atoms,
        "batch_size": 1,
        "cell_size": cell_size,
    }


def _build_cscl_batch_numpy(num_atoms_per_system, batch_size):
    """Build a batched CsCl numpy dict (backend-agnostic)."""
    single = _build_cscl_single_numpy(num_atoms_per_system)
    n = single["atoms_per_system"]

    positions = np.tile(single["positions"], (batch_size, 1))
    atomic_numbers = np.tile(single["atomic_numbers"], batch_size)
    charges = np.tile(single["charges"], batch_size)
    cell = np.tile(single["cell"], (batch_size, 1, 1))
    pbc = np.tile(single["pbc"], (batch_size, 1))
    batch_idx = np.repeat(np.arange(batch_size, dtype=np.int32), n)

    return {
        "positions": positions.astype(np.float32),
        "atomic_numbers": atomic_numbers.astype(np.int32),
        "charges": charges,
        "cell": cell.astype(np.float32),
        "pbc": pbc,
        "batch_idx": batch_idx,
        "atoms_per_system": n,
        "total_atoms": n * batch_size,
        "batch_size": batch_size,
        "cell_size": single["cell_size"],
    }


def create_cscl_system(num_atoms, device="cuda", dtype=torch.float32, backend="torch"):
    """Create a CsCl supercell with approximately num_atoms atoms.

    CsCl/B2 has a primitive-cubic lattice with Cs at (0,0,0) and Cl at
    (0.5,0.5,0.5) in fractional coordinates.
    2 atoms per unit cell.

    Parameters
    ----------
    num_atoms : int
        Target number of atoms (rounded to nearest even number).
    device : str, default='cuda'
        Torch device (only used for ``backend='torch'``).
    dtype : torch.dtype, default=torch.float32
        Floating-point precision.
    backend : str, default='torch'
        Framework backend: ``'torch'`` or ``'jax'``.

    Returns
    -------
    dict
        System with positions, atomic_numbers, charges, cell, pbc.
    """
    np_data = _build_cscl_single_numpy(num_atoms)
    return _dispatch_backend(np_data, backend, device, dtype)


def create_cscl_batch(
    num_atoms_per_system,
    batch_size,
    device="cuda",
    dtype=torch.float32,
    backend="torch",
):
    """Create a batched CsCl system by replicating a supercell.

    Parameters
    ----------
    num_atoms_per_system : int
        Atoms per individual CsCl supercell.
    batch_size : int
        Number of replicas.
    device : str, default='cuda'
        Torch device (only used for ``backend='torch'``).
    dtype : torch.dtype, default=torch.float32
        Floating-point precision.
    backend : str, default='torch'
        Framework backend: ``'torch'`` or ``'jax'``.

    Returns
    -------
    dict
        Batched system with concatenated positions, tiled cells, batch_idx.
    """
    np_data = _build_cscl_batch_numpy(num_atoms_per_system, batch_size)
    return _dispatch_backend(np_data, backend, device, dtype)


# =============================================================================
# Unified System Factory
# =============================================================================


def create_system(
    system_type: str,
    num_atoms: int | None = None,
    pdb_path: str | Path | None = None,
    batch_size: int = 1,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    backend: str = "torch",
) -> dict:
    """Create a benchmark system (single or batched).

    Parameters
    ----------
    system_type : str
        'cscl' or 'nh3'.
    num_atoms : int, optional
        Target atoms per system (required for CsCl, ignored for NH3).
    pdb_path : str or Path, optional
        PDB file path (required for NH3, ignored for CsCl).
    batch_size : int, default=1
        Number of system replicas.
    device : str, default='cuda'
        PyTorch device (only used for ``backend='torch'``).
    dtype : torch.dtype, default=torch.float32
        Floating-point precision.
    backend : str, default='torch'
        Framework backend: ``'torch'`` or ``'jax'``. The returned dict has the
        same keys for both; array types are backend-specific.

    Returns
    -------
    dict
        System dictionary with all required fields for benchmarking.
    """
    if system_type == "cscl":
        if num_atoms is None:
            raise ValueError("num_atoms required for CsCl systems")
        if batch_size == 1:
            return create_cscl_system(
                num_atoms, device=device, dtype=dtype, backend=backend
            )
        else:
            return create_cscl_batch(
                num_atoms, batch_size, device=device, dtype=dtype, backend=backend
            )

    elif system_type == "nh3":
        if pdb_path is None:
            raise ValueError("pdb_path required for NH3 systems")
        if batch_size == 1:
            return load_nh3_system(
                pdb_path, device=device, dtype=dtype, backend=backend
            )
        else:
            return create_nh3_batch(
                pdb_path, batch_size, device=device, dtype=dtype, backend=backend
            )

    else:
        raise ValueError(f"Unknown system type: {system_type}. Use 'cscl' or 'nh3'.")


# =============================================================================
# Scaling Mode Helpers
# =============================================================================


def get_system_size_configs(system_type, atom_counts, nh3_dir=None):
    """Generate configs for system-size scaling (batch=1, vary N).

    Parameters
    ----------
    system_type : str
        'cscl' or 'nh3'.
    atom_counts : list[int]
        Target atom counts.
    nh3_dir : str or Path, optional
        NH3 PDB directory.

    Yields
    ------
    dict
        Config with 'num_atoms', 'pdb_path' (NH3 only), 'batch_size'=1.
    """
    if system_type == "nh3":
        pdb_files = find_nh3_pdbs(nh3_dir)
        for pdb in pdb_files:
            coords, _, _, _ = parse_pdb(pdb)
            n = len(coords)
            if atom_counts and n not in atom_counts:
                continue
            yield {"num_atoms": n, "pdb_path": pdb, "batch_size": 1}
    else:
        for n in atom_counts:
            yield {"num_atoms": n, "pdb_path": None, "batch_size": 1}


def get_constant_total_configs(
    system_type,
    target_atoms,
    atom_counts=None,
    nh3_dir=None,
):
    """Generate configs for constant-total-atoms scaling (128k batch).

    batch_size = target_atoms / atoms_per_system.

    Parameters
    ----------
    system_type : str
        'cscl' or 'nh3'.
    target_atoms : int
        Total atom target (e.g., 131072 = 128k).
    atom_counts : Sequence[int], optional
        Per-system atom-count targets from the benchmark YAML.
    nh3_dir : str or Path, optional
        NH3 PDB directory.

    Yields
    ------
    dict
        Config with 'num_atoms', 'pdb_path', 'batch_size'.
    """
    atom_counts = list(atom_counts or [])
    if system_type == "nh3":
        pdb_files = find_nh3_pdbs(nh3_dir)
        for pdb in pdb_files:
            coords, _, _, _ = parse_pdb(pdb)
            n = len(coords)
            if atom_counts and n not in atom_counts:
                continue
            batch_size = target_atoms // n
            if batch_size < 1:
                continue
            yield {"num_atoms": n, "pdb_path": pdb, "batch_size": batch_size}
    else:
        for n in atom_counts:
            actual = cscl_actual_atoms(n)
            batch_size = target_atoms // actual
            if batch_size < 1:
                continue
            yield {"num_atoms": n, "pdb_path": None, "batch_size": batch_size}


def get_constant_atoms_configs(
    system_type, atoms_per_system_sizes, max_total_atoms=131072, nh3_dir=None
):
    """Generate configs for constant-atoms-per-system scaling (vary batch).

    Batch size grows in powers of 2 until total_atoms exceeds max_total_atoms.

    Parameters
    ----------
    system_type : str
        'cscl' or 'nh3'.
    atoms_per_system_sizes : list[int]
        Fixed atom counts to test (e.g., [256, 8192]).
    max_total_atoms : int, default=131072
        Maximum total atoms (atoms_per_system * batch_size).
    nh3_dir : str or Path, optional
        NH3 PDB directory.

    Yields
    ------
    dict
        Config with 'num_atoms', 'pdb_path', 'batch_size'.
    """
    # Batch sizes: powers of 2, capped by max_total_atoms
    all_batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    if system_type == "nh3":
        pdb_files = find_nh3_pdbs(nh3_dir)
        for pdb in pdb_files:
            coords, _, _, _ = parse_pdb(pdb)
            n = len(coords)
            if n not in atoms_per_system_sizes:
                continue
            for bs in all_batch_sizes:
                if n * bs > max_total_atoms:
                    break  # stop growing batch for this atom size
                yield {"num_atoms": n, "pdb_path": pdb, "batch_size": bs}
    else:
        for n in atoms_per_system_sizes:
            actual = cscl_actual_atoms(n)
            for bs in all_batch_sizes:
                if actual * bs > max_total_atoms:
                    break
                yield {"num_atoms": n, "pdb_path": None, "batch_size": bs}


# =============================================================================
# Runner Helpers (shared across NL/D3/EL run_from_config)
# =============================================================================


def resolve_nh3_dir(sys_config: dict) -> Path | None:
    """Resolve the NH3 PDB directory from a ``config['systems']['nh3']`` subtree.

    YAML key ``pdb_dir`` wins. Relative paths are resolved against the
    packaged ``benchmarks/nh3`` location — that way configs work regardless
    of which runner's directory invoked them. Returns ``None`` when no YAML
    override is set, in which case downstream :func:`find_nh3_pdbs` uses
    :data:`DEFAULT_NH3_DIR`.
    """
    pdb_dir = sys_config.get("pdb_dir")
    if not pdb_dir:
        return None
    pdb_dir = Path(pdb_dir)
    if pdb_dir.is_absolute():
        return pdb_dir
    candidate = SCRIPT_DIR / pdb_dir.name
    if candidate.exists():
        return candidate
    return DEFAULT_NH3_DIR


def configs_for_mode(
    mode_name: str,
    mode_config: dict,
    sys_name: str,
    sys_config: dict,
    nh3_dir: Path | None = None,
    *,
    plan_only: bool = False,
) -> list[dict]:
    """Dispatch to the right scaling-mode helper and return a concrete config list.

    Parameters
    ----------
    mode_name : str
        One of ``'system_size'``, ``'constant_workload'``, ``'batch_scaling'``.
        Unknown names return an empty list so callers can ``continue`` cleanly.
    mode_config : dict
        Subtree at ``config['scaling'][mode_name]``. Required keys per mode:
        ``target_atoms`` (constant_workload), ``max_total_atoms``
        (batch_scaling). Missing keys raise ``KeyError`` — YAML is
        authoritative.
    sys_name : str
        ``'cscl'`` or ``'nh3'``.
    sys_config : dict
        Subtree at ``config['systems'][sys_name]``. Uses ``atom_counts`` and
        ``constant_atoms_sizes``; both default to empty/``[1024, 8192]`` to
        support NH3 configs that omit them (NH3 discovers atom counts from
        PDB filenames instead).
    nh3_dir : Path, optional
        NH3 PDB directory (see :func:`resolve_nh3_dir`). Ignored for CsCl.
    plan_only : bool, default=False
        If True, synthesize NH3 configs from YAML atom counts without requiring
        generated PDB files. Actual benchmark runs prefer the PDB-backed path;
        if generated PDBs are missing, they fall back to these planned configs
        so runners can emit explicit ``success=False`` rows instead of aborting.

    Returns
    -------
    list[dict]
        Concrete configs with ``num_atoms``, ``batch_size``, and optionally
        ``pdb_path``.
    """
    atom_counts = sys_config.get("atom_counts", [])
    constant_atoms_sizes = sys_config.get("constant_atoms_sizes", [1024, 8192])
    if sys_name == "nh3":
        nh3_missing = False
        if not plan_only:
            try:
                find_nh3_pdbs(nh3_dir)
            except FileNotFoundError:
                nh3_missing = True
        if plan_only or nh3_missing:
            if nh3_missing:
                nh3_path = Path(nh3_dir or DEFAULT_NH3_DIR)
                print(
                    f"  WARNING: no NH3 PDB files in {nh3_path}; "
                    "recording planned failures"
                )
            if mode_name == "system_size":
                return [
                    {"num_atoms": n, "pdb_path": None, "batch_size": 1}
                    for n in atom_counts
                ]
            if mode_name == "constant_workload":
                target_atoms = mode_config["target_atoms"]
                return [
                    {
                        "num_atoms": n,
                        "pdb_path": None,
                        "batch_size": target_atoms // n,
                    }
                    for n in atom_counts
                    if target_atoms // n >= 1
                ]
            if mode_name == "batch_scaling":
                configs = []
                for n in constant_atoms_sizes:
                    for batch_size in (
                        1,
                        2,
                        4,
                        8,
                        16,
                        32,
                        64,
                        128,
                        256,
                        512,
                        1024,
                    ):
                        if n * batch_size > mode_config["max_total_atoms"]:
                            break
                        configs.append(
                            {
                                "num_atoms": n,
                                "pdb_path": None,
                                "batch_size": batch_size,
                            }
                        )
                return configs

    if mode_name == "system_size":
        return list(get_system_size_configs(sys_name, atom_counts, nh3_dir))
    if mode_name == "constant_workload":
        return list(
            get_constant_total_configs(
                sys_name,
                mode_config["target_atoms"],
                atom_counts,
                nh3_dir,
            )
        )
    if mode_name == "batch_scaling":
        return list(
            get_constant_atoms_configs(
                sys_name,
                constant_atoms_sizes,
                mode_config["max_total_atoms"],
                nh3_dir,
            )
        )
    return []


def planned_atom_counts(sys_name: str, cfg: dict) -> tuple[int, int, int]:
    """Return ``(atoms_per_system, batch_size, total_atoms)`` without allocation."""
    batch_size = int(cfg["batch_size"])
    if sys_name == "cscl":
        atoms_per_system = cscl_actual_atoms(cfg["num_atoms"])
    else:
        atoms_per_system = int(cfg["num_atoms"])
    return atoms_per_system, batch_size, atoms_per_system * batch_size


def filter_configs_by_total_atoms(
    configs: list[dict],
    sys_name: str,
    max_total_atoms: int | None,
) -> tuple[list[dict], list[tuple[dict, int]]]:
    """Split configs into runnable and skipped rows using a total-atom cap."""
    if max_total_atoms is None:
        return configs, []
    kept = []
    skipped = []
    for cfg in configs:
        _, _, total_atoms = planned_atom_counts(sys_name, cfg)
        if total_atoms > max_total_atoms:
            skipped.append((cfg, total_atoms))
        else:
            kept.append(cfg)
    return kept, skipped
