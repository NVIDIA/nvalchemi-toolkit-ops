# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chemical system generation and loading for benchmarks.

Two systems supported:
- CsCl (cesium chloride): BCC-like crystal, 2 atoms/unit cell, programmatic
- NH3 (ammonia): Packmol-packed PBC boxes, loaded from PDB files

Each system provides: positions, atomic_numbers, cell, pbc, charges (optional),
and batching support via tiling/replication.
"""

from pathlib import Path

import numpy as np
import torch

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
# NH3 System Creation
# =============================================================================


def load_nh3_system(pdb_path, device="cuda", dtype=torch.float32):
    """Load a single NH3 system from PDB file.

    Returns
    -------
    dict
        Keys: positions, atomic_numbers, charges, cell, pbc,
              elements, atoms_per_system, cell_size.
    """
    coords, numbers, elements, cell = parse_pdb(pdb_path)
    charges = np.array(
        [NH3_PARTIAL_CHARGES.get(el, 0.0) for el in elements], dtype=np.float64
    )

    return {
        "positions": torch.tensor(coords, dtype=dtype, device=device),
        "atomic_numbers": torch.tensor(numbers, dtype=torch.int32, device=device),
        "charges": torch.tensor(charges, dtype=torch.float64, device=device),
        "cell": torch.tensor(cell, dtype=dtype, device=device).unsqueeze(
            0
        ),  # [1, 3, 3]
        "pbc": torch.tensor(
            [True, True, True], dtype=torch.bool, device=device
        ).unsqueeze(0),
        "elements": elements,
        "atoms_per_system": len(numbers),
        "cell_size": float(np.diag(cell)[0]),
    }


def create_nh3_batch(pdb_path, batch_size, device="cuda", dtype=torch.float32):
    """Create a batched NH3 system by replicating a single PDB.

    Parameters
    ----------
    pdb_path : str or Path
        Path to NH3 PDB file.
    batch_size : int
        Number of replicas.

    Returns
    -------
    dict
        Batched system with concatenated positions, tiled cells, batch_idx, etc.
    """
    coords, numbers, elements, cell = parse_pdb(pdb_path)
    n = len(numbers)
    charges = np.array(
        [NH3_PARTIAL_CHARGES.get(el, 0.0) for el in elements], dtype=np.float64
    )

    return {
        "positions": torch.tensor(
            np.tile(coords, (batch_size, 1)), dtype=dtype, device=device
        ),
        "atomic_numbers": torch.tensor(
            np.tile(numbers, batch_size), dtype=torch.int32, device=device
        ),
        "charges": torch.tensor(
            np.tile(charges, batch_size), dtype=torch.float64, device=device
        ),
        "cell": torch.tensor(
            np.tile(cell[None], (batch_size, 1, 1)), dtype=dtype, device=device
        ),
        "pbc": torch.ones(batch_size, 3, dtype=torch.bool, device=device),
        "batch_idx": torch.tensor(
            np.repeat(np.arange(batch_size, dtype=np.int32), n), device=device
        ),
        "elements": elements,
        "atoms_per_system": n,
        "total_atoms": n * batch_size,
        "batch_size": batch_size,
        "cell_size": float(np.diag(cell)[0]),
    }


# =============================================================================
# CsCl System Creation
# =============================================================================


def create_cscl_system(num_atoms, device="cuda", dtype=torch.float32):
    """Create a CsCl supercell with approximately num_atoms atoms.

    CsCl is BCC-like: Cs at (0,0,0), Cl at (0.5,0.5,0.5) in fractional coords.
    2 atoms per unit cell.

    Parameters
    ----------
    num_atoms : int
        Target number of atoms (rounded to nearest even number).

    Returns
    -------
    dict
        System with positions, atomic_numbers, charges, cell, pbc.
    """
    a = CSCL_LATTICE_CONSTANT
    atoms_per_cell = 2  # Cs + Cl

    n_cells = max(1, int(np.ceil((num_atoms / atoms_per_cell) ** (1 / 3))))
    actual_atoms = cscl_actual_atoms(num_atoms)

    # Build supercell
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
        "positions": torch.tensor(
            np.array(positions[:actual_atoms]), dtype=dtype, device=device
        ),
        "atomic_numbers": torch.tensor(
            np.array(atomic_numbers[:actual_atoms], dtype=np.int32), device=device
        ),
        "charges": torch.tensor(
            np.array(charges[:actual_atoms], dtype=np.float64), device=device
        ),
        "cell": torch.tensor(cell, dtype=dtype, device=device).unsqueeze(
            0
        ),  # [1, 3, 3]
        "pbc": torch.tensor([[True, True, True]], dtype=torch.bool, device=device),
        "batch_idx": torch.zeros(actual_atoms, dtype=torch.int32, device=device),
        "atoms_per_system": actual_atoms,
        "total_atoms": actual_atoms,
        "batch_size": 1,
        "cell_size": cell_size,
    }


def create_cscl_batch(
    num_atoms_per_system, batch_size, device="cuda", dtype=torch.float32
):
    """Create a batched CsCl system by replicating a supercell.

    Parameters
    ----------
    num_atoms_per_system : int
        Atoms per individual CsCl supercell.
    batch_size : int
        Number of replicas.

    Returns
    -------
    dict
        Batched system with concatenated positions, tiled cells, batch_idx.
    """
    single = create_cscl_system(num_atoms_per_system, device="cpu", dtype=torch.float32)
    n = single["atoms_per_system"]

    positions = single["positions"].repeat(batch_size, 1)
    atomic_numbers = single["atomic_numbers"].repeat(batch_size)
    charges = single["charges"].repeat(batch_size)
    cell = single["cell"].repeat(batch_size, 1, 1)
    pbc = single["pbc"].repeat(batch_size, 1)
    batch_idx = torch.arange(batch_size, dtype=torch.int32).repeat_interleave(n)

    return {
        "positions": positions.to(device=device, dtype=dtype),
        "atomic_numbers": atomic_numbers.to(device=device),
        "charges": charges.to(device=device),
        "cell": cell.to(device=device, dtype=dtype),
        "pbc": pbc.to(device=device),
        "batch_idx": batch_idx.to(device=device),
        "atoms_per_system": n,
        "total_atoms": n * batch_size,
        "batch_size": batch_size,
        "cell_size": single["cell_size"],
    }


# =============================================================================
# Unified System Factory
# =============================================================================


def create_system(
    system_type,
    num_atoms=None,
    pdb_path=None,
    batch_size=1,
    device="cuda",
    dtype=torch.float32,
):
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
        PyTorch device.
    dtype : torch.dtype, default=torch.float32
        Floating-point precision.

    Returns
    -------
    dict
        System dictionary with all required fields for benchmarking.
    """
    if system_type == "cscl":
        if num_atoms is None:
            raise ValueError("num_atoms required for CsCl systems")
        if batch_size == 1:
            return create_cscl_system(num_atoms, device=device, dtype=dtype)
        else:
            return create_cscl_batch(num_atoms, batch_size, device=device, dtype=dtype)

    elif system_type == "nh3":
        if pdb_path is None:
            raise ValueError("pdb_path required for NH3 systems")
        if batch_size == 1:
            return load_nh3_system(pdb_path, device=device, dtype=dtype)
        else:
            return create_nh3_batch(pdb_path, batch_size, device=device, dtype=dtype)

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


def get_constant_total_configs(system_type, target_atoms, nh3_dir=None):
    """Generate configs for constant-total-atoms scaling (128k batch).

    batch_size = target_atoms / atoms_per_system.

    Parameters
    ----------
    system_type : str
        'cscl' or 'nh3'.
    target_atoms : int
        Total atom target (e.g., 131072 = 128k).
    nh3_dir : str or Path, optional
        NH3 PDB directory.

    Yields
    ------
    dict
        Config with 'num_atoms', 'pdb_path', 'batch_size'.
    """
    if system_type == "nh3":
        pdb_files = find_nh3_pdbs(nh3_dir)
        for pdb in pdb_files:
            coords, _, _, _ = parse_pdb(pdb)
            n = len(coords)
            batch_size = target_atoms // n
            if batch_size < 1:
                continue
            yield {"num_atoms": n, "pdb_path": pdb, "batch_size": batch_size}
    else:
        for n in [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]:
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
