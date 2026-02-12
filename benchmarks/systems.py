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
"""Benchmark dataset systems for molecules and crystals.

This module provides a unified interface for creating benchmarking datasets
across different atomistic system types: molecules and periodic crystal
supercells. Each domain provides a set of hardcoded systems with automatic
caching and computation of required properties.

Additionally, this module contains synthetic system generation functions
for creating test systems with controlled properties for performance testing.

Usage
-----
The primary interface is the `create_benchmark_dataset` factory function:

    >>> from benchmarks.systems import create_benchmark_dataset, combine_systems
    >>>
    >>> # Create molecule dataset from SMILES strings
    >>> mol_dataset = create_benchmark_dataset("molecule")
    >>> system = mol_dataset[0]  # Get first molecular system
    >>> coords = system["positions"]  # (num_atoms, 3) tensor
    >>> atomic_nums = system["atomic_numbers"]  # (num_atoms,) tensor
    >>>
    >>> # Create crystal dataset with supercells
    >>> crystal_dataset = create_benchmark_dataset(
    ...     "crystal",
    ...     max_supercell_size=1000
    ... )
    >>>
    >>> # Combine multiple systems for batch processing
    >>> systems = [mol_dataset[i] for i in range(3)]
    >>> combined = combine_systems(systems)
    >>> ptr = combined["ptr"]  # System boundaries: [0, n1, n1+n2, n1+n2+n3]
    >>>
    >>> # Use DataLoader for automatic batching
    >>> loader = mol_dataset.dataloader(batch_size=4, shuffle=True)
    >>> for batch in loader:
    ...     coords = batch["positions"]  # Concatenated positions
    ...     ptr = batch["ptr"]  # System boundaries

System Configuration
--------------------
Before using the datasets, populate the hardcoded system lists:

- MOLECULE_SMILES: List of SMILES strings for molecular systems
- CRYSTAL_COD_IDS: List of COD database crystal structure IDs

Example:
    >>> MOLECULE_SMILES.extend([
    ...     "C",           # methane (5 atoms)
    ...     "CCO",         # ethanol (9 atoms)
    ...     "c1ccccc1",    # benzene (12 atoms)
    ...     # ... more SMILES
    ... ])

Dataset Properties
------------------
All datasets inherit from torch.utils.data.Dataset and provide:

- Automatic caching with torch.save/load
- Device and dtype specification
- System metadata via get_system_info()
- Consistent tensor format across domains
- Batch processing via combine_systems() with ptr tensors
- PyTorch DataLoader integration via dataloader() method

Molecular Systems:
    - positions: (num_atoms, 3) 3D positions in Angstroms
    - atomic_numbers: (num_atoms,) atomic numbers
    - formal_charges: (num_atoms,) formal charges on atoms
    - molecular_charge: () total molecular charge

Crystal Systems:
    - positions: (num_atoms, 3) atomic positions in Angstroms
    - atomic_numbers: (num_atoms,) atomic numbers
    - cell: (3, 3) unit cell matrix in Angstroms
    - periodic: (3,) periodic boundary conditions

Dependencies
------------
Install required packages from benchmarks/requirements.txt:
    - rdkit: For molecular conformer generation from SMILES
    - pymatgen: For crystal structure loading and supercell generation
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import numpy as np
import requests
import torch
from loguru import logger
from pymatgen.core import Structure
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils.data import DataLoader, Dataset

# Hardcoded system definitions. Molecules are meant to include various topologies.
MOLECULE_SMILES: list[str] = [
    "CC(=O)C",
    "c1ccccc1",
    "C1=CC=C2C=CC=CC2=C1",
    "C1=CC2=C3C4=C1C=CC5=C4C6=C(C=C5)C=CC(=C36)C=C2",
    r"C12=C3C4=C5C6=C1C7=C8C9=C1C%10=C%11C(=C29)C3=C2C3=C4C4=C5C5=C9C6=C7C6=C7C8=C1C1=C8C%10=C%10C%11=C2C2=C3C3=C4C4=C5C5=C%11C%12=C(C6=C95)C7=C1C1=C%12C5=C%11C4=C3C3=C5C(=C81)C%10=C23",
    "CC1=C(C2=CC3=NC(=CC4=C(C(=C([N-]4)C=C5C(=C(C(=N5)C=C1[N-]2)C)C=C)C)C=C)C(=C3CCC(=O)O)C)CCC(=O)O.[Fe+2]",
    "CC(=O)OC1=CC=CC=C1C(=O)O",
    r"CC/C=C\C[C@H](/C=C/C=C\C=C\C=C\[C@H]([C@H](C/C=C\CCC(=O)O)O)O)O",
    r"O1CCOC1c1c(C#CC(C)(C)C)cc(c(C#CC(C)(C)C)c1)C#Cc1cc(C#CCCC)cc(C#CCCC)c1",
    """c1cc2c3cccc2OCCOCCOCCOCCOc4cccc5c4cccc5OCCOCCOCCOCCOc6cccc7c6cccc7OCCOCCOCCOCCOc3c1.c1cc2c3cccc2OCCOCCOCCOCCOc4cccc5c4cccc5OCCOCCOCCOCCOc6cccc7c6cccc7OCCOCCOCCOCCOc3c1.c1cc2ccc1C[n+]3ccc(cc3)-c4cc[n+](cc4)Cc5ccc(cc5)C[n+]6ccc(cc6)-c7cc[n+](cc7)C2.c1cc2ccc1C[n+]3ccc(cc3)-c4cc[n+](cc4)Cc5ccc(cc5)C[n+]6ccc(cc6)-c7cc[n+](cc7)C2.c1cc2ccc1C[n+]3ccc(cc3)-c4cc[n+](cc4)Cc5ccc(cc5)C[n+]6ccc(cc6)-c7cc[n+](cc7)C2.F[P-](F)(F)(F)(F)F.F[P-](F)(F)(F)(F)F.F[P-](F)(F)(F)(F)F.F[P-](F)(F)(F)(F)F.F[P-](F)(F)(F)(F)F.F[P-](F)(F)(F)(F)F.F[P-](F)(F)(F)(F)F.F[P-](F)(F)(F)(F)F.F[P-](F)(F)(F)(F)F.F[P-](F)(F)(F)(F)F.F[P-](F)(F)(F)(F)F.F[P-](F)(F)(F)(F)F""",
]
CRYSTAL_COD_IDS: list[int] = [
    1572854,
    1573046,
    1573246,
    1573353,
    1575104,
    7719320,
    7250127,
    1000030,
    1001375,
    1529556,
]


class BenchmarkDataset(Dataset[dict[str, torch.Tensor]], ABC):
    """Base class for atomistic benchmark datasets.

    This abstract base class provides a unified interface for different types
    of atomistic systems (molecules, proteins, crystals) with automatic caching
    and lazy loading of computed properties.

    Parameters
    ----------
    domain : str
        The domain type ("molecule", "protein", "crystal").
    cache_dir : Path | str | None, default=None
        Directory for caching computed data. If None, uses a temporary directory.
    device : torch.device, default=torch.device("cpu")
        Device to place tensors on.
    dtype : torch.dtype, default=torch.float32
        Data type for floating-point tensors.
    force_recompute : bool, default=False
        Whether to force recomputation even if cached data exists.

    Notes
    -----
    Subclasses must implement the `_compute_system` method to define how
    individual systems are processed and converted to PyTorch tensors.

    Examples
    --------
    >>> dataset = MoleculeDataset()
    >>> system = dataset[0]  # Returns dict with positions, atomic_numbers, etc.
    >>> logger.info(f"System has {system['positions'].shape[0]} atoms")
    """

    def __init__(
        self,
        domain: str,
        cache_dir: Path | str | None = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        force_recompute: bool = False,
    ):
        self.domain = domain
        self.device = device
        self.dtype = dtype
        self.force_recompute = force_recompute

        # Set up cache directory
        if cache_dir is None:
            self.cache_dir = Path.cwd() / "benchmark_cache" / domain
        else:
            self.cache_dir = Path(cache_dir) / domain
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize system list and cached data
        self._system_ids = self._get_system_ids()
        self._cached_data: dict[int, dict[str, torch.Tensor]] | None = None

        # Load or compute all systems
        self._ensure_data_available()

    @abstractmethod
    def _get_system_ids(self) -> list[str | int]:
        """Get list of system identifiers for this domain.

        Returns
        -------
        list[str | int]
            List of system identifiers (SMILES, protein IDs, COD IDs, etc.).
        """
        pass

    @abstractmethod
    def _compute_system(self, system_id: str | int) -> dict[str, torch.Tensor]:
        """Compute/download and process a single system.

        Parameters
        ----------
        system_id : str | int
            Identifier for the system to process.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing system data as PyTorch tensors.
        """
        pass

    def _get_cache_path(self) -> Path:
        """Get path to cached data file."""
        # Create a hash of the system configuration for cache invalidation
        config = {
            "domain": self.domain,
            "system_ids": self._system_ids,
            "device": str(self.device),
            "dtype": str(self.dtype),
        }
        # only using MD5 for cache checking, not for security
        config_hash = hashlib.md5(  # noqa: S324
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()
        return self.cache_dir / f"cached_systems_{config_hash}.pt"

    def _ensure_data_available(self):
        """Load cached data or compute all systems if not available."""
        cache_path = self._get_cache_path()

        if cache_path.exists() and not self.force_recompute:
            logger.info(f"Loading cached {self.domain} data from {cache_path}")
            self._cached_data = torch.load(cache_path, map_location=self.device)
        else:
            logger.info(f"Computing {self.domain} systems...")
            self._cached_data = {}

            for idx, system_id in enumerate(self._system_ids):
                logger.info(
                    f"Processing {self.domain} system {idx + 1}/{len(self._system_ids)}: {system_id}"
                )
                try:
                    system_data = self._compute_system(system_id)
                    # Move tensors to specified device
                    for key, tensor in system_data.items():
                        if isinstance(tensor, torch.Tensor):
                            system_data[key] = tensor.to(device=self.device)
                    self._cached_data[idx] = system_data
                except Exception as e:
                    logger.warning(f"Failed to process system {system_id}: {e}")
                    continue

            # Save to cache
            logger.info(f"Caching {self.domain} data to {cache_path}")
            torch.save(self._cached_data, cache_path)

    def __len__(self) -> int:
        """Return number of systems in the dataset."""
        return len(self._cached_data) if self._cached_data else 0

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get system data by index.

        Parameters
        ----------
        idx : int
            Index of the system to retrieve.

        Returns
        -------
        dict[str, torch.Tensor]
            System data as PyTorch tensors.
        """
        if self._cached_data is None or idx not in self._cached_data:
            raise IndexError(f"System {idx} not available")
        return self._cached_data[idx]

    def get_system_info(self, idx: int) -> dict[str, Any]:
        """Get metadata about a system.

        Parameters
        ----------
        idx : int
            Index of the system.

        Returns
        -------
        dict[str, Any]
            System metadata including ID, domain, and computed properties.
        """
        system_data = self[idx]
        return {
            "domain": self.domain,
            "system_id": self._system_ids[idx],
            "num_atoms": system_data["positions"].shape[0],
            "device": str(system_data["positions"].device),
            "dtype": str(system_data["positions"].dtype),
        }

    def get_dataloader(
        self,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        **kwargs,
    ) -> DataLoader:
        """Create a PyTorch DataLoader for batched system processing.

        This method returns a DataLoader that automatically batches multiple
        systems using the `combine_systems` function. Each batch contains
        concatenated positions, atomic numbers, and other system properties,
        along with a `ptr` tensor indicating system boundaries.

        Parameters
        ----------
        batch_size : int, default=1
            Number of systems per batch.
        shuffle : bool, default=False
            Whether to shuffle the dataset before batching.
        num_workers : int, default=0
            Number of worker processes for data loading.
        **kwargs
            Additional arguments passed to DataLoader constructor.

        Returns
        -------
        DataLoader
            PyTorch DataLoader that yields batched system dictionaries.
            Each batch contains:
            - ptr: (batch_size + 1,) system boundaries
            - positions: (total_atoms, 3) concatenated positions
            - atomic_numbers: (total_atoms,) concatenated atomic numbers
            - Additional domain-specific fields

        Notes
        -----
        The DataLoader uses a custom collate function that calls `combine_systems`
        internally. This ensures that systems are properly batched with the
        correct `ptr` tensor for downstream processing.

        For single-system batches (batch_size=1), the `ptr` tensor will be
        [0, num_atoms] where num_atoms is the number of atoms in that system.

        Examples
        --------
        Basic usage:

        >>> dataset = create_benchmark_dataset("molecule")
        >>> loader = dataset.dataloader(batch_size=4, shuffle=True)
        >>>
        >>> for batch in loader:
        ...     coords = batch["positions"]  # All systems concatenated
        ...     ptr = batch["ptr"]  # System boundaries
        ...
        ...     # Process first system in batch
        ...     system_0_coords = coords[ptr[0]:ptr[1]]
        ...
        ...     # Process all systems
        ...     for i in range(len(ptr) - 1):
        ...         system_coords = coords[ptr[i]:ptr[i+1]]
        ...         # ... process system_coords

        Multi-threaded loading:

        >>> loader = dataset.dataloader(
        ...     batch_size=8,
        ...     shuffle=True,
        ...     num_workers=4,
        ...     pin_memory=True  # For GPU transfer
        ... )

        Custom DataLoader options:

        >>> loader = dataset.dataloader(
        ...     batch_size=2,
        ...     drop_last=True,  # Drop incomplete batches
        ...     timeout=30,      # Timeout for worker processes
        ... )
        """

        def collate_systems(
            batch: list[dict[str, torch.Tensor]],
        ) -> dict[str, torch.Tensor]:
            """Custom collate function that combines systems using combine_systems."""
            return combine_systems(batch, device=self.device, dtype=self.dtype)

        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_systems,
            **kwargs,
        )


class MoleculeDataset(BenchmarkDataset):
    """Dataset for molecular systems from SMILES strings.

    This dataset generates 3D conformers from SMILES strings using RDKit,
    providing positions, atomic numbers, formal charges, and molecular
    charge information as PyTorch tensors.

    Parameters
    ----------
    cache_dir : Path | str | None, default=None
        Directory for caching computed data.
    device : torch.device, default=torch.device("cpu")
        Device to place tensors on.
    dtype : torch.dtype, default=torch.float32
        Data type for floating-point tensors.
    force_recompute : bool, default=False
        Whether to force recomputation of cached data.
    num_conformers : int, default=1
        Number of conformers to generate per molecule.

    Notes
    -----
    Requires RDKit for SMILES processing and conformer generation.
    Systems range from small molecules (few atoms) to larger organic
    compounds (hundreds of atoms).
    """

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        force_recompute: bool = False,
        num_conformers: int = 1,
    ):
        self.num_conformers = num_conformers
        super().__init__(
            domain="molecule",
            cache_dir=cache_dir,
            device=device,
            dtype=dtype,
            force_recompute=force_recompute,
        )

    def _get_system_ids(self) -> list[str]:
        """Get list of SMILES strings."""
        return MOLECULE_SMILES.copy()

    def _compute_system(self, smiles: str) -> dict[str, torch.Tensor]:
        """Generate 3D conformer from SMILES string.

        Parameters
        ----------
        smiles : str
            SMILES string representation of the molecule.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing:
            - positions: (num_atoms, 3) 3D positions in Angstroms
            - atomic_numbers: (num_atoms,) atomic numbers
            - formal_charges: (num_atoms,) formal charges on atoms
            - molecular_charge: () total molecular charge
        """

        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # Add hydrogens
        mol = Chem.AddHs(mol)

        # Generate 3D conformer
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol)

        # Extract atomic information
        conformer = mol.GetConformer()
        num_atoms = mol.GetNumAtoms()

        positions = []
        atomic_numbers = []
        formal_charges = []

        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            pos = conformer.GetAtomPosition(i)

            positions.append([pos.x, pos.y, pos.z])
            atomic_numbers.append(atom.GetAtomicNum())
            formal_charges.append(atom.GetFormalCharge())

        # Calculate molecular charge
        molecular_charge = sum(formal_charges)

        return {
            "positions": torch.tensor(positions, dtype=self.dtype),
            "atomic_numbers": torch.tensor(atomic_numbers, dtype=torch.int32),
            "formal_charges": torch.tensor(formal_charges, dtype=torch.int32),
            "molecular_charge": torch.tensor(molecular_charge, dtype=torch.int32),
        }


class CrystalDataset(BenchmarkDataset):
    """Dataset for crystal systems from Crystallography Open Database.

    This dataset downloads CIF files from the COD database, loads crystal
    structures with pymatgen, and optionally creates supercells for benchmarking
    neighbor list performance.

    Parameters
    ----------
    cache_dir : Path | str | None, default=None
        Directory for caching computed data.
    device : torch.device, default=torch.device("cpu")
        Device to place tensors on.
    dtype : torch.dtype, default=torch.float32
        Data type for floating-point tensors.
    force_recompute : bool, default=False
        Whether to force recomputation of cached data.
    max_supercell_size : int | None, default=None
        Maximum supercell size. If specified, creates supercells up to this size.

    Notes
    -----
    Downloads CIF files from crystallography.net COD database.
    Uses pymatgen for structure loading and supercell generation.
    Requires internet connection for initial download.
    """

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        force_recompute: bool = False,
        max_supercell_size: int | None = None,
    ):
        self.max_supercell_size = max_supercell_size
        super().__init__(
            domain="crystal",
            cache_dir=cache_dir,
            device=device,
            dtype=dtype,
            force_recompute=force_recompute,
        )

    def _get_system_ids(self) -> list[int]:
        """Get list of COD database IDs."""
        return CRYSTAL_COD_IDS.copy()

    def _compute_system(self, cod_id: int) -> dict[str, torch.Tensor]:
        """Download and process crystal structure from COD database.

        Parameters
        ----------
        cod_id : int
            COD database identifier.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing:
            - positions: (num_atoms, 3) atomic positions in Angstroms
            - atomic_numbers: (num_atoms,) atomic numbers
            - cell: (3, 3) unit cell matrix in Angstroms
            - periodic: (3,) periodic boundary conditions
        """

        # Download CIF file from COD
        cif_url = f"https://www.crystallography.net/cod/{cod_id}.cif"

        with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp_file:
            try:
                # Validate URL scheme for security
                parsed_url = urlparse(cif_url)
                if parsed_url.scheme not in ("http", "https"):
                    raise ValueError(f"Unsafe URL scheme: {parsed_url.scheme}")

                response = requests.get(cif_url, timeout=30)
                response.raise_for_status()
                tmp_file.write(response.content)
                tmp_file.flush()

                # Load structure with pymatgen
                structure = Structure.from_file(tmp_file.name)

                # Create supercell if requested
                if self.max_supercell_size is not None:
                    # Estimate supercell size based on unit cell volume
                    volume = structure.volume
                    target_volume = self.max_supercell_size * 100  # Rough target
                    scale_factor = max(1, int((target_volume / volume) ** (1 / 3)))

                    if scale_factor > 1:
                        supercell_matrix = [
                            [scale_factor, 0, 0],
                            [0, scale_factor, 0],
                            [0, 0, scale_factor],
                        ]
                        structure.make_supercell(supercell_matrix)

                # Extract information
                positions = structure.cart_coords
                atomic_numbers = np.array(
                    [site.specie.Z for site in structure], dtype=np.int32
                )
                cell = structure.lattice.matrix
                pbc = np.array(
                    [True, True, True]
                )  # pymatgen Structure is always periodic

                return {
                    "positions": torch.tensor(positions, dtype=self.dtype),
                    "atomic_numbers": torch.tensor(atomic_numbers, dtype=torch.int32),
                    "cell": torch.tensor(cell, dtype=self.dtype),
                    "pbc": torch.tensor(pbc, dtype=torch.bool),
                }

            finally:
                # Clean up temporary file
                os.unlink(tmp_file.name)


def combine_systems(
    systems: list[dict[str, torch.Tensor]],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    """Combine multiple atomistic systems into a single batched system.

    This function concatenates multiple system dictionaries along the atom
    dimension and creates a pointer tensor (`ptr`) that indicates the
    boundaries between systems. This is useful for batch processing and
    neighbor list computations across multiple systems.

    Parameters
    ----------
    systems : list[dict[str, torch.Tensor]]
        List of system dictionaries, each containing tensor data for one
        atomistic system. All systems should have the same tensor keys,
        though domain-specific fields are handled appropriately.
    device : torch.device | None, default=None
        Device to place output tensors on. If None, uses the device of
        the first system's positions tensor.
    dtype : torch.dtype | None, default=None
        Data type for floating-point tensors. If None, uses the dtype of
        the first system's positions tensor.

    Returns
    -------
    dict[str, torch.Tensor]
        Combined system dictionary containing:
        - ptr: (num_systems + 1,) cumulative atom counts, starting from 0
        - positions: (total_atoms, 3) concatenated positions
        - atomic_numbers: (total_atoms,) concatenated atomic numbers
        - Additional fields depending on system domain:
          * Molecules: formal_charges, molecular_charge
          * Crystals: cell, pbc (only from first system)

    Notes
    -----
    The `ptr` tensor follows PyTorch Geometric conventions where `ptr[i]`
    gives the starting atom index for system i, and `ptr[-1]` gives the
    total number of atoms. System i contains atoms `ptr[i]:ptr[i+1]`.

    For crystal systems, only the cell and pbc from the first system are
    used, as combining systems with different unit cells is not physically
    meaningful for most applications.

    For molecular charges, individual charges are preserved in a tensor
    rather than being summed, allowing analysis of each system separately.

    Examples
    --------
    Basic usage with molecules:

    >>> dataset = create_benchmark_dataset("molecule")
    >>> systems = [dataset[i] for i in range(3)]
    >>> combined = combine_systems(systems)
    >>>
    >>> # ptr indicates system boundaries
    >>> ptr = combined["ptr"]  # e.g., tensor([0, 15, 32, 48])
    >>> coords = combined["positions"]  # Shape: (48, 3)
    >>>
    >>> # Extract system 1 (atoms 15:32)
    >>> system_1_coords = coords[ptr[1]:ptr[2]]

    GPU acceleration:

    >>> combined = combine_systems(
    ...     systems,
    ...     device=torch.device("cuda:0"),
    ...     dtype=torch.float64
    ... )

    Raises
    ------
    ValueError
        If systems list is empty or if systems have incompatible tensor shapes
        for the same field (except along the atom dimension).
    RuntimeError
        If tensor operations fail due to device/dtype mismatches.
    """
    if not systems:
        raise ValueError("Cannot combine empty list of systems")

    # Determine device and dtype from first system if not specified
    first_coords = systems[0]["positions"]
    if device is None:
        device = first_coords.device
    if dtype is None:
        dtype = first_coords.dtype

    # Calculate ptr tensor (cumulative atom counts)
    atom_counts = [system["positions"].shape[0] for system in systems]
    ptr = torch.cumsum(torch.tensor([0] + atom_counts, dtype=torch.int64), dim=0)
    ptr = ptr.to(device)

    # Initialize combined system with ptr
    combined: dict[str, torch.Tensor] = {"ptr": ptr}

    # Get all unique keys across systems (common fields)
    all_keys = set()
    for system in systems:
        all_keys.update(system.keys())

    # Combine tensors for each field
    for key in all_keys:
        if key in systems[0]:  # Only process fields present in first system
            first_tensor = systems[0][key]

            # Handle different tensor types
            if key == "positions":
                # Always concatenate positions along atom dimension
                tensors = [
                    system[key].to(device=device, dtype=dtype)
                    for system in systems
                    if key in system
                ]
                combined[key] = torch.cat(tensors, dim=0)

            elif key == "atomic_numbers":
                # Concatenate atomic numbers (integer tensor)
                tensors = [
                    system[key].to(device=device) for system in systems if key in system
                ]
                combined[key] = torch.cat(tensors, dim=0)

            elif key == "formal_charges":
                # Per-atom integer fields - concatenate along atom dimension
                tensors = [
                    system[key].to(device=device) for system in systems if key in system
                ]
                combined[key] = torch.cat(tensors, dim=0)

            elif key == "molecular_charge":
                # Per-system scalar values - stack into 1D tensor
                tensors = [
                    system[key].to(device=device) for system in systems if key in system
                ]
                combined[key] = torch.stack(tensors, dim=0)

            elif key in ["cell", "pbc"]:
                # Crystal properties - use only from first system
                # (combining different unit cells is not physically meaningful)
                combined[key] = first_tensor.to(
                    device=device, dtype=dtype if key == "cell" else None
                )

            else:
                # Generic handling: try to concatenate along first dimension
                # This handles any additional custom fields
                try:
                    tensors = [
                        system[key].to(device=device)
                        for system in systems
                        if key in system
                    ]
                    if len(first_tensor.shape) == 0:  # Scalar tensors
                        combined[key] = torch.stack(tensors, dim=0)
                    else:  # Multi-dimensional tensors
                        combined[key] = torch.cat(tensors, dim=0)
                except Exception as e:
                    logger.warning(f"Could not combine field '{key}': {e}")
                    # Keep only the first system's value as fallback
                    combined[key] = first_tensor.to(device=device)

    return combined


# Synthetic system generation functions for benchmarking
# These functions create test systems with controlled properties for performance testing


def create_molecular_system(
    num_atoms: int,
    density: float = 0.35,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    """Create a molecular system (non-periodic) for benchmarking.

    Parameters
    ----------
    num_atoms : int
        Number of atoms in the system.
    density : float, default=0.35
        Number density (atoms per cubic unit).
    device : torch.device, default=CPU
        Device to create tensors on.
    dtype : torch.dtype, default=float64
        Data type for tensors.

    Returns
    -------
    Dict[str, torch.Tensor]
        System dictionary with positions, atomic_charges, cell,
        pbc.
    """
    # Calculate box size for desired density
    volume = num_atoms / density
    box_size = volume ** (1 / 3)

    # Random atomic positions
    positions = torch.rand(num_atoms, 3, dtype=dtype, device=device) * box_size

    # Random charges that sum to zero (neutral system)
    charges = torch.randn(num_atoms, dtype=dtype, device=device)
    charges = charges - charges.mean()  # Ensure neutrality

    # Random atomic numbers (mix of common elements: C=6, N=7, O=8)
    atomic_numbers = torch.randint(6, 9, (num_atoms,), dtype=torch.int32, device=device)

    # Non-periodic simulation cell
    cell = torch.zeros(1, 3, 3, dtype=dtype, device=device)
    cell[0, 0, 0] = box_size
    cell[0, 1, 1] = box_size
    cell[0, 2, 2] = box_size

    pbc = torch.tensor([False, False, False], dtype=torch.bool, device=device)

    return {
        "positions": positions,
        "atomic_charges": charges,
        "atomic_numbers": atomic_numbers,
        "cell": cell,
        "pbc": pbc,
        "system_type": "molecular",
        "num_atoms": num_atoms,
        "density": density,
        "box_size": box_size,
    }


def create_crystal_system(
    num_atoms: int,
    lattice_type: str = "fcc",
    lattice_constant: float = 4.0,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    """Create a crystalline system (periodic) for benchmarking.

    Parameters
    ----------
    num_atoms : int
        Target number of atoms in the system.
    lattice_type : str, default='fcc'
        Type of crystal lattice ('fcc', 'bcc', 'simple_cubic').
    lattice_constant : float, default=4.0
        Lattice constant in Angstroms.
    device : torch.device, default=CPU
        Device to create tensors on.
    dtype : torch.dtype, default=float64
        Data type for tensors.

    Returns
    -------
    Dict[str, torch.Tensor]
        System dictionary with positions, atomic_charges, cell,
        pbc.
    """
    import numpy as np

    if lattice_type == "fcc":
        # Face-centered cubic: 4 atoms per unit cell
        atoms_per_cell = 4
        basis = torch.tensor(
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]],
            dtype=dtype,
        )
    elif lattice_type == "bcc":
        # Body-centered cubic: 2 atoms per unit cell
        atoms_per_cell = 2
        basis = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=dtype)
    elif lattice_type == "simple_cubic":
        # Simple cubic: 1 atom per unit cell
        atoms_per_cell = 1
        basis = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype)
    else:
        raise ValueError(f"Unknown lattice type: {lattice_type}")

    # Determine number of unit cells needed
    n_cells = int(np.ceil((num_atoms / atoms_per_cell) ** (1 / 3)))

    positions = []
    charges = []
    atomic_numbers = []

    # Generate supercell
    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                if len(positions) >= num_atoms:
                    break

                cell_origin = torch.tensor([i, j, k], dtype=dtype) * lattice_constant

                for atom_idx, pos in enumerate(basis):
                    if len(positions) >= num_atoms:
                        break

                    position = cell_origin + pos * lattice_constant
                    positions.append(position)

                    # Alternating charges for electrostatic interactions
                    charge = 1.0 if (i + j + k + atom_idx) % 2 == 0 else -1.0
                    charges.append(charge)

                    # Alternating atomic numbers (Carbon=6, Oxygen=8)
                    atomic_num = 6 if (i + j + k + atom_idx) % 2 == 0 else 8
                    atomic_numbers.append(atomic_num)

    # Convert to tensors and trim to exact size
    positions = torch.stack(positions[:num_atoms]).to(device=device, dtype=dtype)
    charges = torch.tensor(charges[:num_atoms], dtype=dtype, device=device)
    atomic_numbers = torch.tensor(
        atomic_numbers[:num_atoms], dtype=torch.int32, device=device
    )

    # Ensure charge neutrality
    if abs(charges.sum().item()) > 1e-10:
        charges[-1] -= charges.sum()

    # Periodic simulation cell
    cell_size = n_cells * lattice_constant
    cell = torch.zeros(1, 3, 3, dtype=dtype, device=device)
    cell[0, 0, 0] = cell_size
    cell[0, 1, 1] = cell_size
    cell[0, 2, 2] = cell_size

    pbc = torch.tensor([True, True, True], dtype=torch.bool, device=device)

    return {
        "positions": positions,
        "atomic_charges": charges,
        "atomic_numbers": atomic_numbers,
        "cell": cell,
        "pbc": pbc,
        "system_type": "crystal",
        "lattice_type": lattice_type,
        "num_atoms": num_atoms,
        "lattice_constant": lattice_constant,
        "cell_size": cell_size,
    }


def create_random_system(
    num_atoms: int,
    periodic: bool = True,
    density: float = 0.8,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    """Create a random system for benchmarking.

    Parameters
    ----------
    num_atoms : int
        Number of atoms in the system.
    periodic : bool, default=True
        Whether to use periodic boundary conditions.
    density : float, default=0.8
        Number density (atoms per cubic unit).
    device : torch.device, default=CPU
        Device to create tensors on.
    dtype : torch.dtype, default=float64
        Data type for tensors.

    Returns
    -------
    Dict[str, torch.Tensor]
        System dictionary with positions, atomic_charges, cell,
        pbc.
    """
    # Calculate box size for desired density
    volume = num_atoms / density
    box_size = volume ** (1 / 3)

    # Random atomic positions
    positions = torch.rand(num_atoms, 3, dtype=dtype, device=device) * box_size

    # Random charges with some structure
    charges = torch.zeros(num_atoms, dtype=dtype, device=device)
    for i in range(0, num_atoms, 2):
        if i + 1 < num_atoms:
            charges[i] = 1.0
            charges[i + 1] = -1.0
        else:
            charges[i] = 0.0

    # Add small random perturbations
    charges += torch.randn(num_atoms, dtype=dtype, device=device) * 0.1
    charges = charges - charges.mean()  # Ensure neutrality

    # Random atomic numbers (mix of elements: C=6, N=7, O=8, F=9)
    atomic_numbers = torch.randint(
        6, 10, (num_atoms,), dtype=torch.int32, device=device
    )

    # Simulation cell
    cell = torch.zeros(1, 3, 3, dtype=dtype, device=device)
    cell[0, 0, 0] = box_size
    cell[0, 1, 1] = box_size
    cell[0, 2, 2] = box_size

    pbc = torch.tensor([periodic, periodic, periodic], dtype=torch.bool, device=device)

    return {
        "positions": positions,
        "atomic_charges": charges,
        "atomic_numbers": atomic_numbers,
        "cell": cell,
        "pbc": pbc,
        "system_type": "random",
        "num_atoms": num_atoms,
        "density": density,
        "periodic": periodic,
        "box_size": box_size,
    }


def create_test_systems(
    system_types: list[str],
    atom_counts: list[int],
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float64,
) -> dict[str, list[dict[str, torch.Tensor]]]:
    """Create multiple test systems for benchmarking.

    Parameters
    ----------
    system_types : List[str]
        List of system types to create ('molecular', 'crystal', 'random').
    atom_counts : List[int]
        List of atom counts for each system type.
    device : torch.device, default=CPU
        Device to create tensors on.
    dtype : torch.dtype, default=float64
        Data type for tensors.

    Returns
    -------
    Dict[str, List[Dict[str, torch.Tensor]]]
        Dictionary mapping system types to lists of systems.
    """
    systems = {}

    for system_type in system_types:
        systems[system_type] = []

        for num_atoms in atom_counts:
            if system_type == "molecular":
                system = create_molecular_system(num_atoms, device=device, dtype=dtype)
            elif system_type == "crystal":
                system = create_crystal_system(num_atoms, device=device, dtype=dtype)
            elif system_type == "random":
                system = create_random_system(
                    num_atoms, periodic=True, device=device, dtype=dtype
                )
            elif system_type == "random_nonperiodic":
                system = create_random_system(
                    num_atoms, periodic=False, device=device, dtype=dtype
                )
            else:
                raise ValueError(f"Unknown system type: {system_type}")

            systems[system_type].append(system)

    return systems


def create_batch_systems(
    num_systems: int,
    atoms_per_system: int,
    system_type: str = "molecular",
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Create batch systems for batch interaction benchmarking.

    Parameters
    ----------
    num_systems : int
        Number of systems in the batch.
    atoms_per_system : int
        Target number of atoms per system.
    system_type : str, default="molecular"
        Type of systems to create: "molecular", "crystal", or "random".
    dtype : torch.dtype, default=torch.float32
        Data type for positions and other floating-point tensors.
    device : torch.device, default=torch.device('cpu')
        Device to create tensors on.

    Returns
    -------
    dict
        Dictionary containing batch system data:
        - 'positions': (total_atoms, 3)
        - 'batch_atomic_charges': (total_atoms,)
        - 'batch_atomic_numbers': (total_atoms,)
        - 'cells': (num_systems, 3, 3)
        - 'pbc': (num_systems, 3)
        - 'batch_ptr': (num_systems+1,) CSR format
        - 'cutoffs': (num_systems,) suggested cutoffs
    """
    all_positions = []
    all_charges = []
    all_atomic_numbers = []
    all_cells = []
    all_pbc = []
    all_cutoffs = []
    system_boundaries = [0]

    for system_idx in range(num_systems):
        # Create individual system with slight variations
        if system_type == "molecular":
            system = create_molecular_system(
                atoms_per_system, dtype=dtype, device=device
            )
            suggested_cutoff = 8.0
        elif system_type == "crystal":
            system = create_crystal_system(atoms_per_system, dtype=dtype, device=device)
            suggested_cutoff = 6.0
        else:  # random
            system = create_random_system(atoms_per_system, dtype=dtype, device=device)
            suggested_cutoff = 5.0

        # Add slight variations to avoid identical systems
        variation = system_idx * 0.1
        system["positions"] += torch.randn_like(system["positions"]) * variation

        # Collect system data
        all_positions.append(system["positions"])
        all_charges.append(system["atomic_charges"])
        all_atomic_numbers.append(system["atomic_numbers"])
        all_cells.append(system["cell"][0])  # Remove batch dimension
        all_pbc.append(system["pbc"])
        all_cutoffs.append(suggested_cutoff + variation)
        system_boundaries.append(system_boundaries[-1] + system["positions"].shape[0])

    # Concatenate batch data
    positions = torch.cat(all_positions, dim=0)
    batch_charges = torch.cat(all_charges, dim=0)
    batch_atomic_numbers = torch.cat(all_atomic_numbers, dim=0)
    cells = torch.stack(all_cells, dim=0)
    pbc = torch.stack(all_pbc, dim=0)
    cutoffs = torch.tensor(all_cutoffs, dtype=dtype, device=device)
    batch_ptr = torch.tensor(system_boundaries, dtype=torch.int32, device=device)

    return {
        "positions": positions,
        "batch_atomic_charges": batch_charges,
        "batch_atomic_numbers": batch_atomic_numbers,
        "cells": cells,
        "pbc": pbc,
        "batch_ptr": batch_ptr,
        "cutoffs": cutoffs,
        "total_atoms": positions.shape[0],
        "num_systems": num_systems,
        "atoms_per_system_avg": positions.shape[0] / num_systems,
    }


def create_benchmark_dataset(
    domain: Literal["molecule", "crystal"], **kwargs
) -> BenchmarkDataset:
    """Factory function to create benchmark datasets.

    This is the main entry point for creating atomistic benchmark datasets.
    Each dataset type processes different kinds of systems and provides
    consistent PyTorch tensor outputs for benchmarking purposes.

    Parameters
    ----------
    domain : {"molecule", "crystal"}
        Type of atomistic system to create dataset for:
        - "molecule": Small organic molecules from SMILES strings
        - "crystal": Periodic crystal structures from COD database
    **kwargs
        Additional arguments passed to dataset constructor. Common options:

        For all datasets:
        - cache_dir : Path | str | None
            Directory for caching computed data (default: temp directory)
        - device : torch.device
            Device to place tensors on (default: CPU)
        - dtype : torch.dtype
            Data type for floating-point tensors (default: float32)
        - force_recompute : bool
            Whether to recompute even if cached data exists (default: False)

        For molecule datasets:
        - num_conformers : int
            Number of conformers to generate per molecule (default: 1)

        For crystal datasets:
        - max_supercell_size : int | None
            Maximum supercell size for neighbor list benchmarking (default: None)

    Returns
    -------
    BenchmarkDataset
        Initialized dataset for the specified domain. The dataset implements
        torch.utils.data.Dataset interface:
        - len(dataset): Number of systems
        - dataset[i]: System data as dict of PyTorch tensors
        - dataset.get_system_info(i): Metadata about system i

    Examples
    --------
    Basic usage:

    >>> # Create molecule dataset (requires MOLECULE_SMILES to be populated)
    >>> mol_dataset = create_benchmark_dataset("molecule")
    >>> logger.info(f"Dataset contains {len(mol_dataset)} molecules")
    >>>
    >>> # Get first system
    >>> system = mol_dataset[0]
    >>> coords = system["positions"]  # Shape: (num_atoms, 3)
    >>> atomic_nums = system["atomic_numbers"]  # Shape: (num_atoms,)

    Advanced configuration:

    >>> # Crystal dataset with supercells for neighbor list benchmarking
    >>> crystal_dataset = create_benchmark_dataset(
    ...     "crystal",
    ...     max_supercell_size=1000,
    ...     cache_dir="/path/to/cache",
    ...     device=torch.device("cuda:0"),
    ...     dtype=torch.float64
    ... )

    Caching behavior:

    >>> # Force recomputation (useful when system lists change)
    >>> dataset = create_benchmark_dataset(
    ...     "molecule",
    ...     force_recompute=True
    ... )

    Notes
    -----
    - First run will download/compute all systems and cache results
    - Subsequent runs load from cache for fast startup
    - Cache files are invalidated when system lists or configuration changes
    - Internet connection required for initial crystal downloads
    - All output tensors follow consistent naming and shape conventions

    Raises
    ------
    ValueError
        If domain is not one of the supported types
    ImportError
        If required dependencies are not installed (rdkit, pymatgen)
    """
    if domain == "molecule":
        return MoleculeDataset(**kwargs)
    elif domain == "crystal":
        return CrystalDataset(**kwargs)
    else:
        raise ValueError(f"Unknown domain: {domain}")
