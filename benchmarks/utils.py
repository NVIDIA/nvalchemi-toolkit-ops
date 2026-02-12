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
import argparse
import csv
import signal
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from pymatgen.core import Lattice, Structure
from torch.cuda import cudart


class TimeoutError(Exception):
    """Exception raised when a benchmark times out."""

    pass


@contextmanager
def timeout(seconds):
    """Context manager for timing out a code block.

    Parameters
    ----------
    seconds : int
        Number of seconds before timeout.

    Raises
    ------
    TimeoutError
        If the code block takes longer than the specified timeout.

    Notes
    -----
    Uses SIGALRM on Unix systems. On Windows, this is a no-op (no timeout).
    """

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    # Set up signal handler (Unix only)
    if hasattr(signal, "SIGALRM"):
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # No timeout on Windows
        yield


class BenchmarkTimer:
    """High-precision timing utility with CPU/GPU synchronization support.

    Includes graceful error handling for timeouts and CUDA OOM errors.
    """

    def __init__(
        self,
        device: torch.device,
        warmup_runs: int = 3,
        timing_runs: int = 10,
        timeout_seconds: int = 60,
    ):
        """Initialize benchmark timer.

        Parameters
        ----------
        device : torch.device
            Device for computation (CPU or CUDA).
        warmup_runs : int, default=3
            Number of warmup runs before timing.
        timing_runs : int, default=10
            Number of timing runs for averaging.
        timeout_seconds : int, default=60
            Maximum time (in seconds) allowed for a single benchmark run.
            If exceeded, the benchmark will fail gracefully.
        """
        self.device = device
        self.warmup_runs = warmup_runs
        self.timing_runs = timing_runs
        self.timeout_seconds = timeout_seconds
        self.is_cuda = device.type == "cuda"

    def time_function(self, func, *args, **kwargs) -> dict[str, float | None]:
        """Time a function with proper synchronization and error handling.

        Parameters
        ----------
        func : callable
            Function to time.
        *args, **kwargs
            Arguments to pass to function.

        Returns
        -------
        dict[str, float | None]
            Timing statistics including median time in milliseconds, or None if failed.
            Additional keys:
            - "median": Median time in milliseconds (or None if failed)
            - "times": List of individual run times (empty if failed)
            - "peak_memory_mb": Peak GPU memory usage in MB (or None if CPU/failed)
            - "success": Boolean indicating if benchmark completed successfully
            - "error": Error message if benchmark failed (optional)
            - "error_type": Type of error that occurred (optional)

        Notes
        -----
        This method handles:
        - CUDA Out of Memory errors (gracefully caught and reported)
        - Timeout errors (if execution exceeds timeout_seconds)
        - General exceptions during warmup or timing
        """
        try:
            # Warmup runs with timeout
            with timeout(self.timeout_seconds):
                for i in range(self.warmup_runs):
                    try:
                        _ = func(*args, **kwargs)
                        if self.is_cuda:
                            torch.cuda.synchronize()
                    except torch.cuda.OutOfMemoryError:
                        if self.is_cuda:
                            torch.cuda.empty_cache()
                        return {
                            "median": None,
                            "times": [],
                            "peak_memory_mb": None,
                            "success": False,
                            "error": f"CUDA Out of Memory during warmup run {i + 1}",
                            "error_type": "OOM",
                        }
                    except Exception as e:
                        return {
                            "median": None,
                            "times": [],
                            "peak_memory_mb": None,
                            "success": False,
                            "error": f"Warmup run {i + 1} failed: {str(e)}",
                            "error_type": type(e).__name__,
                        }

            # Reset peak memory stats before timing runs
            if self.is_cuda:
                torch.cuda.reset_peak_memory_stats()

            # Timing runs
            times = []
            cudart().cudaProfilerStart()

            for i in range(self.timing_runs):
                try:
                    with timeout(self.timeout_seconds):
                        if self.is_cuda:
                            torch.cuda.synchronize()
                            start_event = torch.cuda.Event(enable_timing=True)
                            end_event = torch.cuda.Event(enable_timing=True)

                            start_event.record()
                            func(*args, **kwargs)
                            end_event.record()

                            torch.cuda.synchronize()
                            elapsed_time = start_event.elapsed_time(
                                end_event
                            )  # milliseconds
                            times.append(elapsed_time)
                        else:
                            start_time = time.perf_counter()
                            func(*args, **kwargs)
                            end_time = time.perf_counter()
                            times.append(
                                (end_time - start_time) * 1000.0
                            )  # Convert to ms

                except torch.cuda.OutOfMemoryError:
                    if self.is_cuda:
                        torch.cuda.empty_cache()
                    cudart().cudaProfilerStop()
                    return {
                        "median": None,
                        "times": times,  # Return partial results
                        "peak_memory_mb": None,
                        "success": False,
                        "error": f"CUDA Out of Memory during timing run {i + 1}/{self.timing_runs}",
                        "error_type": "OOM",
                    }
                except TimeoutError as e:
                    cudart().cudaProfilerStop()
                    return {
                        "median": None,
                        "times": times,  # Return partial results
                        "peak_memory_mb": None,
                        "success": False,
                        "error": f"Timeout during timing run {i + 1}/{self.timing_runs}: {str(e)}",
                        "error_type": "Timeout",
                    }
                except Exception as e:
                    cudart().cudaProfilerStop()
                    return {
                        "median": None,
                        "times": times,  # Return partial results
                        "peak_memory_mb": None,
                        "success": False,
                        "error": f"Timing run {i + 1} failed: {str(e)}",
                        "error_type": type(e).__name__,
                    }

            cudart().cudaProfilerStop()

            # Get peak memory usage
            peak_memory_mb = None
            if self.is_cuda:
                peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)

            if not times:
                return {
                    "median": None,
                    "times": [],
                    "peak_memory_mb": peak_memory_mb,
                    "success": False,
                    "error": "No successful timing runs",
                    "error_type": "NoData",
                }

            return {
                "median": float(np.median(times)),
                "times": times,
                "peak_memory_mb": peak_memory_mb,
                "success": True,
            }

        except TimeoutError as e:
            return {
                "median": None,
                "times": [],
                "peak_memory_mb": None,
                "success": False,
                "error": f"Overall timeout: {str(e)}",
                "error_type": "Timeout",
            }
        except Exception as e:
            return {
                "median": None,
                "times": [],
                "peak_memory_mb": None,
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "error_type": type(e).__name__,
            }


def save_benchmark_results(
    results: list[dict], output_file: str, method_name: str
) -> None:
    """Save benchmark results to CSV file.

    Parameters
    ----------
    results : List[Dict]
        List of benchmark result dictionaries.
    output_file : str
        Path to output CSV file.
    method_name : str
        Name of the benchmarked method.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        print(f"No results to save for {method_name}")
        return

    # Get all unique keys from all results
    all_keys = set()
    for result in results:
        all_keys.update(result.keys())

    fieldnames = ["method"] + sorted(all_keys)

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            row = {"method": method_name, **result}
            writer.writerow(row)

    print(f"Saved {len(results)} benchmark results to {output_path}")


def parse_benchmark_args() -> argparse.Namespace:
    """Parse standard benchmark command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Benchmark cuAlchemi methods")

    parser.add_argument(
        "--system-types",
        nargs="+",
        default=["crystal", "random"],
        choices=["molecular", "crystal", "random", "random_nonperiodic"],
        help="System types to benchmark",
    )

    parser.add_argument(
        "--atom-counts",
        nargs="+",
        type=int,
        default=[64, 128, 256, 512, 1024, 2048, 4096, 10000, 20000, 50000, 100000],
        help="Atom counts to benchmark",
    )

    parser.add_argument(
        "--device", default="cuda:0", help="Device for computation (cuda:0, cpu, etc.)"
    )

    parser.add_argument(
        "--dtype",
        default="float64",
        choices=["float16", "float32", "float64"],
        help="Data type for computations",
    )

    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Directory for output CSV files",
    )

    parser.add_argument(
        "--warmup-runs", type=int, default=3, help="Number of warmup runs"
    )

    parser.add_argument(
        "--timing-runs",
        type=int,
        default=10,
        help="Number of timing runs for averaging",
    )

    parser.add_argument(
        "--test-compile", action="store_true", help="Test torch.compile compatibility"
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    return parser.parse_args()


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert string to torch dtype.

    Parameters
    ----------
    dtype_str : str
        String representation of dtype.

    Returns
    -------
    torch.dtype
        Corresponding PyTorch dtype.
    """
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    return dtype_map[dtype_str]


def print_system_info(system: dict[str, torch.Tensor], verbose: bool = False) -> None:
    """Print information about a test system.

    Parameters
    ----------
    system : Dict[str, torch.Tensor]
        System dictionary.
    verbose : bool, default=False
        Whether to print detailed information.
    """
    num_atoms = system["num_atoms"]
    system_type = system["system_type"]
    device = system["positions"].device
    dtype = system["positions"].dtype

    print(f"System: {system_type}, {num_atoms} atoms, {device}, {dtype}")

    if verbose:
        if "density" in system:
            print(f"  Density: {system['density']:.3f}")
        if "box_size" in system:
            print(f"  Box size: {system['box_size']:.3f}")
        if "lattice_type" in system:
            print(f"  Lattice: {system['lattice_type']}")
        print(f"  Periodic: {system['pbc'].tolist()}")
        print(f"  Total charge: {system['atomic_charges'].sum().item():.6f}")


def get_memory_usage(device: torch.device) -> float | None:
    """Get current GPU memory usage in GB.

    Parameters
    ----------
    device : torch.device
        Device to check memory for.

    Returns
    -------
    Optional[float]
        Memory usage in GB, or None if not CUDA.
    """
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / 1024**3
    return None


def print_benchmark_summary(
    results: list[dict], method_name: str, verbose: bool = False
) -> None:
    """Print a formatted summary of benchmark results.

    Parameters
    ----------
    results : List[Dict]
        List of benchmark result dictionaries.
    method_name : str
        Name of the benchmarked method category.
    verbose : bool, default=False
        Whether to print detailed breakdown.
    """
    if not results:
        print(f"\nNo benchmark results for {method_name}!")
        return

    print(f"\n{'=' * 60}")
    print(f"{method_name.upper()} BENCHMARK SUMMARY")
    print(f"{'=' * 60}")

    # Overall statistics
    total_tests = len(results)
    avg_time = (
        sum(r.get("time_mean", r.get("compile_total_time_mean", 0)) for r in results)
        / total_tests
    )

    functions = {r.get("function", "unknown") for r in results}
    atom_counts = sorted({r.get("num_atoms", 0) for r in results})
    system_types = {r.get("system_type", "unknown") for r in results}

    print(f"Total tests: {total_tests}")
    print(f"Functions tested: {', '.join(functions)}")
    print(f"System types: {', '.join(system_types)}")
    print(
        f"Atom counts: {min(atom_counts)} - {max(atom_counts)} ({len(atom_counts)} sizes)"
    )
    print(f"Average time: {avg_time:.4f}s")

    # Summary by function
    print("\nPERFORMANCE BY FUNCTION:")
    print(
        f"{'Function':<25} {'Avg Time (ms)':<12} {'Std Dev (ms)':<10} {'Best Perf':<15}"
    )
    print(f"{'-' * 62}")

    by_function = {}
    for result in results:
        func = result.get("function", "unknown")
        if func not in by_function:
            by_function[func] = []
        by_function[func].append(result)

    for func, func_results in by_function.items():
        times = [r["time_mean"] for r in func_results if "time_mean" in r] + [
            r["compile_total_time_mean"]
            for r in func_results
            if "compile_total_time_mean" in r
        ]
        if times:
            avg_time = np.mean(times)
            std_time = np.std(times)
            best_perf_key = (
                "pairs_per_sec"
                if "pairs_per_sec" in func_results[0]
                else "atoms_per_sec"
            )
            if best_perf_key in func_results[0]:
                best_perf = max(r[best_perf_key] for r in func_results)
                best_perf_str = f"{best_perf:.0f} {best_perf_key.split('_')[0]}/s"
            else:
                best_perf_str = "N/A"

            print(f"{func:<25} {avg_time:<12.4f} {std_time:<10.4f} {best_perf_str:<15}")

    # Summary by atom count
    if len(atom_counts) > 1:
        print("\nSCALING BY ATOM COUNT:")
        print(f"{'Atoms':<8} {'Avg Time (ms)':<12} {'Best Perf':<15}")
        print(f"{'-' * 35}")

        by_atoms = {}
        for result in results:
            atoms = result.get("num_atoms", 0)
            if atoms not in by_atoms:
                by_atoms[atoms] = []
            by_atoms[atoms].append(result)

        for atoms in sorted(by_atoms.keys()):
            atom_results = by_atoms[atoms]
            times = [r["time_mean"] for r in atom_results if "time_mean" in r] + [
                r["compile_total_time_mean"]
                for r in atom_results
                if "compile_total_time_mean" in r
            ]
            if times:
                avg_time = np.mean(times)
                # Find best performance metric
                perf_metrics = ["pairs_per_sec", "atoms_per_sec", "kvectors_per_sec"]
                best_perf_str = "N/A"
                for metric in perf_metrics:
                    if any(metric in r for r in atom_results):
                        best_perf = max(r[metric] for r in atom_results if metric in r)
                        best_perf_str = f"{best_perf:1.3e} {metric.split('_')[0]}/s"
                        break

                print(f"{atoms:<8} {avg_time:<12.4f} {best_perf_str:<15}")

    # Summary by cutoff (if applicable)
    cutoff_keys = ["cutoff", "real_cutoff", "dispersion_cutoff", "cutoff"]
    cutoff_key = None
    for key in cutoff_keys:
        if any(key in r for r in results):
            cutoff_key = key
            break

    if cutoff_key and verbose:
        cutoffs = sorted({r[cutoff_key] for r in results if cutoff_key in r})
        if len(cutoffs) > 1:
            print(f"\nSCALING BY {cutoff_key.upper()}:")
            print(f"{'Cutoff':<8} {'Avg Time (ms)':<12} {'Avg Pairs':<12}")
            print(f"{'-' * 32}")

            by_cutoff = {}
            for result in results:
                cutoff = result.get(cutoff_key)
                if cutoff is not None:
                    if cutoff not in by_cutoff:
                        by_cutoff[cutoff] = []
                    by_cutoff[cutoff].append(result)

            for cutoff in sorted(by_cutoff.keys()):
                cutoff_results = by_cutoff[cutoff]
                times = [r["time_mean"] for r in cutoff_results if "time_mean" in r] + [
                    r["compile_total_time_mean"]
                    for r in cutoff_results
                    if "compile_total_time_mean" in r
                ]
                pairs = [r.get("num_pairs", 0) for r in cutoff_results]
                if times:
                    avg_time = np.mean(times)
                    avg_pairs = np.mean(pairs) if pairs else 0
                    print(f"{cutoff:<8.1f} {avg_time:<12.4f} {avg_pairs:<12.0f}")

    # torch.compile summary (if applicable)
    compile_results = [r for r in results if r.get("compile_compatible") is not None]
    if compile_results:
        compatible_count = len(
            [r for r in compile_results if r.get("compile_compatible")]
        )
        total_compile_tests = len(compile_results)

        print("\nTORCH.COMPILE COMPATIBILITY:")
        print(f"Compatible: {compatible_count}/{total_compile_tests} functions")

        if compatible_count > 0:
            speedups = [
                r.get("compile_speedup", 1.0)
                for r in compile_results
                if r.get("compile_compatible") and r.get("compile_speedup")
            ]
            if speedups:
                avg_speedup = np.mean(speedups)
                max_speedup = np.max(speedups)
                print(f"Average speedup: {avg_speedup:.2f}x")
                print(f"Best speedup: {max_speedup:.2f}x")

    # Memory usage summary (if applicable)
    memory_results = [
        r for r in results if "memory_gb" in r and r["memory_gb"] is not None
    ]
    if memory_results:
        avg_memory = np.mean([r["memory_gb"] for r in memory_results])
        max_memory = np.max([r["memory_gb"] for r in memory_results])
        print("\nMEMORY USAGE:")
        print(f"Average GPU memory: {avg_memory:.2f} GB")
        print(f"Peak GPU memory: {max_memory:.2f} GB")

    print(f"{'=' * 60}")


def format_performance_table(results: list[dict], group_by: str = "function") -> str:
    """Format results into a performance comparison table.

    Parameters
    ----------
    results : List[Dict]
        Benchmark results.
    group_by : str, default='function'
        Key to group results by.

    Returns
    -------
    str
        Formatted table string.
    """
    if not results:
        return "No results to format"

    # Group results
    grouped = {}
    for result in results:
        key = result.get(group_by, "unknown")
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(result)

    # Create table
    header = f"{'Method':<20} {'Avg Time (ms)':<12} {'Best Perf':<15}"
    separator = "-" * len(header)
    rows = [header, separator]

    for key, group_results in grouped.items():
        times = [r["time_mean"] for r in group_results if "time_mean" in r] + [
            r["compile_total_time_mean"]
            for r in group_results
            if "compile_total_time_mean" in r
        ]
        if times:
            avg_time = np.mean(times)

            # Find best performance metric
            best_perf_str = "N/A"
            for metric in ["atoms_per_sec", "pairs_per_sec", "kvectors_per_sec"]:
                if any(metric in r for r in group_results):
                    best_perf = max(r[metric] for r in group_results if metric in r)
                    best_perf_str = f"{best_perf:1.3e} {metric.split('_')[0]}/s"
                    break

            row = f"{key:<20} {avg_time:<12.4f} {best_perf_str:<15}"
            rows.append(row)

    return "\n".join(rows)


# ==============================================================================
# Pymatgen Structure Utilities
# ==============================================================================


def create_bulk_structure(
    symbol: str, crystal_type: str, a: float, cubic: bool = False
) -> Structure:
    """Create a bulk crystal structure using pymatgen.

    Creates standard crystal structures with common lattice types.

    Parameters
    ----------
    symbol : str
        Chemical symbol of the element (e.g., "Al", "Fe").
    crystal_type : str
        Crystal structure type. Supported: "fcc", "bcc", "sc" (simple cubic).
    a : float
        Lattice constant in Angstroms.
    cubic : bool, default=False
        If True, create a cubic supercell for non-cubic structures.

    Returns
    -------
    Structure
        pymatgen Structure object representing the bulk crystal.

    Examples
    --------
    >>> fcc_al = create_bulk_structure("Al", "fcc", a=4.05)
    >>> bcc_fe = create_bulk_structure("Fe", "bcc", a=2.87, cubic=True)
    """
    lattice = Lattice.cubic(a)

    if crystal_type.lower() == "fcc":
        # Face-centered cubic: atoms at corners and face centers
        coords = np.array(
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]
        )
        species = [symbol] * 4
    elif crystal_type.lower() == "bcc":
        # Body-centered cubic: atoms at corners and body center
        coords = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        species = [symbol] * 2
    elif crystal_type.lower() in ["sc", "simple_cubic"]:
        # Simple cubic: atom at corner only
        coords = np.array([[0.0, 0.0, 0.0]])
        species = [symbol]
    else:
        raise ValueError(
            f"Unsupported crystal type: {crystal_type}. "
            "Supported types: 'fcc', 'bcc', 'sc' (simple cubic)"
        )

    structure = Structure(lattice, species, coords, coords_are_cartesian=False)

    return structure


def get_structure_atomic_numbers(structure: Structure) -> np.ndarray:
    """Extract atomic numbers from a pymatgen Structure.

    Parameters
    ----------
    structure : Structure
        pymatgen Structure object.

    Returns
    -------
    np.ndarray
        Array of atomic numbers (integers) with shape (num_atoms,).

    Examples
    --------
    >>> structure = create_bulk_structure("Al", "fcc", a=4.05)
    >>> atomic_numbers = get_structure_atomic_numbers(structure)
    >>> print(atomic_numbers)  # [13, 13, 13, 13] for Al
    """
    return np.array([site.specie.Z for site in structure], dtype=np.int32)


def create_molecule_structure(name: str, box_size: float = 10.0) -> Structure:
    """Create a simple molecular structure with predefined coordinates.

    Provides common small molecules with approximate equilibrium geometries.

    Parameters
    ----------
    name : str
        Name of the molecule. Supported: "H2O", "CO2", "CH4".
    box_size : float, default=10.0
        Size of the cubic box in Angstroms (used for periodic boundary conditions).

    Returns
    -------
    Structure
        pymatgen Structure object with the molecule centered in a box.

    Raises
    ------
    ValueError
        If the molecule name is not supported.

    Notes
    -----
    Coordinates are approximate equilibrium geometries and not optimized.
    The molecules are placed in a cubic box with periodic boundary conditions.

    Examples
    --------
    >>> water = create_molecule_structure("H2O", box_size=15.0)
    >>> co2 = create_molecule_structure("CO2")
    """
    # Define molecule coordinates (Cartesian, in Angstroms)
    # Centered around origin, will be shifted to box center
    molecules = {
        "H2O": {
            "species": ["O", "H", "H"],
            "coords": np.array(
                [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]
            ),
        },
        "CO2": {
            "species": ["C", "O", "O"],
            "coords": np.array([[0.0, 0.0, 0.0], [1.16, 0.0, 0.0], [-1.16, 0.0, 0.0]]),
        },
        "CH4": {
            "species": ["C", "H", "H", "H", "H"],
            "coords": np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.629, 0.629, 0.629],
                    [-0.629, -0.629, 0.629],
                    [-0.629, 0.629, -0.629],
                    [0.629, -0.629, -0.629],
                ]
            ),
        },
    }

    if name not in molecules:
        raise ValueError(
            f"Unsupported molecule: {name}. "
            f"Supported molecules: {list(molecules.keys())}"
        )

    mol_data = molecules[name]
    species = mol_data["species"]
    coords = mol_data["coords"]

    # Center the molecule in the box
    coords_centered = coords + box_size / 2.0

    # Create cubic lattice
    lattice = Lattice.cubic(box_size)

    # Create structure with Cartesian coordinates
    structure = Structure(lattice, species, coords_centered, coords_are_cartesian=True)

    return structure
