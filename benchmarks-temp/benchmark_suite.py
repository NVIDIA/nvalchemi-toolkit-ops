#!/usr/bin/env python3
"""
Unified Benchmark Suite for NL, D3, and Electrostatics
=======================================================
Comprehensive benchmarking with correct CUDA timing patterns.

Usage:
    python benchmark_suite.py --help
    python benchmark_suite.py --benchmark nl --cutoffs 6 15
    python benchmark_suite.py --benchmark d3 --cutoffs 15 25
    python benchmark_suite.py --benchmark electrostatics --accuracy 1e-4 1e-6
    python benchmark_suite.py --benchmark all --timing-runs 20

Output:
    Creates timestamped directory: benchmarks_YYYY-MM-DD_HH-MM-SS/
    With CSVs for each benchmark type.

Timing Pattern (verified by senior engineer):
    - Batch timing: start.record() → N × fn() → end.record() → sync()
    - Sync is OUTSIDE the loop to avoid overhead in measurements

Unit System (Electrostatics):
    INPUT:  positions [Å], charges [e], cell [Å]
    OUTPUT: energy [e²/Å], forces [e²/Å²]
    To convert to eV: multiply by k_e = 14.3996 eV·Å/e²
"""

import argparse
import csv
import gc
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import warp as wp
from natsort import natsorted

# Official APIs from NVIDIA Alchemi Toolkit-Ops
from nvalchemiops.neighborlist import (
    batch_naive_neighbor_list,
    batch_cell_list,
    estimate_max_neighbors,
)
from nvalchemiops.interactions.dispersion import dftd3

# Electrostatics APIs
from nvalchemiops.interactions.electrostatics import (
    particle_mesh_ewald,
    ewald_summation,
    ewald_real_space,
    ewald_reciprocal_space,
    generate_k_vectors_ewald_summation,
    generate_k_vectors_pme,  # For pre-generating PME k-vectors
    estimate_pme_parameters,
    estimate_ewald_parameters,
)

# GPU memory tracking
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    PYNVML_AVAILABLE = True
except ImportError:
    print("WARNING: pynvml not installed. Install with: pip install nvidia-ml-py")
    PYNVML_AVAILABLE = False
    GPU_HANDLE = None

# ============== CONSTANTS ==============
ELEMENT_Z = {'H': 1, 'N': 7}
ANGSTROM_TO_BOHR = 1.8897259886
D3_A1, D3_A2, D3_S8 = 0.4145, 4.8593, 1.2177  # PBE functional

# Electrostatics constants
PARTIAL_CHARGES = {'H': 0.3, 'N': -0.9}  # Neutral NH3 molecule
PME_SPLINE_ORDER = 4  # B-spline order for PME
COULOMB_CONST = 14.3996447794  # eV·Å/e² (k_e = e²/4πε₀)

# OOM prevention: skip high accuracy for large systems
SKIP_ACCURACY_FOR_LARGE_SYSTEMS = {
    1e-6: 131072,  # Skip 1e-6 for systems >= 128k atoms
}

# Relative paths from script directory
SCRIPT_DIR = Path(__file__).parent
DEFAULT_NH3_DIR = SCRIPT_DIR / 'nh3'
DEFAULT_OUTPUT_BASE = SCRIPT_DIR / 'benchmark-results'
DEFAULT_D3_PARAMS = Path.home() / '.cache/nvalchemiops/dftd3_parameters.pt'


# ============== UTILITIES ==============
def get_timestamp():
    """Generate human-readable timestamp: YYYY-MM-DD_HH-MM-SS"""
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def clean_gpu():
    """Aggressively clean GPU memory."""
    torch.cuda.synchronize()
    wp.synchronize()
    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def get_gpu_mem():
    """Get GPU memory usage."""
    if PYNVML_AVAILABLE:
        info = pynvml.nvmlDeviceGetMemoryInfo(GPU_HANDLE)
        return {'used': info.used, 'total': info.total, 'percent': 100.0 * info.used / info.total}
    else:
        used = torch.cuda.memory_allocated()
        total = torch.cuda.get_device_properties(0).total_memory
        return {'used': used, 'total': total, 'percent': 100.0 * used / total}


def format_num(n):
    """Format atom count using binary prefix (1024-based)."""
    if n >= 1024:
        return f'{n // 1024}k'
    return str(n)


def cuda_timed_runs(fn, num_runs):
    """
    Time a function using CUDA events with CORRECT batch timing pattern.
    
    Pattern (verified by senior engineer - DF changes):
        1. sync() - ensure GPU is idle
        2. start.record() - record start timestamp (ONCE)
        3. for N runs: fn() - execute function N times
        4. end.record() - record end timestamp (ONCE, OUTSIDE loop)
        5. sync() - wait for GPU to finish (ONCE, OUTSIDE loop)
        6. elapsed_time = (end - start) / N
    
    This measures sustained throughput without sync overhead pollution.
    
    Returns:
        float: Mean time per run in seconds
    """
    # Initial sync to ensure GPU is idle
    torch.cuda.synchronize()
    wp.synchronize()
    
    # Create timing events (ONCE, outside loop)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    # Record start timestamp
    start.record()
    
    # Execute function N times (NO sync inside loop!)
    for _ in range(num_runs):
        fn()
    
    # Record end timestamp (OUTSIDE loop)
    end.record()
    
    # Wait for GPU to finish (ONCE, OUTSIDE loop)
    torch.cuda.synchronize()
    wp.synchronize()
    
    # Calculate mean time per run
    elapsed_ms = start.elapsed_time(end)  # Total time for all runs
    mean_time_sec = (elapsed_ms / 1000.0) / num_runs
    
    return mean_time_sec


# ============== DATA LOADING ==============
def parse_pdb(path):
    """Parse PDB file, return coords, atomic numbers, elements, and cell."""
    lines = Path(path).read_text().splitlines()
    coords, numbers, elements, cell = [], [], [], None
    
    for line in lines:
        if line.startswith('CRYST1'):
            p = line.split()
            cell = np.diag([float(p[1]), float(p[2]), float(p[3])]).astype(np.float32)
        if line.startswith(('HETATM', 'ATOM')):
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            el = line[76:78].strip() if len(line) >= 78 else line[12:14].strip()[0]
            elements.append(el)
            numbers.append(ELEMENT_Z.get(el, 1))
    
    return np.asarray(coords, np.float32), np.asarray(numbers, np.int32), elements, cell


def create_batch(pdb_path, batch_size, device):
    """Create batched data for benchmarks."""
    coords, numbers, elements, cell = parse_pdb(pdb_path)
    n = len(numbers)
    
    return {
        'coord': torch.tensor(np.tile(coords, (batch_size, 1)), dtype=torch.float32, device=device),
        'cell': torch.tensor(np.tile(cell[None], (batch_size, 1, 1)), dtype=torch.float32, device=device),
        'pbc': torch.ones(batch_size, 3, dtype=torch.bool, device=device),
        'batch_idx': torch.tensor(np.repeat(np.arange(batch_size, dtype=np.int32), n), device=device),
        'numbers': torch.tensor(np.tile(numbers, batch_size), dtype=torch.int32, device=device),
        'elements': elements,
        'atoms_per_system': n,
        'total_atoms': n * batch_size,
        'batch_size': batch_size,
        'cell_size': np.diag(cell).tolist(),
    }


def create_batch_electrostatics(pdb_path, batch_size, device):
    """Create batched data including charges for electrostatics benchmarks."""
    coords, numbers, elements, cell = parse_pdb(pdb_path)
    n = len(numbers)
    
    # Create charges from element symbols
    charges = np.array([PARTIAL_CHARGES.get(el, 0.0) for el in elements], dtype=np.float64)
    
    return {
        'coord': torch.tensor(np.tile(coords, (batch_size, 1)), dtype=torch.float32, device=device),
        'cell': torch.tensor(np.tile(cell[None], (batch_size, 1, 1)), dtype=torch.float32, device=device),
        'pbc': torch.ones(batch_size, 3, dtype=torch.bool, device=device),
        'batch_idx': torch.tensor(np.repeat(np.arange(batch_size, dtype=np.int32), n), device=device),
        'charges': torch.tensor(np.tile(charges, batch_size), dtype=torch.float64, device=device),
        'elements': elements,
        'atoms_per_system': n,
        'total_atoms': n * batch_size,
        'batch_size': batch_size,
        'cell_size': np.diag(cell).tolist(),
    }


def generate_neighbor_list(data, cutoff):
    """Generate neighbor list in LIST format (COO with pointer)."""
    maxnb = estimate_max_neighbors(cutoff, atomic_density=0.2, safety_factor=1.0)
    
    neighbor_list, neighbor_ptr, neighbor_shifts = batch_naive_neighbor_list(
        positions=data['coord'],
        cutoff=cutoff,
        batch_idx=data['batch_idx'],
        pbc=data['pbc'],
        cell=data['cell'],
        max_neighbors=maxnb,
        return_neighbor_list=True,
    )
    return neighbor_list, neighbor_ptr, neighbor_shifts


# ============== NL BENCHMARK ==============
def benchmark_nl(data, cutoff, method, num_runs, warmup_runs=3):
    """
    Benchmark neighbor list construction.
    
    Uses correct batch timing pattern - no sync inside timing loop.
    """
    coord = data['coord']
    cell = data['cell']
    pbc = data['pbc']
    batch_idx = data['batch_idx']
    total_atoms = data['total_atoms']
    
    maxnb = estimate_max_neighbors(cutoff, atomic_density=0.2, safety_factor=1.0)
    nl_func = batch_cell_list if method == 'cell' else batch_naive_neighbor_list
    
    # Warmup runs (not timed)
    clean_gpu()
    for _ in range(warmup_runs):
        nl_func(positions=coord, cell=cell, pbc=pbc, cutoff=cutoff, 
                batch_idx=batch_idx, max_neighbors=maxnb)
    torch.cuda.synchronize()
    wp.synchronize()
    
    # Memory measurement (separate from timing)
    clean_gpu()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    result = nl_func(positions=coord, cell=cell, pbc=pbc, cutoff=cutoff, 
                     batch_idx=batch_idx, max_neighbors=maxnb)
    torch.cuda.synchronize()
    wp.synchronize()
    mem_peak = torch.cuda.max_memory_allocated()
    mem_delta = mem_peak - mem_before
    gpu_info = get_gpu_mem()
    
    # Timing with correct batch pattern
    def run_nl():
        nl_func(positions=coord, cell=cell, pbc=pbc, cutoff=cutoff, 
                batch_idx=batch_idx, max_neighbors=maxnb)
    
    mean_time = cuda_timed_runs(run_nl, num_runs)
    time_us = (mean_time * 1e6) / total_atoms
    
    return {
        'time_us': time_us,
        'neighbors': int(result[0].shape[1]) if hasattr(result[0], 'shape') else 0,
        'mem_delta_mb': mem_delta / 1024**2,
        'mem_percent': 100.0 * mem_delta / gpu_info['total'],
    }


# ============== D3 BENCHMARK ==============
def benchmark_d3(data, cutoff, d3_params, num_runs, warmup_runs=3):
    """
    Benchmark DFT-D3 dispersion calculation.
    
    Times NL and D3 separately, both using correct batch timing pattern.
    """
    clean_gpu()
    
    coord = data['coord']
    cell = data['cell']
    pbc = data['pbc']
    batch_idx = data['batch_idx']
    numbers = data['numbers']
    total_atoms = data['total_atoms']
    
    # Convert to Bohr
    coord_bohr = coord * ANGSTROM_TO_BOHR
    cell_bohr = cell * ANGSTROM_TO_BOHR
    cutoff_bohr = cutoff * ANGSTROM_TO_BOHR
    
    maxnb = estimate_max_neighbors(cutoff, atomic_density=0.2, safety_factor=1.0)
    
    # Warmup runs
    for _ in range(warmup_runs):
        nbmat, _, nbmat_shifts = batch_cell_list(
            positions=coord_bohr, cell=cell_bohr, pbc=pbc,
            cutoff=cutoff_bohr, batch_idx=batch_idx, max_neighbors=maxnb
        )
        _ = dftd3(
            positions=coord_bohr, cell=cell_bohr, numbers=numbers,
            batch_idx=batch_idx, neighbor_matrix=nbmat, neighbor_matrix_shifts=nbmat_shifts,
            d3_params=d3_params, a1=D3_A1, a2=D3_A2, s8=D3_S8
        )
    torch.cuda.synchronize()
    wp.synchronize()
    
    # Memory measurement
    clean_gpu()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    nbmat, _, nbmat_shifts = batch_cell_list(
        positions=coord_bohr, cell=cell_bohr, pbc=pbc,
        cutoff=cutoff_bohr, batch_idx=batch_idx, max_neighbors=maxnb
    )
    _ = dftd3(
        positions=coord_bohr, cell=cell_bohr, numbers=numbers,
        batch_idx=batch_idx, neighbor_matrix=nbmat, neighbor_matrix_shifts=nbmat_shifts,
        d3_params=d3_params, a1=D3_A1, a2=D3_A2, s8=D3_S8
    )
    torch.cuda.synchronize()
    wp.synchronize()
    mem_peak = torch.cuda.max_memory_allocated()
    mem_delta = mem_peak - mem_before
    gpu_info = get_gpu_mem()
    
    # Time NL with correct batch pattern
    def run_nl():
        nonlocal nbmat, nbmat_shifts
        nbmat, _, nbmat_shifts = batch_cell_list(
            positions=coord_bohr, cell=cell_bohr, pbc=pbc,
            cutoff=cutoff_bohr, batch_idx=batch_idx, max_neighbors=maxnb
        )
    
    mean_time_nl = cuda_timed_runs(run_nl, num_runs)
    time_nl_us = (mean_time_nl * 1e6) / total_atoms
    
    # Time D3 with correct batch pattern
    def run_d3():
        _ = dftd3(
            positions=coord_bohr, cell=cell_bohr, numbers=numbers,
            batch_idx=batch_idx, neighbor_matrix=nbmat, neighbor_matrix_shifts=nbmat_shifts,
            d3_params=d3_params, a1=D3_A1, a2=D3_A2, s8=D3_S8
        )
    
    mean_time_d3 = cuda_timed_runs(run_d3, num_runs)
    time_d3_us = (mean_time_d3 * 1e6) / total_atoms
    
    return {
        'time_nl_us': time_nl_us,
        'time_d3_us': time_d3_us,
        'mem_delta_mb': mem_delta / 1024**2,
        'mem_percent': 100.0 * mem_delta / gpu_info['total'],
    }


# ============== ELECTROSTATICS BENCHMARK ==============
def ewald_with_charge_gradients(positions, charges, cell, alpha, k_vectors,
                                 nl_data, nl_shifts, nl_ptr, 
                                 batch_idx, mask_value, compute_forces=True):
    """Ewald summation with charge gradients using lower-level APIs.
    
    The high-level ewald_summation doesn't expose compute_charge_gradients,
    so we use ewald_real_space + ewald_reciprocal_space directly.
    """
    # Ensure alpha has correct shape for batch mode
    if batch_idx is not None and alpha.dim() == 0:
        num_systems = batch_idx.max().item() + 1
        alpha = alpha.expand(num_systems)
    
    # LIST format kwargs
    nl_kwargs = {
        'neighbor_list': nl_data,
        'neighbor_ptr': nl_ptr,
        'neighbor_shifts': nl_shifts,
    }
    
    # Real-space with charge gradients
    rs = ewald_real_space(
        positions=positions, charges=charges, cell=cell,
        alpha=alpha, mask_value=mask_value, batch_idx=batch_idx,
        compute_forces=compute_forces, compute_charge_gradients=True,
        **nl_kwargs,
    )
    
    # Reciprocal-space with charge gradients
    rec = ewald_reciprocal_space(
        positions=positions, charges=charges, cell=cell,
        k_vectors=k_vectors, alpha=alpha, batch_idx=batch_idx,
        compute_forces=compute_forces, compute_charge_gradients=True,
    )
    
    return rs[0] + rec[0], rs[1] + rec[1], rs[2] + rec[2]


def benchmark_pme(data, nl_data, nl_shifts, nl_ptr, alpha, mesh_dimensions,
                  accuracy, compute_charge_gradients, num_runs, warmup_runs=3):
    """Benchmark PME using correct batch timing pattern."""
    positions = data['coord'].to(torch.float64)
    charges = data['charges']
    cell = data['cell'].to(torch.float64)
    batch_idx = data['batch_idx']
    total_atoms = data['total_atoms']
    
    # Pre-generate k_vectors ONCE (not on every timing run!)
    k_vectors, k_squared = generate_k_vectors_pme(cell, mesh_dimensions)
    
    nl_kwargs = {
        'neighbor_list': nl_data,
        'neighbor_ptr': nl_ptr,
        'neighbor_shifts': nl_shifts,
        'k_vectors': k_vectors,
        'k_squared': k_squared,
    }
    
    # Warmup runs
    for _ in range(warmup_runs):
        _ = particle_mesh_ewald(
            positions=positions, charges=charges, cell=cell,
            alpha=alpha, mesh_dimensions=mesh_dimensions,
            spline_order=PME_SPLINE_ORDER, batch_idx=batch_idx,
            compute_forces=True, compute_charge_gradients=compute_charge_gradients,
            accuracy=accuracy, **nl_kwargs,
        )
    torch.cuda.synchronize()
    wp.synchronize()
    
    # Memory measurement
    clean_gpu()
    torch.cuda.reset_peak_memory_stats()
    _ = particle_mesh_ewald(
        positions=positions, charges=charges, cell=cell,
        alpha=alpha, mesh_dimensions=mesh_dimensions,
        spline_order=PME_SPLINE_ORDER, batch_idx=batch_idx,
        compute_forces=True, compute_charge_gradients=compute_charge_gradients,
        accuracy=accuracy, **nl_kwargs,
    )
    torch.cuda.synchronize()
    wp.synchronize()
    mem_peak = torch.cuda.max_memory_allocated()
    gpu_info = get_gpu_mem()
    
    # Timing with correct batch pattern
    def run_pme():
        _ = particle_mesh_ewald(
            positions=positions, charges=charges, cell=cell,
            alpha=alpha, mesh_dimensions=mesh_dimensions,
            spline_order=PME_SPLINE_ORDER, batch_idx=batch_idx,
            compute_forces=True, compute_charge_gradients=compute_charge_gradients,
            accuracy=accuracy, **nl_kwargs,
        )
    
    mean_time = cuda_timed_runs(run_pme, num_runs)
    time_us = (mean_time * 1e6) / total_atoms
    
    return {
        'time_us': time_us,
        'mem_peak_gb': mem_peak / 1024**3,
        'mem_percent': 100.0 * mem_peak / gpu_info['total'],
    }


def benchmark_ewald(data, nl_data, nl_shifts, nl_ptr, alpha, k_cutoff,
                    accuracy, compute_charge_gradients, num_runs, warmup_runs=3):
    """Benchmark Ewald summation using correct batch timing pattern."""
    positions = data['coord'].to(torch.float64)
    charges = data['charges']
    cell = data['cell'].to(torch.float64)
    batch_idx = data['batch_idx']
    total_atoms = data['total_atoms']
    num_atoms = positions.shape[0]
    mask_value = num_atoms
    
    # Pre-generate k_vectors ONCE (not on every timing run!)
    # Dimension fix for batched mode
    k_vectors = generate_k_vectors_ewald_summation(cell, k_cutoff)
    if k_vectors.ndim == 2:
        k_vectors = k_vectors.unsqueeze(0)
    
    nl_kwargs = {
        'neighbor_list': nl_data,
        'neighbor_ptr': nl_ptr,
        'neighbor_shifts': nl_shifts,
        'k_vectors': k_vectors,  # Pass pre-generated k_vectors
    }
    
    if compute_charge_gradients:
        def run_ewald():
            _ = ewald_with_charge_gradients(
                positions=positions, charges=charges, cell=cell,
                alpha=alpha, k_vectors=k_vectors,
                nl_data=nl_data, nl_shifts=nl_shifts, nl_ptr=nl_ptr,
                batch_idx=batch_idx, mask_value=mask_value, compute_forces=True,
            )
    else:
        def run_ewald():
            _ = ewald_summation(
                positions=positions, charges=charges, cell=cell,
                alpha=alpha, k_cutoff=k_cutoff, batch_idx=batch_idx,
                compute_forces=True, accuracy=accuracy, **nl_kwargs,
            )
    
    # Warmup runs
    for _ in range(warmup_runs):
        run_ewald()
    torch.cuda.synchronize()
    wp.synchronize()
    
    # Memory measurement
    clean_gpu()
    torch.cuda.reset_peak_memory_stats()
    run_ewald()
    torch.cuda.synchronize()
    wp.synchronize()
    mem_peak = torch.cuda.max_memory_allocated()
    gpu_info = get_gpu_mem()
    
    # Timing with correct batch pattern
    mean_time = cuda_timed_runs(run_ewald, num_runs)
    time_us = (mean_time * 1e6) / total_atoms
    
    return {
        'time_us': time_us,
        'mem_peak_gb': mem_peak / 1024**3,
        'mem_percent': 100.0 * mem_peak / gpu_info['total'],
    }


# ============== BENCHMARK RUNNERS ==============
def run_nl_benchmark(args, pdb_files, output_dir):
    """Run NL benchmark suite."""
    print(f'\n{"="*70}')
    print(f'NL BENCHMARK: CONSTANT {format_num(args.target_atoms)} TOTAL ATOMS')
    print(f'Cutoffs: {args.cutoffs} Å')
    print(f'Methods: {args.nl_methods}')
    print(f'Timing runs: {args.timing_runs}')
    print(f'{"="*70}')
    
    results = []
    device = torch.device('cuda')
    
    for pdb_path in pdb_files:
        coords, _, _, cell = parse_pdb(pdb_path)
        atoms_per_system = len(coords)
        cell_size = np.diag(cell)[0]
        
        batch_size = args.target_atoms // atoms_per_system
        if batch_size < 1:
            continue
        
        actual_total = batch_size * atoms_per_system
        print(f'\n{pdb_path.name}: {format_num(atoms_per_system)} atoms × {batch_size} batch = {format_num(actual_total)} total')
        
        for cutoff in args.cutoffs:
            if cell_size < 2 * cutoff:
                print(f'  {cutoff}Å: WARNING cell {cell_size:.1f}Å < 2×cutoff')
            
            data = create_batch(pdb_path, batch_size, device)
            
            for method in args.nl_methods:
                try:
                    r = benchmark_nl(data, cutoff, method, args.timing_runs)
                    results.append({
                        'target_atoms': args.target_atoms,
                        'atoms_per_system': atoms_per_system,
                        'batch_size': batch_size,
                        'total_atoms': actual_total,
                        'cutoff': cutoff,
                        'method': method,
                        'time_us': r['time_us'],
                        'neighbors': r['neighbors'],
                        'mem_delta_mb': r['mem_delta_mb'],
                        'mem_percent': r['mem_percent'],
                    })
                    print(f'  {cutoff}Å {method:5s}: {r["time_us"]:.3f} μs/atom')
                except Exception as e:
                    print(f'  {cutoff}Å {method:5s}: FAILED - {e}')
            
            clean_gpu()
    
    # Save results
    csv_path = output_dir / 'benchmark_nl_results.csv'
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(results)
        print(f'\nSaved: {csv_path} ({len(results)} rows)')
    
    return results


def run_d3_benchmark(args, pdb_files, output_dir, d3_params):
    """Run D3 benchmark suite."""
    print(f'\n{"="*70}')
    print(f'D3 BENCHMARK: CONSTANT {format_num(args.target_atoms)} TOTAL ATOMS')
    print(f'Cutoffs: {args.cutoffs} Å')
    print(f'Timing runs: {args.timing_runs}')
    print(f'{"="*70}')
    
    results = []
    device = torch.device('cuda')
    
    for pdb_path in pdb_files:
        coords, _, _, cell = parse_pdb(pdb_path)
        atoms_per_system = len(coords)
        cell_size = np.diag(cell)[0]
        
        batch_size = args.target_atoms // atoms_per_system
        if batch_size < 1:
            continue
        
        actual_total = batch_size * atoms_per_system
        print(f'\n{pdb_path.name}: {format_num(atoms_per_system)} atoms × {batch_size} batch = {format_num(actual_total)} total')
        
        for cutoff in args.cutoffs:
            if cell_size < 2 * cutoff:
                print(f'  {cutoff}Å: WARNING cell {cell_size:.1f}Å < 2×cutoff')
            
            data = create_batch(pdb_path, batch_size, device)
            
            try:
                r = benchmark_d3(data, cutoff, d3_params, args.timing_runs)
                results.append({
                    'target_atoms': args.target_atoms,
                    'atoms_per_system': atoms_per_system,
                    'batch_size': batch_size,
                    'total_atoms': actual_total,
                    'cutoff': cutoff,
                    'time_nl_us': r['time_nl_us'],
                    'time_d3_us': r['time_d3_us'],
                    'mem_delta_mb': r['mem_delta_mb'],
                    'mem_percent': r['mem_percent'],
                })
                print(f'  {cutoff}Å: NL={r["time_nl_us"]:.3f}, D3={r["time_d3_us"]:.3f} μs/atom')
            except Exception as e:
                print(f'  {cutoff}Å: FAILED - {e}')
            
            clean_gpu()
    
    # Save results
    csv_path = output_dir / 'benchmark_d3_results.csv'
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(results)
        print(f'\nSaved: {csv_path} ({len(results)} rows)')
    
    return results


def run_d3_scaling_benchmark(args, pdb_files, output_dir, d3_params):
    """Run D3 scaling benchmark (batch=1, varying system size)."""
    print(f'\n{"="*70}')
    print(f'D3 SCALING BENCHMARK: SYSTEM SIZE SCALING (batch=1)')
    print(f'Cutoffs: {args.cutoffs} Å')
    print(f'Timing runs: {args.timing_runs}')
    print(f'{"="*70}')
    
    results = []
    device = torch.device('cuda')
    batch_size = 1
    
    for pdb_path in pdb_files:
        coords, _, _, cell = parse_pdb(pdb_path)
        atoms_per_system = len(coords)
        cell_size = np.diag(cell)[0]
        
        print(f'\n{pdb_path.name}: {format_num(atoms_per_system)} atoms × {batch_size} batch')
        
        for cutoff in args.cutoffs:
            if cell_size < 2 * cutoff:
                print(f'  {cutoff}Å: WARNING cell {cell_size:.1f}Å < 2×cutoff')
            
            data = create_batch(pdb_path, batch_size, device)
            
            try:
                r = benchmark_d3(data, cutoff, d3_params, args.timing_runs)
                results.append({
                    'atoms_per_system': atoms_per_system,
                    'batch_size': batch_size,
                    'total_atoms': atoms_per_system,
                    'cutoff': cutoff,
                    'time_nl_us': r['time_nl_us'],
                    'time_d3_us': r['time_d3_us'],
                    'mem_delta_mb': r['mem_delta_mb'],
                    'mem_percent': r['mem_percent'],
                })
                print(f'  {cutoff}Å: NL={r["time_nl_us"]:.3f}, D3={r["time_d3_us"]:.3f} μs/atom')
            except Exception as e:
                print(f'  {cutoff}Å: FAILED - {e}')
            
            clean_gpu()
    
    # Save results
    csv_path = output_dir / 'benchmark_d3_scaling_results.csv'
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(results)
        print(f'\nSaved: {csv_path} ({len(results)} rows)')
    
    return results


def run_electrostatics_scaling_benchmark(args, pdb_files, output_dir):
    """Run electrostatics scaling benchmark (batch=1, varying system size)."""
    print(f'\n{"="*70}')
    print(f'ELECTROSTATICS SCALING BENCHMARK: SYSTEM SIZE SCALING (batch=1)')
    print(f'Accuracy levels: {args.accuracy_levels}')
    print(f'Timing runs: {args.timing_runs}')
    print(f'Max atoms: {format_num(args.max_atoms)}')
    print(f'{"="*70}')
    
    results = []
    device = torch.device('cuda')
    batch_size = 1
    
    for accuracy in args.accuracy_levels:
        print(f'\n{"="*40} ACCURACY: {accuracy:.0e} {"="*40}')
        
        for pdb_path in pdb_files:
            coords, _, elements, cell = parse_pdb(pdb_path)
            n_atoms = len(coords)
            
            # Check limits
            if n_atoms > args.max_atoms:
                print(f'{pdb_path.name}: {format_num(n_atoms)} atoms - SKIPPED (> max)')
                continue
            
            # OOM prevention for high accuracy
            if accuracy in SKIP_ACCURACY_FOR_LARGE_SYSTEMS:
                threshold = SKIP_ACCURACY_FOR_LARGE_SYSTEMS[accuracy]
                if n_atoms >= threshold:
                    print(f'{pdb_path.name}: {format_num(n_atoms)} atoms - SKIPPED (OOM risk at {accuracy:.0e})')
                    continue
            
            print(f'\n{pdb_path.name}: {format_num(n_atoms)} atoms, accuracy={accuracy:.0e}')
            
            try:
                data = create_batch_electrostatics(pdb_path, batch_size, device)
                positions_f64 = data['coord'].to(torch.float64)
                cell_f64 = data['cell'].to(torch.float64)
                batch_idx = data['batch_idx']
                
                # Get optimal parameters
                pme_params = estimate_pme_parameters(
                    positions=positions_f64, cell=cell_f64,
                    batch_idx=batch_idx, accuracy=accuracy
                )
                ewald_params = estimate_ewald_parameters(
                    positions=positions_f64, cell=cell_f64,
                    batch_idx=batch_idx, accuracy=accuracy
                )
                
                alpha = pme_params.alpha.clone()
                real_space_cutoff = pme_params.real_space_cutoff.item()
                mesh_dims = tuple(pme_params.mesh_dimensions)
                k_cutoff = ewald_params.reciprocal_space_cutoff.item()
                
                del pme_params, ewald_params
                clean_gpu()
                
                print(f'  Params: alpha={alpha.item():.4f}, r_cut={real_space_cutoff:.2f}Å, k_cut={k_cutoff:.2f}')
                
                # Generate neighbor list
                nl_data, nl_ptr, nl_shifts = generate_neighbor_list(data, real_space_cutoff)
                n_neighbors = nl_data.shape[1]
                print(f'  NL: {n_neighbors:,} pairs')
                
                # Benchmark PME (no charge gradients)
                pme_result = benchmark_pme(
                    data, nl_data, nl_shifts, nl_ptr, alpha, mesh_dims,
                    accuracy, compute_charge_gradients=False, num_runs=args.timing_runs
                )
                print(f'  PME:      {pme_result["time_us"]:.3f} μs/atom, mem={pme_result["mem_peak_gb"]:.1f} GB')
                clean_gpu()
                
                # Benchmark Ewald (no charge gradients)
                ewald_result = benchmark_ewald(
                    data, nl_data, nl_shifts, nl_ptr, alpha, k_cutoff,
                    accuracy, compute_charge_gradients=False, num_runs=args.timing_runs
                )
                print(f'  Ewald:    {ewald_result["time_us"]:.3f} μs/atom, mem={ewald_result["mem_peak_gb"]:.1f} GB')
                clean_gpu()
                
                # Benchmark PME with charge gradients
                pme_cg_result = benchmark_pme(
                    data, nl_data, nl_shifts, nl_ptr, alpha, mesh_dims,
                    accuracy, compute_charge_gradients=True, num_runs=args.timing_runs
                )
                print(f'  PME+cg:   {pme_cg_result["time_us"]:.3f} μs/atom, mem={pme_cg_result["mem_peak_gb"]:.1f} GB')
                clean_gpu()
                
                # Benchmark Ewald with charge gradients
                ewald_cg_result = benchmark_ewald(
                    data, nl_data, nl_shifts, nl_ptr, alpha, k_cutoff,
                    accuracy, compute_charge_gradients=True, num_runs=args.timing_runs
                )
                print(f'  Ewald+cg: {ewald_cg_result["time_us"]:.3f} μs/atom, mem={ewald_cg_result["mem_peak_gb"]:.1f} GB')
                
                ratio = ewald_result['time_us'] / pme_result['time_us']
                print(f'  Ratio: {ratio:.2f}x ({"Ewald" if ratio < 1 else "PME"} faster)')
                
                results.append({
                    'accuracy': accuracy,
                    'n_atoms': n_atoms,
                    'batch_size': batch_size,
                    'alpha': alpha.item(),
                    'real_space_cutoff': real_space_cutoff,
                    'k_cutoff': k_cutoff,
                    'mesh_dims': str(mesh_dims),
                    'n_neighbors': n_neighbors,
                    'pme_time_us': pme_result['time_us'],
                    'pme_mem_gb': pme_result['mem_peak_gb'],
                    'ewald_time_us': ewald_result['time_us'],
                    'ewald_mem_gb': ewald_result['mem_peak_gb'],
                    'pme_cg_time_us': pme_cg_result['time_us'],
                    'pme_cg_mem_gb': pme_cg_result['mem_peak_gb'],
                    'ewald_cg_time_us': ewald_cg_result['time_us'],
                    'ewald_cg_mem_gb': ewald_cg_result['mem_peak_gb'],
                    'ratio': ratio,
                })
                
            except Exception as e:
                print(f'  ERROR: {e}')
                import traceback
                traceback.print_exc()
            finally:
                clean_gpu()
    
    # Save results
    csv_path = output_dir / 'benchmark_electrostatics_results.csv'
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(results)
        print(f'\nSaved: {csv_path} ({len(results)} rows)')
    
    return results


def run_electrostatics_batched_benchmark(args, pdb_files, output_dir):
    """Run electrostatics batched benchmark (constant 128k total atoms)."""
    print(f'\n{"="*70}')
    print(f'ELECTROSTATICS BATCHED BENCHMARK: CONSTANT {format_num(args.target_atoms)} TOTAL ATOMS')
    print(f'Accuracy levels: {args.accuracy_levels}')
    print(f'Timing runs: {args.timing_runs}')
    print(f'{"="*70}')
    
    results = []
    device = torch.device('cuda')
    
    for accuracy in args.accuracy_levels:
        print(f'\n{"="*40} ACCURACY: {accuracy:.0e} {"="*40}')
        
        for pdb_path in pdb_files:
            coords, _, elements, cell = parse_pdb(pdb_path)
            atoms_per_system = len(coords)
            
            batch_size = args.target_atoms // atoms_per_system
            if batch_size < 1:
                continue
            
            # OOM prevention for high accuracy
            if accuracy in SKIP_ACCURACY_FOR_LARGE_SYSTEMS:
                threshold = SKIP_ACCURACY_FOR_LARGE_SYSTEMS[accuracy]
                if atoms_per_system >= threshold:
                    print(f'SKIP: {pdb_path.name} ({atoms_per_system} atoms) at {accuracy:.0e} (OOM risk)')
                    continue
            
            actual_total = batch_size * atoms_per_system
            print(f'\n{pdb_path.name}: {format_num(atoms_per_system)} × {batch_size} = {format_num(actual_total)} total')
            
            try:
                data = create_batch_electrostatics(pdb_path, batch_size, device)
                positions_f64 = data['coord'].to(torch.float64)
                cell_f64 = data['cell'].to(torch.float64)
                batch_idx = data['batch_idx']
                
                # Get optimal parameters (batched)
                pme_params = estimate_pme_parameters(
                    positions=positions_f64, cell=cell_f64,
                    batch_idx=batch_idx, accuracy=accuracy
                )
                ewald_params = estimate_ewald_parameters(
                    positions=positions_f64, cell=cell_f64,
                    batch_idx=batch_idx, accuracy=accuracy
                )
                
                # For batched mode: use scalar alpha
                alpha = pme_params.alpha.mean().clone()
                real_space_cutoff = pme_params.real_space_cutoff[0].item()
                mesh_dims = tuple(pme_params.mesh_dimensions)
                k_cutoff = ewald_params.reciprocal_space_cutoff.max().item()
                
                del pme_params, ewald_params
                clean_gpu()
                
                print(f'  Params: alpha={alpha.item():.4f}, r_cut={real_space_cutoff:.2f}Å')
                
                # Generate neighbor list
                nl_data, nl_ptr, nl_shifts = generate_neighbor_list(data, real_space_cutoff)
                n_neighbors = nl_data.shape[1]
                print(f'  NL: {n_neighbors:,} pairs')
                
                # Benchmark all 4 methods
                pme_result = benchmark_pme(
                    data, nl_data, nl_shifts, nl_ptr, alpha, mesh_dims,
                    accuracy, compute_charge_gradients=False, num_runs=args.timing_runs
                )
                print(f'  PME:      {pme_result["time_us"]:.3f} μs/atom')
                clean_gpu()
                
                ewald_result = benchmark_ewald(
                    data, nl_data, nl_shifts, nl_ptr, alpha, k_cutoff,
                    accuracy, compute_charge_gradients=False, num_runs=args.timing_runs
                )
                print(f'  Ewald:    {ewald_result["time_us"]:.3f} μs/atom')
                clean_gpu()
                
                pme_cg_result = benchmark_pme(
                    data, nl_data, nl_shifts, nl_ptr, alpha, mesh_dims,
                    accuracy, compute_charge_gradients=True, num_runs=args.timing_runs
                )
                print(f'  PME+cg:   {pme_cg_result["time_us"]:.3f} μs/atom')
                clean_gpu()
                
                ewald_cg_result = benchmark_ewald(
                    data, nl_data, nl_shifts, nl_ptr, alpha, k_cutoff,
                    accuracy, compute_charge_gradients=True, num_runs=args.timing_runs
                )
                print(f'  Ewald+cg: {ewald_cg_result["time_us"]:.3f} μs/atom')
                
                ratio = ewald_result['time_us'] / pme_result['time_us']
                print(f'  Ratio: {ratio:.2f}x')
                
                results.append({
                    'accuracy': accuracy,
                    'target_atoms': args.target_atoms,
                    'atoms_per_system': atoms_per_system,
                    'batch_size': batch_size,
                    'total_atoms': actual_total,
                    'alpha': alpha.item(),
                    'real_space_cutoff': real_space_cutoff,
                    'k_cutoff': k_cutoff,
                    'mesh_dims': str(mesh_dims),
                    'n_neighbors': n_neighbors,
                    'pme_time_us': pme_result['time_us'],
                    'pme_mem_gb': pme_result['mem_peak_gb'],
                    'ewald_time_us': ewald_result['time_us'],
                    'ewald_mem_gb': ewald_result['mem_peak_gb'],
                    'pme_cg_time_us': pme_cg_result['time_us'],
                    'pme_cg_mem_gb': pme_cg_result['mem_peak_gb'],
                    'ewald_cg_time_us': ewald_cg_result['time_us'],
                    'ewald_cg_mem_gb': ewald_cg_result['mem_peak_gb'],
                    'ratio': ratio,
                })
                
            except Exception as e:
                print(f'  ERROR: {e}')
                import traceback
                traceback.print_exc()
            finally:
                clean_gpu()
    
    # Save results
    csv_path = output_dir / 'benchmark_electrostatics_128k_results.csv'
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(results)
        print(f'\nSaved: {csv_path} ({len(results)} rows)')
    
    return results


# ============== MAIN ==============
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Unified Benchmark Suite for NL, D3, and Electrostatics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python benchmark_suite.py -b nl --cutoffs 6 15 25
    python benchmark_suite.py -b d3 d3s --cutoffs 15 25
    python benchmark_suite.py -b el elb --accuracy 1e-4 1e-6
    python benchmark_suite.py -b all --timing-runs 20

Benchmark aliases:
    nl          Neighbor list (batched 128k)
    d3          D3 dispersion (batched 128k)
    d3s         D3 scaling (batch=1, varying size)
    el          Electrostatics scaling (batch=1, varying size)
    elb         Electrostatics batched (128k total)
    all         Run all benchmarks
        """
    )
    
    # Benchmark selection - allow multiple
    parser.add_argument(
        '--benchmark', '-b',
        nargs='+',
        choices=['nl', 'd3', 'd3s', 'el', 'elb', 'all'],
        default=['all'],
        help='Benchmarks to run (default: all). Can specify multiple: -b el elb'
    )
    
    # Timing configuration
    parser.add_argument(
        '--timing-runs', '-n',
        type=int,
        default=20,
        help='Number of timing runs for each measurement (default: 20)'
    )
    
    # NL/D3 parameters
    parser.add_argument(
        '--cutoffs', '-c',
        type=float,
        nargs='+',
        default=[6.0, 15.0, 25.0],
        help='Cutoff radii in Angstrom (default: 6 15 25)'
    )
    
    parser.add_argument(
        '--nl-methods',
        nargs='+',
        choices=['naive', 'cell'],
        default=['naive', 'cell'],
        help='NL methods to benchmark (default: naive cell)'
    )
    
    # Electrostatics parameters
    parser.add_argument(
        '--accuracy', '-a',
        type=float,
        nargs='+',
        default=[1e-4, 1e-6],
        dest='accuracy_levels',
        help='Accuracy levels for electrostatics (default: 1e-4 1e-6)'
    )
    
    parser.add_argument(
        '--max-atoms',
        type=int,
        default=131072,
        help='Maximum atoms for scaling benchmarks (default: 131072 = 128k)'
    )
    
    # System configuration
    parser.add_argument(
        '--target-atoms', '-t',
        type=int,
        default=131072,
        help='Target total atoms for batched benchmarks (default: 131072 = 128k)'
    )
    
    # Paths
    parser.add_argument(
        '--nh3-dir',
        type=Path,
        default=DEFAULT_NH3_DIR,
        help=f'Directory with NH3 PDB files (default: {DEFAULT_NH3_DIR})'
    )
    
    parser.add_argument(
        '--output-base',
        type=Path,
        default=DEFAULT_OUTPUT_BASE,
        help=f'Base directory for output (default: {DEFAULT_OUTPUT_BASE})'
    )
    
    parser.add_argument(
        '--d3-params',
        type=Path,
        default=DEFAULT_D3_PARAMS,
        help=f'Path to D3 parameters file (default: {DEFAULT_D3_PARAMS})'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Print configuration
    print('='*70)
    print('UNIFIED BENCHMARK SUITE')
    print('='*70)
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA: {torch.cuda.get_device_name(0)}')
    print(f'Benchmarks: {", ".join(args.benchmark)}')
    print(f'Timing runs: {args.timing_runs}')
    print(f'Cutoffs (NL/D3): {args.cutoffs} Å')
    print(f'Accuracy (Electrostatics): {args.accuracy_levels}')
    print(f'Target atoms: {format_num(args.target_atoms)}')
    print(f'Max atoms (scaling): {format_num(args.max_atoms)}')
    print(f'NH3 directory: {args.nh3_dir}')
    print()
    print('Timing pattern: BATCH (verified by senior engineer)')
    print('  start.record() → N × fn() → end.record() → sync()')
    print('  Sync is OUTSIDE loop - no overhead pollution')
    print('='*70)
    
    # Create timestamped output directory
    timestamp = get_timestamp()
    output_dir = args.output_base / f'benchmark_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'\nOutput directory: {output_dir}')
    
    # Load PDB files
    pdb_files = natsorted(args.nh3_dir.glob('ammonia_pbc_*.pdb'), key=lambda p: p.name)
    if not pdb_files:
        print(f'ERROR: No PDB files found in {args.nh3_dir}')
        return 1
    print(f'Found {len(pdb_files)} PDB files')
    
    # Load D3 parameters if needed
    d3_params = None
    d3_benchmarks = {'d3', 'd3s'}
    if d3_benchmarks & set(args.benchmark) or 'all' in args.benchmark:
        if args.d3_params.exists():
            d3_params = torch.load(args.d3_params, map_location='cuda', weights_only=True)
            print(f'Loaded D3 parameters from {args.d3_params}')
        else:
            print(f'WARNING: D3 parameters not found at {args.d3_params}')
            if d3_benchmarks & set(args.benchmark):
                print('ERROR: D3 parameters required for D3 benchmarks')
                return 1
    
    # Expand 'all' to all benchmarks
    benchmarks = set(args.benchmark)
    if 'all' in benchmarks:
        benchmarks = {'nl', 'd3', 'd3s', 'el', 'elb'}
    
    # Run benchmarks
    if 'nl' in benchmarks:
        run_nl_benchmark(args, pdb_files, output_dir)
    
    if 'd3' in benchmarks and d3_params is not None:
        run_d3_benchmark(args, pdb_files, output_dir, d3_params)
    
    if 'd3s' in benchmarks and d3_params is not None:
        run_d3_scaling_benchmark(args, pdb_files, output_dir, d3_params)
    
    if 'el' in benchmarks:
        run_electrostatics_scaling_benchmark(args, pdb_files, output_dir)
    
    if 'elb' in benchmarks:
        run_electrostatics_batched_benchmark(args, pdb_files, output_dir)
    
    print(f'\n{"="*70}')
    print('BENCHMARK COMPLETE')
    print(f'Results saved to: {output_dir}')
    print('='*70)
    
    return 0


if __name__ == '__main__':
    exit(main())
