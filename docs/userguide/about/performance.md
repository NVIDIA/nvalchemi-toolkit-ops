(performance)=

# Performance Guide

Optimal settings depend on GPU, system size, cutoff, density and dtype. Measure
on your own configuration.

## Cost model

At small and medium sizes, per-call cost is usually dominated by host/device
synchronisation and allocation rather than kernel time.

Synchronisation comes from shapes or control flow that depend on device data.
Each component exposes arguments that let you supply the value instead, and
these matter more than preallocating output buffers:

| Argument omitted | Consequence |
| --- | --- |
| `batch_ptr` (given only `batch_idx`) | segment pointers derived via `torch.bincount`; output length depends on data |
| `num_systems` (DFT-D3) | inferred from `cell` or `batch_idx` |
| `method` (neighbour lists) | strategy selected per call on the host |
| `max_neighbors`, `max_total_cells` | sizing helpers run per call and read device values |
| rebuild decision read as a Python `bool` | forces a sync per step |

Allocation cost comes from output and scratch buffers created per call. Every
entry point accepts them.

### Verifying

`torch._dynamo.explain` reports graph breaks without changing execution. Each
entry carries the reason and the user frame that caused it:

```python
import torch._dynamo

report = torch._dynamo.explain(step_fn)(*args)
print(report.graph_break_count)
for b in report.break_reasons:
    print(b.reason.splitlines()[0], "|", b.user_stack[-1])
```

A break on `Tensor.item()`, `torch.bincount` or similar points at a shape or
control-flow decision still being made from device data.

`TORCH_LOGS=graph_breaks` gives the same information from an ordinary run,
which is useful when the call is buried in a larger model:

```bash
TORCH_LOGS=graph_breaks python train.py
```

`fullgraph=True` also surfaces breaks, by raising on the first one. That is a
behaviour change rather than a diagnostic, so prefer it as an assertion in
tests over a debugging tool.

Allocation is separate from tracing; check allocator deltas across steps:

```python
step_fn(); torch.cuda.synchronize()
before = torch.cuda.memory_allocated()
for _ in range(10):
    step_fn()
torch.cuda.synchronize()
print(torch.cuda.memory_allocated() - before)   # expect ~0
```

## Neighbour lists

Strategies differ in scaling. Some are quadratic in atom count with low fixed
overhead; others build a spatial data structure with fixed setup cost that
amortises as the system grows. The crossover moves with GPU, cutoff and
density.

### Selecting a strategy

```python
from nvalchemiops.torch.neighbors import suggest_neighbor_list_method

method = suggest_neighbor_list_method(...)   # setup only
```

{func}`~nvalchemiops.torch.neighbors.suggest_neighbor_list_method` inspects the
problem on the host. Call it once at setup and pass the result as `method=`;
calling it per step, or inside `torch.compile`, reintroduces a sync.

{func}`~nvalchemiops.torch.neighbors.estimate_neighbor_list_costs` returns the
full ranked list with costs if you want to see the runner-up.

### Example: Preallocating a cell list

```{note}
Each strategy takes a different set of buffers. To know what to allocate, refer
to the API reference for the entry point you are using --- every argument typed
`torch.Tensor | None` can be preallocated and passed in, with its shape and
dtype given there:
{func}`~nvalchemiops.torch.neighbors.cell_list`,
{func}`~nvalchemiops.torch.neighbors.naive_neighbor_list`,
{func}`~nvalchemiops.torch.neighbors.cluster_tile_neighbor_list`,
{func}`~nvalchemiops.torch.neighbors.batch_cell_list`,
{func}`~nvalchemiops.torch.neighbors.batch_naive_neighbor_list`,
{func}`~nvalchemiops.torch.neighbors.batch_cluster_tile_neighbor_list`.
```

`cell_list` accepts every buffer it would otherwise allocate. Sizing needs two
host-side helpers:
{func}`~nvalchemiops.torch.neighbors.estimate_cell_list_sizes` for the grid and
{func}`~nvalchemiops.neighbors.neighbor_utils.estimate_max_neighbors` for the
matrix width;
{func}`~nvalchemiops.torch.neighbors.neighbor_utils.allocate_cell_list` returns
the build-side tensors as a group.

```python
import torch
from nvalchemiops.torch.neighbors import cell_list, estimate_cell_list_sizes
from nvalchemiops.torch.neighbors.cell_list import allocate_query_sort_scratch
from nvalchemiops.torch.neighbors.neighbor_utils import (
    allocate_cell_list, estimate_max_neighbors,
)

# --- setup ---
max_total_cells, search_radius = estimate_cell_list_sizes(cell, pbc, cutoff)
max_neighbors = estimate_max_neighbors(cutoff, atomic_density=density)

# build-side state: grid dimensions, per-atom cell assignment, cell contents
(cells_per_dimension, neighbor_search_radius, atom_periodic_shifts,
 atom_to_cell_mapping, atoms_per_cell_count, cell_atom_start_indices,
 cell_atom_list) = allocate_cell_list(
    n_atoms, max_total_cells, search_radius, device
)

# query outputs
neighbor_matrix = torch.empty((n_atoms, max_neighbors), dtype=torch.int32,
                              device=device)
neighbor_matrix_shifts = torch.empty((n_atoms, max_neighbors, 3),
                                     dtype=torch.int8, device=device)
num_neighbors = torch.empty(n_atoms, dtype=torch.int32, device=device)

# sort scratch: needed by the sorted atom-centric and pair-centric paths,
# unused by direct atom-centric
sorted_positions, sorted_shifts = allocate_query_sort_scratch(
    n_atoms, dtype=positions.dtype, device=device
)

# --- per step ---
cell_list(
    positions, cutoff, cell, pbc,
    max_neighbors=max_neighbors,
    neighbor_matrix=neighbor_matrix,
    neighbor_matrix_shifts=neighbor_matrix_shifts,
    num_neighbors=num_neighbors,
    cells_per_dimension=cells_per_dimension,
    neighbor_search_radius=neighbor_search_radius,
    atom_periodic_shifts=atom_periodic_shifts,
    atom_to_cell_mapping=atom_to_cell_mapping,
    atoms_per_cell_count=atoms_per_cell_count,
    cell_atom_start_indices=cell_atom_start_indices,
    cell_atom_list=cell_atom_list,
    sorted_positions=sorted_positions,
    sorted_shifts=sorted_shifts,
    strategy="pair_centric",
)
```

Notes:

- `max_total_cells` is capped by `max_nbins` (default 524288). When the natural
  `(box / cutoff)³` grid exceeds the cap, cells per dimension are halved until
  it fits, which raises atoms-per-cell and increases inner-loop work
  quadratically. Check the returned value against `(box / cutoff)³` for large
  boxes.
- `strategy` selects `"pair_centric"` or `"atom_centric"`; `atom_centric_path`
  selects `"direct"` or `"sorted"` within the latter. `"auto"` decides on the
  host.
- Omit `sorted_positions` / `sorted_shifts` for direct atom-centric; they are
  not read.
- {func}`~nvalchemiops.torch.neighbors.cell_list.build_cell_list` and
  {func}`~nvalchemiops.torch.neighbors.cell_list.query_cell_list` are separable
  if you rebuild the grid less often than you query it.

Batched systems use {func}`~nvalchemiops.torch.neighbors.batch_cell_list` with
`estimate_batch_cell_list_sizes`, and additionally take `batch_idx` and
`batch_ptr`.

### Measuring the crossover

1. Preallocate each strategy's buffers.
2. Time each with warmup, taking medians.
3. Rotate positions between iterations; some implementations cache on input
   identity.
4. Pin the winner with `method=`.

Time build-included and build-amortised separately. With a Verlet skin the
amortised figure reflects steady-state MD; the included figure gives rebuild
cost.

For many small systems, measure the batched path at constant total atom count
while varying the split.

### Rebuild detection

`nvalchemiops.torch.neighbors.rebuild_detection` provides
`neighbor_list_needs_rebuild` and `cell_list_needs_rebuild` (plus `batch_`
variants), which test displacement against a skin on device rather than
rebuilding every step.

The result is a device tensor. Reading it to a Python bool syncs; pass it
through as `rebuild_flags` instead to keep the decision on device under
`torch.compile` or CUDA graphs.

## Ewald and PME

### Preallocation

```{note}
As with neighbour lists, the API reference for the entry point lists what can be
supplied: {func}`~nvalchemiops.torch.interactions.electrostatics.ewald_summation` and
{func}`~nvalchemiops.torch.interactions.electrostatics.particle_mesh_ewald`. The
real/reciprocal halves are separately
callable as {func}`~nvalchemiops.torch.interactions.electrostatics.ewald_real_space`,
{func}`~nvalchemiops.torch.interactions.electrostatics.ewald_reciprocal_space` and
{func}`~nvalchemiops.torch.interactions.electrostatics.pme_reciprocal_space`, which
is what to profile against when
attributing cost.
```

`alpha`, the k-vector set and the PME mesh depend on cell and accuracy, not
positions. Recomputing them per step re-runs host-side estimation
({func}`~nvalchemiops.torch.interactions.electrostatics.estimate_ewald_parameters`,
{func}`~nvalchemiops.torch.interactions.electrostatics.estimate_pme_mesh_dimensions`,
{func}`~nvalchemiops.torch.interactions.electrostatics.generate_k_vectors_ewald_summation`):

```python
from nvalchemiops.torch.interactions.electrostatics import (
    estimate_ewald_parameters, estimate_pme_mesh_dimensions,
    generate_k_vectors_ewald_summation,
)

params = estimate_ewald_parameters(positions, cell, accuracy=accuracy)
k_vectors = generate_k_vectors_ewald_summation(
    cell, params.reciprocal_space_cutoff
)
mesh = estimate_pme_mesh_dimensions(cell, params.alpha, accuracy=accuracy)
```

Under NPT, refresh these on cell updates rather than per force evaluation.
{func}`~nvalchemiops.torch.interactions.electrostatics.estimate_pme_parameters`
returns alpha and the mesh together as a
{class}`~nvalchemiops.torch.interactions.electrostatics.PMEParameters`;
{func}`~nvalchemiops.torch.interactions.electrostatics.estimate_ewald_parameters`
returns
an {class}`~nvalchemiops.torch.interactions.electrostatics.EwaldParameters`.

PME additionally precomputes B-spline moduli via
{func}`~nvalchemiops.torch.interactions.electrostatics.compute_bspline_moduli_1d`,
which depend only on the mesh and spline order.

Both methods consume a neighbour list for real space, so the neighbour-list
section applies to that list as well. Pass the matrix and shifts you already
built rather than letting the entry point construct one.

For batched systems pass `batch_ptr` alongside `batch_idx`, and
`max_atoms_per_system` where the entry point accepts it.

### Work split

`alpha` sets the balance between real and reciprocal space. Larger `alpha`
narrows real space and widens reciprocal space.

Scaling:

| Component | Scales with |
| --- | --- |
| Real space | pair count: cutoff³ × density |
| Ewald reciprocal | atom count × k-vector count; k-vector count grows with cell volume |
| PME reciprocal | FFT mesh, largely independent of atom count |

Profile the two halves separately before tuning. The dominant half determines
whether to change `alpha`, the cutoff, or the method.

When comparing Ewald against PME, hold the real-space cutoff and accuracy
fixed. Both pin `alpha` and therefore the work split.

### Precision

Reciprocal-space sums cancel heavily across atoms, so reductions benefit from
float64 even where per-element arithmetic does not. Some kernels use float32
per-element with float64 reductions.

Validate any float32 path against a float64 reference on your own system, and
check forces as well as energies.

## DFT-D3

### Preallocation

Reference tables are element-specific and independent of geometry. Build
`D3Parameters` once and move it to the device at setup:

```{note}
{func}`~nvalchemiops.torch.interactions.dispersion.dftd3` lists every
optional buffer and table it accepts;
{class}`~nvalchemiops.torch.interactions.dispersion.D3Parameters` documents
the reference tables and their shapes.
```

```python
from nvalchemiops.torch.interactions.dispersion import dftd3, D3Parameters

d3_params = D3Parameters(...).to(device)      # setup
```

Per call, supply:

- `d3_params`, rather than letting the tables be rebuilt or re-transferred.
- `neighbor_matrix` and `neighbor_matrix_shifts` (or the COO form with
  `neighbor_list`, `neighbor_ptr`, `unit_shifts`) from a list you already built.
- `num_systems` for batched calls. Omitting it infers the count from `cell` or
  `batch_idx`, which synchronises.
- `device`, if you want to skip inference from the positions tensor.

Set `compute_virial` only when the virial is needed.

### Sizing

{func}`~nvalchemiops.torch.interactions.dispersion.dftd3` takes no cutoff
 argument: the cutoff is whatever the supplied
neighbour list was built with.

D3 cutoffs are typically several times larger than an electrostatics real-space
cutoff. Neighbour count per atom scales with the cube of the cutoff, so the
neighbour matrix dominates memory here and `max_neighbors` must be sized
against the D3 cutoff rather than reused from an electrostatics list.

## JAX

The preceding guidance is written against the Torch API. The tuning targets are
the same -- avoid per-call allocation, hoist host-side sizing, pin the strategy
-- but the mechanisms differ.

Not benchmarked; the following follows from the API surface.

### What carries over

`nvalchemiops.jax.neighbors` mirrors the Torch surface:
`suggest_neighbor_list_method`, `estimate_neighbor_list_costs`,
`estimate_cell_list_sizes`, `estimate_max_neighbors`, `allocate_cell_list`,
`build_cell_list` / `query_cell_list`, and the `rebuild_detection` helpers.
Strategy selection and sizing are host-side in both backends, so both belong in
setup.

### Differences

**Shapes are static.** JAX traces on shape, so `max_neighbors` and
`max_total_cells` are effectively compile-time constants. Changing either
retraces. Pick values that cover the whole run rather than the current
configuration, and avoid recomputing them from device data inside a jitted
function.

**Buffers are donated, not mutated.** Torch preallocation works by writing into
tensors you own. JAX is functional: pass the same buffers in and use
`jax.jit(..., donate_argnums=...)` so XLA reuses the storage rather than
allocating new outputs each call.

**`graph_mode="warp"`.** `cell_list` takes a `graph_mode` argument absent from
the Torch signature. `"warp"` targets jitted call sites that donate and reuse
the cell-list caches and output buffers, fusing build and query. Combined with
an explicit `max_total_cells` it requires an explicit `neighbor_search_radius`,
since the fused path cannot inspect the constructed grid between the two
stages.

**`max_total_cells` is an argument.** The Torch entry point derives it
internally from `estimate_cell_list_sizes`; the JAX one accepts it directly and
estimates only when it is `None`. Passing it explicitly keeps the estimate out
of the traced region.

**Rebuild flags.** The same device-tensor rule applies, and matters more:
reading a rebuild decision to a Python bool forces a host sync and breaks the
trace. Keep it on device.

### Measuring

Time after the first call. The first invocation of a jitted function includes
tracing and XLA compilation, which will otherwise dominate. `block_until_ready()`
is the synchronisation point.

```python
fn = jax.jit(step_fn, donate_argnums=(...,))
fn(*args)[0].block_until_ready()          # warm up: trace + compile

times = []
for _ in range(n):
    t0 = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    times.append(time.perf_counter() - t0)
```

Watch for retracing during a sweep: a changed `max_neighbors` or atom count
recompiles, and that cost can land inside a timed region.

## Environment overrides

Defaults are chosen to be reasonable; these exist for measurement and for cases
where a heuristic is wrong for a given workload.

| Variable | Effect |
| --- | --- |
| `NVALCHEMIOPS_EWALD_RECIP_TILED` | Force the Ewald reciprocal fill to use (`1`) or avoid (`0`) a tiled launch. |
| `NVALCHEMIOPS_EWALD_RECIP_MIN_ATOMS` | Atom-count threshold for automatic tiled fill selection. |
| `NVALCHEMIOPS_EWALD_RECIP_TILE_DIM` | Block width for tiled reciprocal kernels. |
| `NVALCHEMIOPS_ELECTROSTATICS_FP32` | Evaluate monopole electrostatics in float32 where the inputs are float32: the per-pair real-space cores and the no-store reciprocal path. Real space is shared by Ewald and PME, so both are affected. Changes float32 results at the ~1e-07 level. |
| `NVALCHEMIOPS_DYNAMICS_TILE_DIM` | Block width for tiled dynamics kernels. |
| `ALCH_EWALD_BATCH_BLOCK_SIZE` | Block size for batched Ewald kernels. |

## Benchmarking

- Warm up before timing; first-call cost includes kernel compilation.
- Take medians, not means.
- Synchronise around the timed region.
- Vary inputs between iterations.
- Record peak memory alongside time.
- Benchmark energy and forces if production uses forces; backward cost is not
  proportional to forward cost.
- Measure `torch.compile` per component. It helps some paths and hurts others,
  particularly where a hand-written analytic backward is re-derived.
