<!-- markdownlint-disable MD013 MD049 -->

(dynamics_userguide)=

# Molecular Dynamics Integrators

Molecular dynamics (MD) simulations propagate atomic positions and velocities forward
in time using numerical integrators. ALCHEMI Toolkit-Ops provides GPU-accelerated
implementations of standard MD integrators and thermostats via [NVIDIA Warp](https://nvidia.github.io/warp/),
with full PyTorch autograd support for machine learning applications.

```{tip}
For most applications, start with {func}`~nvalchemiops.dynamics.integrators.velocity_verlet.velocity_verlet_position_update`
for NVE simulations, or {func}`~nvalchemiops.dynamics.integrators.langevin.langevin_baoab_half_step` for NVT simulations.
These integrators are time-reversible, symplectic, and provide excellent energy conservation or temperature control.
```

## Overview of Available Integrators

ALCHEMI Toolkit-Ops provides integrators for different statistical ensembles:

| Integrator | Ensemble | Conservation | Best For |
|------------|----------|--------------|----------|
| **Velocity Verlet** | NVE | Energy | Testing, production NVE runs |
| **Langevin (BAOAB)** | NVT | Temperature | Canonical sampling, equilibration |
| **Nosé-Hoover Chain** | NVT | Temperature | Deterministic thermostat, long runs |
| **NPT (MTK)** | NPT | Temperature, Pressure | Constant pressure simulations |
| **NPH (MTK)** | NPH | Enthalpy, Pressure | Adiabatic constant pressure |
| **Velocity Rescaling** | - | - | Quick equilibration (non-canonical) |

All integrators support:

- Single-system and batched calculations (via `batch_idx`; most integrators also support `atom_ptr` -- see individual function docs for details)
- Automatic differentiation (positions, velocities, forces)
- Both mutating (in-place) and non-mutating (out) variants
- Float32 and float64 precision

## Quick Start

::::{tab-set}

:::{tab-item} Velocity Verlet (NVE)
:sync: verlet

```python
import warp as wp
from nvalchemiops.dynamics.integrators import (
    velocity_verlet_position_update,
    velocity_verlet_velocity_finalize
)

# Setup
positions = wp.array(pos_np, dtype=wp.vec3d, device="cuda:0")
velocities = wp.array(vel_np, dtype=wp.vec3d, device="cuda:0")
forces = wp.array(force_np, dtype=wp.vec3d, device="cuda:0")
masses = wp.array(mass_np, dtype=wp.float64, device="cuda:0")
dt = wp.array([0.001], dtype=wp.float64, device="cuda:0")

# MD loop
for step in range(num_steps):
    # Step 1: Update positions and half-step velocities
    velocity_verlet_position_update(positions, velocities, forces, masses, dt)

    # Step 2: Recalculate forces at new positions
    forces = compute_forces(positions)  # User-defined

    # Step 3: Finalize velocity update
    velocity_verlet_velocity_finalize(velocities, forces, masses, dt)
```

:::

:::{tab-item} Langevin (NVT)
:sync: langevin

```python
import warp as wp
from nvalchemiops.dynamics.integrators import (
    langevin_baoab_half_step,
    langevin_baoab_finalize
)

# Setup (NVT parameters)
temperature = wp.array([1.0], dtype=wp.float64, device="cuda:0")  # kT in energy units
friction = wp.array([1.0], dtype=wp.float64, device="cuda:0")  # friction coefficient

# MD loop
for step in range(num_steps):
    # Step 1: BAOAB half-step (B-A-O-A)
    langevin_baoab_half_step(
        positions, velocities, forces, masses, dt,
        temperature, friction, random_seed=step
    )

    # Step 2: Recalculate forces
    forces = compute_forces(positions)

    # Step 3: Final B step
    langevin_baoab_finalize(velocities, forces, masses, dt)
```

:::

:::{tab-item} Velocity Initialization
:sync: init

```python
import warp as wp
from nvalchemiops.dynamics.utils import (
    initialize_velocities,
    compute_kinetic_energy,
    compute_temperature
)

# Target temperature (k_B*T in energy units)
temperature = wp.array([1.0], dtype=wp.float64, device="cuda:0")

# Scratch arrays for COM removal (required when remove_com=True)
total_momentum = wp.zeros(1, dtype=wp.vec3d, device="cuda:0")
total_mass = wp.zeros(1, dtype=wp.float64, device="cuda:0")
com_velocities = wp.zeros(1, dtype=wp.vec3d, device="cuda:0")

# Initialize velocities from Maxwell-Boltzmann distribution
initialize_velocities(
    velocities, masses, temperature,
    total_momentum, total_mass, com_velocities,
    random_seed=42,
    remove_com=True  # Remove center-of-mass motion
)

# Verify temperature
ke = compute_kinetic_energy(velocities, masses)
T_out = wp.zeros(1, dtype=wp.float64, device="cuda:0")
num_atoms_per_system = wp.array([100], dtype=wp.int32, device="cuda:0")
compute_temperature(ke, T_out, num_atoms_per_system)
print(f"Target: {temperature.numpy()[0]}, Actual: {T_out.numpy()[0]}")
```

:::

::::

## Batch Mode: Simulating Multiple Systems

All integrators support three execution modes for efficient multi-system simulations:

### Single System Mode (Default)

Standard mode for simulating one system:

```python
dt = wp.array([0.001], dtype=wp.float64, device="cuda:0")
velocity_verlet_position_update(positions, velocities, forces, masses, dt)
```

### Batch Mode with `batch_idx` (Atomic Operations)

For systems with varying atom counts, where each atom is tagged with its system ID.
Launches with `dim=num_atoms_total` (one thread per atom):

```python
# 3 systems: 30, 40, and 30 atoms
batch_idx = wp.array([0]*30 + [1]*40 + [2]*30, dtype=wp.int32, device="cuda:0")

# Per-system timesteps
dt = wp.array([0.001, 0.002, 0.0015], dtype=wp.float64, device="cuda:0")

velocity_verlet_position_update(
    positions, velocities, forces, masses, dt, batch_idx=batch_idx
)
```

**Use batch_idx when:**

- Systems have similar sizes
- You want maximum parallelism (one thread per atom)
- Memory access patterns are coalesced

### Batch Mode with `atom_ptr` (Sequential Per-System)

CSR-style pointers defining atom ranges, where each thread processes one complete system.
Launches with `dim=num_systems`:

```python
# Same 3 systems as above: [0:30], [30:70], [70:100]
atom_ptr = wp.array([0, 30, 70, 100], dtype=wp.int32, device="cuda:0")

# Per-system timesteps
dt = wp.array([0.001, 0.002, 0.0015], dtype=wp.float64, device="cuda:0")

velocity_verlet_position_update(
    positions, velocities, forces, masses, dt, atom_ptr=atom_ptr
)
```

**Use atom_ptr when:**

- Systems have very different sizes
- You need per-system operations (reductions, thermostat chains)
- Each system needs independent sequential processing

## Integrator Details

### Velocity Verlet (NVE)

The velocity Verlet algorithm is a second-order symplectic integrator that exactly
conserves energy in the absence of numerical error:

$$
\begin{aligned}
\mathbf{r}(t + \Delta t) &= \mathbf{r}(t) + \mathbf{v}(t) \Delta t + \frac{1}{2} \mathbf{a}(t) \Delta t^2 \\
\mathbf{v}(t + \Delta t) &= \mathbf{v}(t) + \frac{1}{2}[\mathbf{a}(t) + \mathbf{a}(t + \Delta t)] \Delta t
\end{aligned}
$$

**Key Properties:**

- Time-reversible and symplectic
- Excellent long-term energy conservation
- Requires two force evaluations per step (before and after position update)

**References:**

- Swope et al. (1982). J. Chem. Phys. 76, 637

### Langevin Dynamics (NVT)

Langevin dynamics adds friction and random forces to maintain constant temperature.
We implement the BAOAB splitting scheme for optimal configurational sampling:

$$
B: \mathbf{v} \leftarrow \mathbf{v} + \frac{\Delta t}{2m}\mathbf{F} \\
A: \mathbf{r} \leftarrow \mathbf{r} + \frac{\Delta t}{2}\mathbf{v} \\
O: \mathbf{v} \leftarrow e^{-\gamma \Delta t} \mathbf{v} + \sqrt{\frac{k_B T (1 - e^{-2\gamma \Delta t})}{m}} \boldsymbol{\xi} \\
A: \mathbf{r} \leftarrow \mathbf{r} + \frac{\Delta t}{2}\mathbf{v} \\
B: \mathbf{v} \leftarrow \mathbf{v} + \frac{\Delta t}{2m}\mathbf{F}
$$

where $\gamma$ is the friction coefficient and $\boldsymbol{\xi} \sim \mathcal{N}(0, 1)$.

**Key Properties:**

- Maintains canonical (NVT) ensemble
- Friction coefficient $\gamma$ controls thermalization rate
- Stochastic (requires random seed)
- BAOAB splitting provides optimal sampling

**References:**

- Leimkuhler & Matthews (2013). J. Chem. Phys. 138, 174102

### Nosé-Hoover Chain (NVT)

Deterministic thermostat using extended phase space with chain of thermostats:

$$
\begin{aligned}
\dot{\mathbf{r}}_i &= \mathbf{v}_i \\
\dot{\mathbf{v}}_i &= \frac{\mathbf{F}_i}{m_i} - \dot{\eta}_1 \mathbf{v}_i \\
\dot{\eta}_1 &= \frac{2 \cdot KE - N_{\text{DOF}} k_B T}{Q_1} \\
\dot{\eta}_k &= \frac{Q_{k-1} \dot{\eta}_{k-1}^2 - k_B T}{Q_k} \quad (k > 1)
\end{aligned}
$$

**Key Properties:**

- Deterministic (no random forces)
- Rigorously canonical ensemble
- Chain length typically 3-5 for good ergodicity
- Requires thermostat masses $Q_k$ (computed via `nhc_compute_masses`)

**References:**

- Martyna, Tobias, Klein (1994). J Chem Phys, 101, 4177

### Velocity Rescaling

Simple rescaling of velocities to match target temperature:

$$
\mathbf{v}_i \leftarrow \mathbf{v}_i \cdot \sqrt{\frac{T_{\text{target}}}{T_{\text{current}}}}
$$

**Key Properties:**

- Very fast equilibration
- **Does NOT** produce canonical ensemble
- Useful for initial equilibration before switching to proper thermostat
- Can cause artifacts if used for production runs

### NPT (Isothermal-Isobaric)

Constant temperature and pressure simulations using Martyna-Tobias-Klein (MTK)
equations. The convenience `run_npt_step` applies particle thermostat friction
inside its NPT velocity half-steps; it is not a drop-in primitive for integrator
splits that apply particle Nosé-Hoover velocity scaling separately.

```python
from nvalchemiops.dynamics.integrators import run_npt_step

# Run a complete NPT step
run_npt_step(
    positions, velocities, forces, masses, dt,
    cell, cell_velocities,
    target_temperature, target_pressure,
    nhc_positions, nhc_velocities, nhc_masses,
    barostat_mass, dof
)
```

**Key Properties:**

- Maintains constant temperature and pressure
- Supports isotropic (scalar), orthorhombic (3 components), and fully anisotropic (9 components) pressure control
- Uses an NPT velocity primitive that includes particle thermostat drag
- Use `nph_velocity_half_step` or another no-thermostat velocity primitive when composing a separate particle NHC Trotter scaling step

### NPH (Isenthalpic-Isobaric)

Constant enthalpy and pressure simulations without thermostat:

```python
from nvalchemiops.dynamics.integrators import run_nph_step

# Run a complete NPH step
run_nph_step(
    positions, velocities, forces, masses, dt,
    cell, cell_velocities,
    target_pressure,
    barostat_mass, dof
)
```

**Key Properties:**

- Maintains constant pressure without temperature control
- Useful for adiabatic simulations at fixed pressure
- Supports isotropic and anisotropic pressure modes

## Geometry Optimization

### FIRE (Fast Inertial Relaxation Engine)

Accelerated gradient descent for finding energy minima:

```python
import warp as wp
from nvalchemiops.dynamics.optimizers import fire_step

# FIRE control parameters (per-system arrays)
alpha = wp.array([0.1], dtype=wp.float64, device="cuda:0")
dt = wp.array([0.1], dtype=wp.float64, device="cuda:0")
alpha_start = wp.array([0.1], dtype=wp.float64, device="cuda:0")
f_alpha = wp.array([0.99], dtype=wp.float64, device="cuda:0")
dt_min = wp.array([1e-3], dtype=wp.float64, device="cuda:0")
dt_max = wp.array([1.0], dtype=wp.float64, device="cuda:0")
maxstep = wp.array([0.1], dtype=wp.float64, device="cuda:0")
n_steps_positive = wp.array([0], dtype=wp.int32, device="cuda:0")
n_min = wp.array([5], dtype=wp.int32, device="cuda:0")
f_dec = wp.array([0.5], dtype=wp.float64, device="cuda:0")
f_inc = wp.array([1.1], dtype=wp.float64, device="cuda:0")

# Scratch arrays
uphill_flag = wp.array([0], dtype=wp.int32, device="cuda:0")
vf = wp.array([0.0], dtype=wp.float64, device="cuda:0")
vv = wp.array([0.0], dtype=wp.float64, device="cuda:0")
ff = wp.array([0.0], dtype=wp.float64, device="cuda:0")

for step in range(max_steps):
    # Compute forces
    forces = compute_forces(positions)

    # FIRE step (all parameters are arrays)
    fire_step(
        positions, velocities, forces, masses,
        alpha, dt, alpha_start, f_alpha, dt_min, dt_max,
        maxstep, n_steps_positive, n_min, f_dec, f_inc,
        uphill_flag, vf, vv, ff
    )

    # Check convergence
    fmax = wp.max(wp.abs(forces)).numpy()
    if fmax < force_tolerance:
        break
```

**Key Properties:**

- Adaptive timestep and mixing parameter
- Much faster than steepest descent
- Suitable for local minimization (not global search)

### FIRE2 (Improved FIRE)

Improved FIRE optimizer with adaptive damping and velocity mixing:

```python
from nvalchemiops.dynamics.optimizers import fire2_step, fire2_update

# FIRE2 with Warp arrays
fire2_step(
    positions, velocities, forces,
    batch_idx=batch_idx,  # Required for FIRE2
    alpha=alpha, dt=dt, nsteps_inc=nsteps_inc,
    vf=vf, v_sumsq=v_sumsq, f_sumsq=f_sumsq, max_norm=max_norm
)

# Mix/update only for custom final apply phases
fire2_update(
    velocities, forces, batch_idx,
    alpha, dt, nsteps_inc,
    vf, v_sumsq, f_sumsq, max_norm,
)
```

**PyTorch Interface:**

For PyTorch users, FIRE2 has dedicated high-level adapters:

```python
from nvalchemiops.torch import fire2_step_coord, fire2_step_coord_cell

# Coordinate-only optimization
fire2_step_coord(
    positions, velocities, forces, batch_idx,
    alpha, dt, nsteps_inc
)

# Variable-cell optimization (coordinates + cell DOFs)
fire2_step_coord_cell(
    positions, velocities, forces,
    cell, cell_velocities, cell_force,
    batch_idx,
    alpha, dt, nsteps_inc,
    cell_force_scale=1.0,
)
```

**Key Properties:**

- Uses `batch_idx` for batched operations (required)
- `fire2_update` performs FIRE2 reduction, adaptive parameter update, and velocity mixing without applying positions
- `fire2_update(compute_max_norm=False)` skips extended-DOF max-norm reduction for custom final apply phases that recompute their own physical displacement norm
- Applies coupled affine cell/coordinate updates for variable-cell optimization through `fire2_step_coord_cell`
- Normalizes stress-derived cell forces by the number of atoms in each system; `cell_force_scale=1.0` is the default extra multiplier
- Improved convergence compared to original FIRE
- PyTorch adapters handle tensor conversion automatically

### L-BFGS (Limited-memory Quasi-Newton)

L-BFGS builds an implicit approximation to the inverse Hessian from the last
few position and force differences and uses it to pick a search direction, then
takes a step along it bounded by a `maxstep` trust region. It usually reaches a
given force tolerance in far fewer force evaluations than FIRE or FIRE2 — the
cost that dominates relaxation with a machine-learned potential.

**Choosing between FIRE2 and L-BFGS.** FIRE2 costs one force evaluation per
step and carries almost no state, which makes it a good fit for very large
systems or for starting geometries far from any minimum. L-BFGS spends more
memory (`2 * m` history vectors) but converges in far fewer evaluations, and
costs exactly one evaluation per step just as FIRE2 does. If your force
evaluation is expensive relative to a handful of microseconds of kernel time,
prefer L-BFGS.

**Prepare the state once.** `lbfgs_prepare_state` allocates, initializes and
validates every array the optimizer needs and returns them as an `LBFGSState`.
Calling it again is how you restart. A step allocates nothing, so it stays
capturable in a CUDA graph.

The arrays stay yours. `LBFGSState` is a plain dataclass, so every field is
reachable by name — `state.fmax`, `state.status` — and you can build one from
buffers you already have and check it with `state.validate()`, which compares
shapes, dtypes and device without touching the GPU. If you do, match what
preparation produces: zero everything, then `alpha_step` to `1.0`,
`iteration` to `-1` (the "never evaluated" marker) and `status` to
`LBFGS_NEED_EVAL`, which is numerically zero.

See the `nvalchemiops.dynamics.optimizers.lbfgs` module documentation for the
shapes; briefly, with `P` degrees of freedom, `M` systems and history size `m`:
`x_base`, `force_base` and `direction` are `(P,)` vectors, `s_history` and
`y_history` are `(m, P)`, `ys`/`yy`/`alpha_hist`/`beta_hist` are `(m, M)`
float64, `ss` through `alpha_step` are `(M,)` float64, and `status` through
`history_count` are `(M,)` int32. The per-system scalars stay float64 whatever
precision the coordinates use.

**You own the loop.** Each `lbfgs_step` call consumes exactly one force
evaluation. Inspect `status` to decide when to stop:

```python
import numpy as np
import warp as wp
from nvalchemiops.dynamics.optimizers import (
    LBFGS_NEED_EVAL,
    lbfgs_prepare_state,
    lbfgs_step,
)

state = lbfgs_prepare_state(num_atoms, num_systems, device=device)

while True:
    forces = model(positions)
    lbfgs_step(
        positions=positions,
        forces=forces,
        state=state,
        batch_idx=batch_idx,
        n_particles=n_particles,
        force_tol=0.05,   # eV/A, on the largest per-atom force
        maxstep=0.2,      # A, largest displacement in one step
    )
    if not (state.status.numpy() == LBFGS_NEED_EVAL).any():
        break
```

The PyTorch binding takes the same state and mutates it in place. The JAX
binding *returns* a new one, since JAX arrays are immutable; both state classes
are registered pytrees there, so a state crosses `jax.jit` as a single argument
and one `donate_argnums` entry donates every field.

`status` takes three values per system:

| Value | Meaning |
| --- | --- |
| `LBFGS_NEED_EVAL` | Keep going; `positions` hold a new trial point. |
| `LBFGS_CONVERGED` | Done; `positions` hold the relaxed geometry. |

There is no failure status. A line search can exhaust its budget and give up; a
trust-region step cannot, because a short enough step along a descent direction
always makes progress, and an uphill direction is replaced by steepest descent
with the history discarded.

**Batching.** Systems are identified by a sorted `batch_idx` and relax
independently: each keeps its own history and trust-region step, and systems
that finish early are skipped by the remaining kernels. Note that
`n_particles` is the atom count per system, used by the optional RMS
convergence criterion.

**No energy is needed.** The step length comes from the `maxstep` trust region
rather than from a line search comparing energies, so the optimizer never reads
one. That is deliberate: an Armijo test compares *total energies* while the
search direction comes from *forces*, and for a model whose forces are not the
gradient of its reported energy — a direct force head, say — the two describe
different surfaces and the test rejects good steps no matter how it is tuned.
A consequence worth knowing: without Armijo the energy is not guaranteed to
decrease monotonically, though the force does converge.

**Variable cell.** The packed layout interleaves each system's atoms with its
two cell entries, so the `LBFGSState` must be sized for
`num_atoms + 2 * num_systems` degrees of freedom, and a second
`LBFGSCellState` from `lbfgs_prepare_cell_state` carries the chart. Build
`ext_atom_ptr` and `ext_batch_idx` with the generic `extend_atom_ptr` and
`atom_ptr_to_batch_idx` utilities and pass them in; because nothing assumes a
uniform split, ragged batches whose systems have different atom counts work the
same way uniform ones do. Pass `cell` and `n_particles` to the preparation
function to have the chart captured there, or fill it yourself with
`lbfgs_set_reference_cell` and `lbfgs_cell_kappa` before the first step.

**Memory.** The history dominates: `2 * m` vectors of `num_dofs` each. At
`m = 6` and float32 coordinates that is roughly `192` bytes per degree of
freedom. Reduce `m` if memory is tight; `m` between 3 and 7 is typical.

## Temperature Control Utilities

### Computing Temperature

```python
import warp as wp
from nvalchemiops.dynamics.utils import (
    compute_kinetic_energy,
    compute_temperature
)

# Compute kinetic energy
ke = compute_kinetic_energy(velocities, masses)

# Convert to temperature (assumes k_B = 1)
T_out = wp.zeros(1, dtype=wp.float64, device=velocities.device)
num_atoms_per_system = wp.array([100], dtype=wp.int32, device=velocities.device)
compute_temperature(ke, T_out, num_atoms_per_system)
```

## Cell Utilities

For periodic boundary conditions and variable-cell simulations:

```python
from nvalchemiops.dynamics.utils import (
    compute_cell_volume,
    compute_cell_inverse,
    scale_positions_with_cell,
    wrap_positions_to_cell,
    cartesian_to_fractional,
    fractional_to_cartesian,
)

# Compute cell volume
volume = compute_cell_volume(cell)

# Wrap positions into primary cell
wrap_positions_to_cell(positions, cell)

# Scale positions when cell changes (preserving fractional coordinates)
scale_positions_with_cell(positions, cell_old, cell_new)
```

## Constraint Utilities (SHAKE/RATTLE)

Holonomic constraints for fixing bond lengths:

```python
import warp as wp
from nvalchemiops.dynamics.utils import shake_constraints, rattle_constraints

max_error = wp.zeros(1, dtype=wp.float64, device=positions.device)

# After position update: correct positions to satisfy constraints
shake_constraints(
    positions, positions_old, masses,
    bond_atom_i, bond_atom_j, bond_lengths_sq,
    max_error, num_iter=10,
)

# After velocity update: correct velocities to satisfy constraints
rattle_constraints(
    positions, velocities, masses,
    bond_atom_i, bond_atom_j,
    max_error, num_iter=10,
)
```

## Common Pitfalls

### Mutating vs Non-Mutating Functions

```python
# WRONG: Using mutating function incorrectly
new_positions = velocity_verlet_position_update(...)  # Returns None!

# CORRECT: Use non-mutating variant (requires pre-allocated output arrays)
positions_out = wp.zeros_like(positions)
velocities_out = wp.zeros_like(velocities)
positions_out, velocities_out = velocity_verlet_position_update_out(
    positions, velocities, forces, masses, dt,
    positions_out, velocities_out,
)

# CORRECT: Use mutating variant in-place
velocity_verlet_position_update(positions, velocities, forces, masses, dt)
# positions and velocities are now modified
```

### Timestep as Array

All integrators require timestep as a Warp array, not a scalar:

```python
# WRONG
velocity_verlet_position_update(positions, velocities, forces, masses, 0.001)

# CORRECT
dt = wp.array([0.001], dtype=wp.float64, device="cuda:0")
velocity_verlet_position_update(positions, velocities, forces, masses, dt)
```

### Batch Mode Mutual Exclusivity

Cannot use both `batch_idx` and `atom_ptr` simultaneously:

```python
# WRONG
velocity_verlet_position_update(
    positions, velocities, forces, masses, dt,
    batch_idx=batch_idx,
    atom_ptr=atom_ptr  # Error: provide one or the other
)

# CORRECT
velocity_verlet_position_update(
    positions, velocities, forces, masses, dt,
    batch_idx=batch_idx
)
```

## Further Reading

- {doc}`/modules/warp/dynamics` - Full Warp API reference
- {doc}`/modules/torch/dynamics` - PyTorch FIRE2 adapter reference
- Examples: `examples/dynamics/` in the repository
- [NVIDIA Warp Documentation](https://nvidia.github.io/warp/)
