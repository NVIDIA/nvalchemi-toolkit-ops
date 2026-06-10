Electrostatics
==============

Examples demonstrating GPU-accelerated computation of long-range electrostatic
interactions in periodic systems using Coulomb, Ewald summation, and Particle
Mesh Ewald (PME).

These examples show how to:

* Compute direct Coulomb interactions (damped and undamped)
* Use Ewald summation for periodic systems with automatic parameter estimation
* Apply two-dimensional slab corrections for Ewald and PME interfacial systems
* Apply Particle Mesh Ewald (PME) for O(N log N) scaling
* Work with neighbor list and neighbor matrix formats
* Perform batch evaluation for multiple systems
* Leverage autograd for computing forces and gradients
* Compute multipole Ewald and PME totals with charges, dipoles, and
  quadrupoles (l_max = 0 / 1 / 2), including stress tensors via the
  cell gradient and force-loss-style training via the second-order
  backward Warp kernel
* Extract atom-centered multipole features by projecting the periodic
  potential onto receiver GTOs
* Amortize the position-independent reciprocal-space state with the
  multipole SCF cache and reuse it across many step evaluations
* Train on forces, stress, and charge gradients via the energy-derivative
  contract (the recommended replacement for the deprecated direct-output flags)

The full Torch Ewald/PME APIs support first- and second-order energy-derived
training workflows. The full JAX Ewald/PME APIs support first-order
energy-derived gradients for positions, charges, and row-vector displacement
virials. Higher-order JAX support is limited to tested position and charge
scalar losses; PME cell/stress/strain higher-order derivatives are unsupported.
Electrostatics does not expose public Hessian or Jacobian APIs.

Point-charge Ewald/PME examples use ``float64`` for accuracy-sensitive
reciprocal-space calculations and gradient checks. The APIs also support
``float32`` when throughput is the priority; keep all floating inputs and
precomputed metadata in a call on a consistent dtype.
