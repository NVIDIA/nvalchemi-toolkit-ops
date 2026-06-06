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
* Train on forces, stress, and charge gradients via the energy-derivative
  contract (the recommended replacement for the deprecated direct-output flags)

The full Torch Ewald/PME APIs support first- and second-order energy-derived
training workflows. The full JAX Ewald/PME APIs support first-order
energy-derived gradients for positions, charges, and strain-consistent cell
gradients; JAX PME higher-order derivatives raise ``NotImplementedError`` until
a native PME Hessian-vector product is available.
