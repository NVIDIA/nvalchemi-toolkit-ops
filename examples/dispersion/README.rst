Dispersion Interactions
=======================

Examples demonstrating GPU-accelerated computation of DFT-D3
dispersion corrections with environment-dependent C6 coefficients.

These examples show how to:

* Compute dispersion energies and forces for molecules
* Process batches of crystal structures
* Integrate with PyTorch for differentiable workflows

Examples 01 to 03 use the direct real-space ``dftd3``, which handles molecules and
open boundary conditions.

Examples 04 and 05 use ``fourier_dftd3``, which evaluates the correction on a
particle mesh with no pair cutoff on the dispersion sum. It is **periodic only**,
so both are crystal examples --- a CsCl cell built from the published DFT-D3
tables:

* ``04_fourier_dftd3_crystal.py`` --- the Torch API, and how to check mesh
  sensitivity
* ``05_jax_fourier_dftd3_crystal.py`` --- the JAX API under ``jax.jit``
