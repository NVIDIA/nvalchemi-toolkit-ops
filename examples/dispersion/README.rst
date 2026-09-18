Dispersion Interactions
=======================

Examples demonstrating GPU-accelerated computation of DFT-D3
dispersion corrections with environment-dependent C6 coefficients.

These examples show how to:

* Compute dispersion energies and forces for molecules
* Process batches of crystal structures
* Integrate with PyTorch for differentiable workflows
* Evaluate the correction on a particle mesh, with no cutoff on the dispersion
  sum, using FourierD3
* Compile the JAX pipeline with ``jax.jit``
