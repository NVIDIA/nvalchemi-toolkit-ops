:mod:`nvalchemiops.jax.interactions.electrostatics`: Electrostatics
====================================================================

.. currentmodule:: nvalchemiops.jax.interactions.electrostatics

The electrostatics module provides GPU-accelerated implementations of
long-range electrostatic interactions for molecular simulations with **JAX** bindings.
These functions accept standard ``jax.Array`` inputs.

.. tip::
    For the underlying framework-agnostic Warp kernels, see :doc:`../warp/electrostatics`.

High-Level Interface
--------------------

These are the primary entry points for most users; these methods should
be ``jax.jit`` compatible.

.. autofunction:: ewald_summation
.. autofunction:: particle_mesh_ewald

Coulomb Interactions
--------------------

Direct pairwise Coulomb interactions.

.. autofunction:: coulomb_energy
.. autofunction:: coulomb_forces
.. autofunction:: coulomb_energy_forces

Ewald Components
----------------

Individual components of the Ewald summation method.

.. autofunction:: ewald_real_space
.. autofunction:: ewald_reciprocal_space

PME Components
--------------

Individual components of the Particle Mesh Ewald method.

.. autofunction:: pme_reciprocal_space
.. autofunction:: pme_green_structure_factor
.. autofunction:: pme_energy_corrections
.. autofunction:: pme_energy_corrections_with_charge_grad

K-Vector Generation
-------------------

.. autofunction:: generate_miller_indices
.. autofunction:: generate_k_vectors_ewald_summation
.. autofunction:: generate_k_vectors_pme

Parameter Estimation
--------------------

Functions for automatic parameter estimation based on desired accuracy tolerance.

.. autofunction:: estimate_ewald_parameters
.. autofunction:: estimate_pme_parameters
.. autofunction:: estimate_pme_mesh_dimensions
.. autofunction:: mesh_spacing_to_dimensions

.. autoclass:: EwaldParameters
   :members:

.. autoclass:: PMEParameters
   :members:
