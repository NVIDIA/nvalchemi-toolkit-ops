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

These are the primary entry points for most users. They are compatible with
``jax.jit`` when setup-only PME parameters such as ``mesh_dimensions`` and
``alpha`` are supplied explicitly whenever those values would otherwise be
estimated from traced inputs. ``miller_bounds`` is also a static shape control:
under ``jax.jit``, pass it as a concrete tuple or build ``k_vectors`` outside
the compiled function.
Energy derivatives are defined for positions, charges, and cell. Setup values
such as ``alpha`` and mesh controls are constants; precomputed reciprocal
metadata such as ``k_vectors``, ``k_squared``, ``volume``, and ``cell_inv_t`` is
accepted for cell-differentiated calls as static metadata that is assumed to
correspond to the current ``cell``; cache-generation derivatives are not
recovered. Energy-returning Ewald, PME, and slab paths support atom-weighted
losses such as ``(weights * energies).sum()`` for positions, charges, and
supported cell derivatives. JAX PME supports first-order cell/strain gradients,
but PME cell/strain HVPs, including full PME with ``slab_correction=True``, are
explicitly unsupported until a native transposable PME cell-HVP path is
implemented and tested.
Point-charge Ewald/PME inputs support ``float32`` and ``float64``. Keep all
floating inputs and precomputed metadata in a call on a consistent dtype.

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
.. autofunction:: compute_bspline_moduli_1d

Slab Correction
---------------

Explicit-output Yeh-Berkowitz/Ballenegger slab correction for systems with two
periodic directions. Component-level calls can request energies, forces, charge
gradients, and virials with the same flags used by the Ewald and PME wrappers.
The high-level Ewald and PME wrappers can include the slab term in their energy
autodiff path.

.. autofunction:: compute_slab_correction

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
