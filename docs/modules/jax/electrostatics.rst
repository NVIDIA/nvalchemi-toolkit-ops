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
such as ``alpha`` and mesh controls are constants. The lower-level
``ewald_reciprocal_space`` component follows the tangent carried by its
``k_vectors`` argument. Vectors constructed from ``cell`` in the traced
computation therefore contribute their reciprocal-cell derivative. A
precomputed array, or one passed through ``jax.lax.stop_gradient``, has zero
tangent and is fixed for that differentiation. Full
``ewald_summation(k_vectors=...)`` and PME precomputed metadata treat explicit
vectors as static metadata. Use ``ewald_reciprocal_space_from_miller_indices``
or ``ewald_summation(miller_indices=...)`` for retained topology materialized
from the live cell. Energy-returning Ewald, PME, and slab paths support atom-weighted
losses such as ``(weights * energies).sum()`` for positions, charges, and
supported cell derivatives. Monopole entry points provide the keyword-only
``energy_reduction`` option. The default, ``"atom"``, returns one energy per
atom with shape ``(N,)``. Set it to ``"system"`` to sum energies within each
system and return shape ``(B,)``. Other requested outputs keep their existing
shapes.
For changing-cell full Ewald, retain signed Miller indices with
``generate_ewald_miller_indices`` and pass them through ``ewald_summation``;
the full-Ewald custom-JVP then includes the reciprocal-cell tangent. A retained
topology must cover all intended cells. Enlarging it changes ``K`` and can
trigger JIT recompilation. The legacy ``generate_miller_indices`` remains the
batched bounds helper for static-shape JIT generation.
JAX PME supports first-order cell/strain gradients,
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
.. autofunction:: ewald_reciprocal_space_from_miller_indices

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
.. autofunction:: generate_ewald_miller_indices
.. autofunction:: k_vectors_from_miller_indices
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
