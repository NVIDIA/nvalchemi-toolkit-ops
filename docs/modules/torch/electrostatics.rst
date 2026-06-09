:mod:`nvalchemiops.torch.interactions.electrostatics`: Electrostatics
======================================================================

.. currentmodule:: nvalchemiops.torch.interactions.electrostatics

The electrostatics module provides GPU-accelerated implementations of
long-range electrostatic interactions for molecular simulations with **PyTorch** bindings.
These functions accept standard ``torch.Tensor`` inputs and support automatic differentiation.
Ewald and PME support full autograd for positions, charges, and cell parameters.
DSF supports charge gradients via autograd; forces and virials are computed analytically.
Setup parameters such as ``alpha``, cutoffs, mesh controls, batch metadata, and
neighbor topology are treated as constants. Cell-derived caches such as
``k_vectors``, ``k_squared``, ``volume``, and ``cell_inv_t`` are accepted when
``cell.requires_grad`` is true, but they are static metadata and are assumed to
correspond to the current ``cell``; their cache-generation derivatives are not
recovered. Energy-returning Ewald, PME, and slab paths support atom-weighted
losses such as ``(weights * energies).sum()`` for positions, charges, and
supported cell derivatives.
Point-charge Ewald/PME inputs support ``float32`` and ``float64``. Keep all
floating inputs and precomputed metadata in a call on a consistent dtype.

.. tip::
    For the underlying framework-agnostic Warp kernels, see :doc:`../warp/electrostatics`.

High-Level Interface
--------------------

These are the primary entry points for most users.

.. autofunction:: ewald_summation
.. autofunction:: particle_mesh_ewald

Slab Correction
---------------

Two-dimensional slab correction for systems with two periodic axes and one
non-periodic axis. The high-level Ewald and PME interfaces can add this
correction directly. Component-level workflows should add
``compute_slab_correction`` explicitly to ``ewald_real_space`` plus either
``ewald_reciprocal_space`` for Ewald or ``pme_reciprocal_space`` for PME.

.. autofunction:: compute_slab_correction

Coulomb Interactions
--------------------

Direct pairwise Coulomb interactions.

.. autofunction:: coulomb_energy
.. autofunction:: coulomb_forces
.. autofunction:: coulomb_energy_forces

DSF Coulomb
-----------

Damped Shifted Force (DSF) pairwise electrostatics with :math:`\mathcal{O}(N)` scaling.

.. autofunction:: dsf_coulomb

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

K-Vector Generation
-------------------

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
