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

Multipole Electrostatics
------------------------

GTO-smeared multipole electrostatics for systems carrying per-atom charges,
dipoles, and quadrupoles (``l_max`` 0/1/2). Moments are passed as a single
packed ``multipole_moments`` tensor built with :func:`pack_multipole_moments`.

High-Level Interface
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: multipole_ewald_summation

.. autofunction:: nvalchemiops.torch.interactions.electrostatics.pme_multipole.multipole_particle_mesh_ewald

.. currentmodule:: nvalchemiops.torch.interactions.electrostatics

Energy Components
~~~~~~~~~~~~~~~~~

.. autofunction:: multipole_electrostatic_energy
.. autofunction:: multipole_real_space_energy
.. autofunction:: multipole_reciprocal_space_energy

Atom-Centered Features
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: multipole_electrostatic_features

Moment Packing
~~~~~~~~~~~~~~

.. autofunction:: pack_multipole_moments

SCF Cache (Amortized Workflow)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reuse the position-independent reciprocal-space state across many
evaluations at fixed cell (MD steps / SCF iterations).

.. autofunction:: prepare_multipole_scf_cache
.. autofunction:: multipole_scf_step_energy
.. autofunction:: multipole_scf_step_features

.. autoclass:: MultipoleSCFCache
   :members:

Parameter Estimation
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: estimate_multipole_ewald_parameters
.. autofunction:: estimate_multipole_pme_parameters

.. autoclass:: MultipoleEwaldParameters
   :members:

.. autoclass:: MultipolePMEParameters
   :members:
