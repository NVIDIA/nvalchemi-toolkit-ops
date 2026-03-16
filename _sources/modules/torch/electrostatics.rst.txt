:mod:`nvalchemiops.torch.interactions.electrostatics`: Electrostatics
========================

.. currentmodule:: nvalchemiops.torch.interactions.electrostatics

The electrostatics module provides GPU-accelerated implementations of
long-range electrostatic interactions for molecular simulations with **PyTorch** bindings.
These functions accept standard ``torch.Tensor`` inputs and support automatic differentiation.
Ewald and PME support full autograd for positions, charges, and cell parameters.
DSF supports charge gradients via autograd; forces and virials are computed analytically.

.. tip::
    For the underlying framework-agnostic Warp kernels, see :doc:`../warp/electrostatics`.

High-Level Interface
--------------------

These are the primary entry points for most users.

.. autofunction:: ewald_summation
.. autofunction:: particle_mesh_ewald

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
.. autofunction:: pme_green_structure_factor
.. autofunction:: pme_energy_corrections
.. autofunction:: pme_energy_corrections_with_charge_grad

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
