:mod:`nvalchemiops.interactions.electrostatics`: Electrostatic Interactions (Warp)
==================================================================================

.. automodule:: nvalchemiops.interactions.electrostatics
    :no-members:
    :no-inherited-members:

Core Warp Kernels
-----------------

This module provides the framework-agnostic **NVIDIA Warp** kernels for electrostatic interactions.
These functions operate directly on ``warp.array`` objects and can be used to build custom
integrators or bindings for other frameworks.

.. note::
    For PyTorch-compatible functions that accept ``torch.Tensor`` inputs and support autograd,
    please see :doc:`../torch/electrostatics`.

Coulomb Kernels
^^^^^^^^^^^^^^^

Direct pairwise Coulomb interactions.

.. autofunction:: coulomb_energy
.. autofunction:: coulomb_energy_forces
.. autofunction:: coulomb_energy_matrix
.. autofunction:: coulomb_energy_forces_matrix
.. autofunction:: batch_coulomb_energy
.. autofunction:: batch_coulomb_energy_forces
.. autofunction:: batch_coulomb_energy_matrix
.. autofunction:: batch_coulomb_energy_forces_matrix

DSF Kernels
^^^^^^^^^^^

Damped Shifted Force (DSF) pairwise electrostatic summation.

.. autofunction:: dsf_csr
.. autofunction:: dsf_matrix

Ewald Kernels
^^^^^^^^^^^^^

Ewald summation kernels for real-space and reciprocal-space.

**Real-Space:**

.. autofunction:: ewald_real_space_energy
.. autofunction:: ewald_real_space_energy_forces
.. autofunction:: ewald_real_space_energy_matrix
.. autofunction:: ewald_real_space_energy_forces_matrix
.. autofunction:: batch_ewald_real_space_energy
.. autofunction:: batch_ewald_real_space_energy_forces
.. autofunction:: batch_ewald_real_space_energy_matrix
.. autofunction:: batch_ewald_real_space_energy_forces_matrix

**Reciprocal-Space:**

.. autofunction:: ewald_reciprocal_space_fill_structure_factors
.. autofunction:: ewald_reciprocal_space_compute_energy
.. autofunction:: ewald_subtract_self_energy
.. autofunction:: ewald_reciprocal_space_energy_forces
.. autofunction:: batch_ewald_reciprocal_space_fill_structure_factors
.. autofunction:: batch_ewald_reciprocal_space_compute_energy
.. autofunction:: batch_ewald_subtract_self_energy
.. autofunction:: batch_ewald_reciprocal_space_energy_forces

PME Kernels
^^^^^^^^^^^

Particle Mesh Ewald kernels. Note that FFT operations are typically offloaded to the host framework.

.. warning::
   Keep in mind that currently, convolution required by the PME algorithm
   needs an FFT interface, and as of the current release, ``warp`` does not
   have an exact analogous method to the API used in PyTorch (e.g. ``ffttreq``).
   For this reason, the ``warp`` kernel set cannot be run completely end-to-end
   in the same way as the other kernels can be. See the PyTorch bindings instead.

.. autofunction:: pme_green_structure_factor
.. autofunction:: batch_pme_green_structure_factor
.. autofunction:: pme_energy_corrections
.. autofunction:: pme_energy_corrections_with_charge_grad
.. autofunction:: batch_pme_energy_corrections
.. autofunction:: batch_pme_energy_corrections_with_charge_grad
