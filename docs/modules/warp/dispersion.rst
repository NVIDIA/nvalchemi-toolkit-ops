:mod:`nvalchemiops.interactions.dispersion`: Dispersion Corrections
===================================================================

.. automodule:: nvalchemiops.interactions.dispersion
    :no-members:
    :no-inherited-members:

Warp-Level Interface
--------------------

DFT-D3(BJ) Dispersion Corrections
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tip::
   This is the low-level Warp interface that operates on ``warp.array`` objects.
   For PyTorch tensor support, see :doc:`../torch/dispersion`.

The DFT-D3 implementation supports two neighbor representation formats:

- **Neighbor matrix** (dense): ``[num_atoms, max_neighbors]`` with padding
- **Neighbor list** (sparse CSR): Compressed sparse row format with ``idx_j`` and ``neighbor_ptr``

Both formats produce identical results and support all features including periodic
boundary conditions, batching, and smooth cutoff functions.

Non-Periodic Systems
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: nvalchemiops.interactions.dispersion._dftd3.dftd3_matrix
.. autofunction:: nvalchemiops.interactions.dispersion._dftd3.dftd3

Periodic Boundary Conditions (PBC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: nvalchemiops.interactions.dispersion._dftd3.dftd3_matrix_pbc
.. autofunction:: nvalchemiops.interactions.dispersion._dftd3.dftd3_pbc

Pairwise Dispersion and Dispersion PME (LJ-PME)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Framework-agnostic Warp launchers for the geometric-rule dispersion
(:math:`r^{-6}`) interactions. The real-space launchers mirror the Coulomb
launchers; the PME launchers provide the reciprocal-space convolve and energy
corrections.

.. autofunction:: nvalchemiops.interactions.dispersion.lj_real_kernels.lj_dispersion_energy
.. autofunction:: nvalchemiops.interactions.dispersion.lj_real_kernels.lj_dispersion_energy_forces
.. autofunction:: nvalchemiops.interactions.dispersion.lj_pme_kernels.lj_pme_convolve
.. autofunction:: nvalchemiops.interactions.dispersion.lj_pme_kernels.lj_pme_energy_corrections
