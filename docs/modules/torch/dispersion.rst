:mod:`nvalchemiops.torch.interactions.dispersion`: Dispersion Corrections
==========================================================================

.. currentmodule:: nvalchemiops.torch.interactions.dispersion

The dispersion module provides PyTorch-bindings for the GPU accelerated
implementations of dispersion interactions.

.. automodule:: nvalchemiops.torch.interactions.dispersion
    :no-members:
    :no-inherited-members:

.. tip::
    For the underlying framework-agnostic Warp kernels, see :doc:`../warp/dispersion`.

High-Level Interface
--------------------

DFT-D3(BJ) Dispersion Corrections
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The DFT-D3 implementation supports two neighbor representation formats:

- **Neighbor matrix** (dense): ``[num_atoms, max_neighbors]`` with padding
- **Neighbor list** (sparse CSR): Compressed sparse row format with ``idx_j`` and ``neighbor_ptr``

Both formats produce identical results and support all features including periodic
boundary conditions, batching, and smooth cutoff functions. The high-level wrapper
automatically dispatches to the appropriate kernels based on which format is provided.

.. autofunction:: nvalchemiops.torch.interactions.dispersion.dftd3

Data Structures
---------------

This data structure is not necessarily required to use the kernels, however is provided
for convenience---the ``dataclass`` will validate shapes and keys for parameters
required by the kernels.

.. autoclass:: nvalchemiops.torch.interactions.dispersion.D3Parameters
    :members:
    :undoc-members:

Internal Custom Operators
-------------------------

These are low-level custom operators that wrap the Warp kernels. For most use cases,
prefer the high-level :func:`dftd3` wrapper above. These operators are exposed for
advanced users who need fine-grained control or ``torch.compile`` compatibility.

Non-Periodic Systems
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: nvalchemiops.torch.interactions.dispersion._dftd3._dftd3_matrix_op
.. autofunction:: nvalchemiops.torch.interactions.dispersion._dftd3._dftd3_op

Periodic Boundary Conditions (PBC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: nvalchemiops.torch.interactions.dispersion._dftd3._dftd3_matrix_pbc_op
.. autofunction:: nvalchemiops.torch.interactions.dispersion._dftd3._dftd3_pbc_op

Pairwise Dispersion and Dispersion PME (LJ-PME)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Long-range van der Waals dispersion (attractive :math:`r^{-6}`) using the
geometric C6 combination rule. ``lj_dispersion_*`` are the direct pairwise
analogs of ``coulomb_*``; ``dispersion_pme`` is the mesh-Ewald (LJ-PME) analog
of ``particle_mesh_ewald``. Inputs are per-atom LJ :math:`\sigma`/:math:`\epsilon`
(converted internally to :math:`C_6`).

.. autofunction:: nvalchemiops.torch.interactions.dispersion.lj_dispersion_energy
.. autofunction:: nvalchemiops.torch.interactions.dispersion.lj_dispersion_forces
.. autofunction:: nvalchemiops.torch.interactions.dispersion.lj_dispersion_energy_forces
.. autofunction:: nvalchemiops.torch.interactions.dispersion.dispersion_reciprocal_space
.. autofunction:: nvalchemiops.torch.interactions.dispersion.dispersion_pme
.. autofunction:: nvalchemiops.torch.interactions.dispersion.estimate_dispersion_pme_parameters
.. autoclass:: nvalchemiops.torch.interactions.dispersion.DispersionPMEParameters
    :members:
