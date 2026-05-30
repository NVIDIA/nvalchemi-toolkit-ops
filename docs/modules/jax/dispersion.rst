:mod:`nvalchemiops.jax.interactions.dispersion`: Dispersion Corrections
========================================================================

.. currentmodule:: nvalchemiops.jax.interactions.dispersion

The dispersion module provides JAX bindings for the GPU-accelerated
implementations of dispersion interactions.

.. automodule:: nvalchemiops.jax.interactions.dispersion
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
The method should be ``jax.jit`` compatible.

.. autofunction:: nvalchemiops.jax.interactions.dispersion.dftd3

Data Structures
---------------

This data structure is not necessarily required to use the kernels, however is provided
for convenience---the ``dataclass`` will validate shapes and keys for parameters
required by the kernels.

.. autoclass:: nvalchemiops.jax.interactions.dispersion.D3Parameters
    :members:
    :undoc-members:

Internal Implementation
-----------------------

These are low-level implementation functions that wrap the Warp kernels for JAX.
For most use cases, prefer the high-level :func:`dftd3` wrapper above.

Neighbor Matrix Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.jax.interactions.dispersion._dftd3._dftd3_nm_impl

Neighbor List Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.jax.interactions.dispersion._dftd3._dftd3_nl_impl

Pairwise Dispersion and Dispersion PME (LJ-PME)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

JAX bindings for the geometric-rule dispersion (:math:`r^{-6}`) interactions,
mirroring the PyTorch API.

.. autofunction:: nvalchemiops.jax.interactions.dispersion.lj_dispersion_energy
.. autofunction:: nvalchemiops.jax.interactions.dispersion.lj_dispersion_forces
.. autofunction:: nvalchemiops.jax.interactions.dispersion.lj_dispersion_energy_forces
.. autofunction:: nvalchemiops.jax.interactions.dispersion.dispersion_reciprocal_space
.. autofunction:: nvalchemiops.jax.interactions.dispersion.dispersion_pme
.. autofunction:: nvalchemiops.jax.interactions.dispersion.estimate_dispersion_pme_parameters
