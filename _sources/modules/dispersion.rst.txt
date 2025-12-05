:mod:`nvalchemiops.interactions.dispersion`: Dispersion Corrections
===================================================================

.. automodule:: nvalchemiops.interactions.dispersion
    :no-members:
    :no-inherited-members:

High-Level Interface
--------------------

DFT-D3(BJ) Dispersion Corrections
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The DFT-D3 implementation supports two neighbor representation formats:

- **Neighbor matrix** (dense): ``[num_atoms, max_neighbors]`` with padding
- **Neighbor list** (sparse COO): ``[2, num_pairs]`` without padding

Both formats produce identical results and support all features including periodic
boundary conditions, batching, and smooth cutoff functions. The high-level wrapper
automatically dispatches to the appropriate kernels based on which format is provided.

.. autofunction:: nvalchemiops.interactions.dispersion.dftd3

Data Structures
---------------

This data structure is not necessarily required to use the kernels, however is provided
for convenience---the ``dataclass`` will validate shapes and keys for parameters
required by the kernels.

.. autoclass:: nvalchemiops.interactions.dispersion.dftd3.D3Parameters
    :members:
    :undoc-members:
