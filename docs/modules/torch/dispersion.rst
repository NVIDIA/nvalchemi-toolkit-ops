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

FourierD3: Particle-Mesh DFT-D3
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Evaluates a periodic particle-mesh variant of DFT-D3(BJ) in :math:`O(N \log N)`, using the
same damped dispersion lattice sum with a modified coordination-number model, and with no
pair cutoff on the dispersion sum. Both neighbour formats are supported, as above.

Two differences from :func:`dftd3` are worth noting before use:

- ``cell`` is **required**. FourierD3 evaluates a periodic sum; there is no open-boundary
  path. Molecules in vacuum should continue to use :func:`dftd3`.
- ``cutoff`` has **no default**, and must equal the radius the neighbour list was built
  with. The coordination-number function is constructed to reach zero exactly there.

.. autofunction:: nvalchemiops.torch.interactions.dispersion.fourier_dftd3

Data Structures
---------------

This data structure is not necessarily required to use the kernels, however is provided
for convenience---the ``dataclass`` will validate shapes and keys for parameters
required by the kernels.

.. autoclass:: nvalchemiops.torch.interactions.dispersion.D3Parameters
    :members:
    :undoc-members:

.. autoclass:: nvalchemiops.torch.interactions.dispersion.FourierD3Parameters
    :members:
    :undoc-members:

:class:`FourierD3Setup` is a different kind of object: a cache of cell- and mesh-derived
quantities, optional for correctness but required for
``torch.compile(mode="reduce-overhead")``, since ``torch.linalg.inv`` cannot be recorded into
a CUDA graph. It is tied to the cell, batch size, species count, precision and device it was
built for, and refuses any call that does not match.

.. autoclass:: nvalchemiops.torch.interactions.dispersion.FourierD3Setup
    :members:
    :undoc-members:

Internal Custom Operators
-------------------------

These are low-level custom operators that wrap the Warp kernels. For most use cases,
prefer the high-level :func:`dftd3` wrapper above. These operators are exposed for
advanced users who need fine-grained control or ``torch.compile`` compatibility.

Non-Periodic Systems
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.torch.interactions.dispersion._dftd3._dftd3_matrix_op
.. autofunction:: nvalchemiops.torch.interactions.dispersion._dftd3._dftd3_op

Periodic Boundary Conditions (PBC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.torch.interactions.dispersion._dftd3._dftd3_matrix_pbc_op
.. autofunction:: nvalchemiops.torch.interactions.dispersion._dftd3._dftd3_pbc_op

FourierD3 Operators
^^^^^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.torch.interactions.dispersion._fourier_dftd3._fd3_prologue_op
.. autofunction:: nvalchemiops.torch.interactions.dispersion._fourier_dftd3._fd3_spread_op
.. autofunction:: nvalchemiops.torch.interactions.dispersion._fourier_dftd3._fd3_kspace_op
.. autofunction:: nvalchemiops.torch.interactions.dispersion._fourier_dftd3._fd3_gather_op
.. autofunction:: nvalchemiops.torch.interactions.dispersion._fourier_dftd3._fd3_finalise_op
