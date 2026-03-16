:mod:`nvalchemiops.torch`: Dynamics Optimizers
===================================================

.. currentmodule:: nvalchemiops.torch

The dynamics module provides PyTorch bindings for GPU-accelerated geometry
optimization algorithms.

.. tip::
    For the underlying framework-agnostic Warp kernels and full MD integrators,
    see :doc:`../warp/dynamics`.

.. automodule:: nvalchemiops.torch
    :no-members:
    :no-inherited-members:

FIRE2 Optimizer
---------------

PyTorch adapter for the FIRE2 (Fast Inertial Relaxation Engine v2) geometry optimizer.
These functions accept PyTorch tensors, allocate scratch buffers via PyTorch's CUDA
caching allocator, and call the pure-Warp FIRE2 kernels.

Coordinate-Only Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.torch.fire2.fire2_step_coord

Variable-Cell Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^

For optimizing both atomic coordinates and simulation cell parameters simultaneously.

.. autofunction:: nvalchemiops.torch.fire2.fire2_step_coord_cell

Extended Array Interface
^^^^^^^^^^^^^^^^^^^^^^^^

For advanced use cases where you manage packed extended arrays directly.

.. autofunction:: nvalchemiops.torch.fire2.fire2_step_extended
