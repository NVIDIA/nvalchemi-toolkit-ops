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

L-BFGS Optimizer
----------------

Quasi-Newton relaxation with a ``maxstep`` trust region. Each step consumes one
force evaluation and reports progress through the ``status`` buffer. The step
length comes from a ``maxstep`` trust region, so no energy is read.

.. note::
   Every optimizer buffer is caller-owned: nothing here allocates or
   initializes state on your behalf. Zero the buffers, then set ``alpha_step``
   to ``1.0``, ``iteration`` to ``-1`` and ``status`` to ``LBFGS_NEED_EVAL``.
   The module documentation lists the required shapes and dtypes.

.. autofunction:: nvalchemiops.torch.lbfgs.lbfgs_step_coord
.. autofunction:: nvalchemiops.torch.lbfgs.lbfgs_step_extended

Variable-cell relaxation maps coordinates and cell into one packed coordinate
vector, so the two-loop recursion couples them automatically. Build
``ext_atom_ptr`` and ``ext_batch_idx`` with
:func:`~nvalchemiops.dynamics.utils.cell_filter.extend_atom_ptr` and
:func:`~nvalchemiops.batch_utils.atom_ptr_to_batch_idx`, which handle ragged
batches as well as uniform ones.

.. autofunction:: nvalchemiops.torch.lbfgs.lbfgs_set_reference_cell
.. autofunction:: nvalchemiops.torch.lbfgs.lbfgs_cell_kappa
.. autofunction:: nvalchemiops.torch.lbfgs.lbfgs_step_coord_cell
