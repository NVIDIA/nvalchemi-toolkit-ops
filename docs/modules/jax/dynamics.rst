:mod:`nvalchemiops.jax.lbfgs`: Geometry Optimization
====================================================

JAX bindings for the batched L-BFGS geometry optimizer.

.. automodule:: nvalchemiops.jax.lbfgs
    :no-members:
    :no-inherited-members:

.. tip::
   Every buffer is caller-owned, and JAX arrays are immutable, so these entry
   points take the buffers individually and return them as a flat tuple in the
   same order. Donate them with ``jax.jit(donate_argnums=...)`` so XLA can
   reuse the memory; see the module documentation above for the required
   initial contents and the donation and CUDA-graph contract.

Coordinate Relaxation
---------------------

.. autofunction:: nvalchemiops.jax.lbfgs.lbfgs_step_coord
.. autofunction:: nvalchemiops.jax.lbfgs.lbfgs_converged

Variable-Cell Relaxation
------------------------

Coordinates and cell are mapped into one packed coordinate vector, so the
two-loop recursion couples them automatically. Build ``ext_atom_ptr`` and
``ext_batch_idx`` with
:func:`~nvalchemiops.dynamics.utils.cell_filter.extend_atom_ptr` and
:func:`~nvalchemiops.batch_utils.atom_ptr_to_batch_idx`, which handle ragged
batches as well as uniform ones.

.. autofunction:: nvalchemiops.jax.lbfgs.lbfgs_set_reference_cell
.. autofunction:: nvalchemiops.jax.lbfgs.lbfgs_cell_kappa
.. autofunction:: nvalchemiops.jax.lbfgs.lbfgs_step_coord_cell
