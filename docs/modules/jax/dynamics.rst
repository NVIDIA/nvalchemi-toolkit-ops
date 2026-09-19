:mod:`nvalchemiops.jax.lbfgs`: Geometry Optimization
====================================================

JAX bindings for the batched L-BFGS geometry optimizer.

.. automodule:: nvalchemiops.jax.lbfgs
    :no-members:
    :no-inherited-members:

.. tip::
   JAX arrays are immutable, so these entry points take the state and *return*
   a new one rather than writing in place. Both state classes are registered
   pytrees, so a state passes through ``jax.jit`` as one argument and
   ``donate_argnums`` donates every field at once. See the module
   documentation above for the donation and CUDA-graph contract.

Coordinate Relaxation
---------------------

.. autofunction:: nvalchemiops.jax.lbfgs.lbfgs_prepare_state
.. autofunction:: nvalchemiops.jax.lbfgs.lbfgs_step_coord

Variable-Cell Relaxation
------------------------

Coordinates and cell are mapped into one packed coordinate vector, so the
two-loop recursion couples them automatically. Build ``ext_atom_ptr`` and
``ext_batch_idx`` with
:func:`~nvalchemiops.dynamics.utils.cell_filter.extend_atom_ptr` and
:func:`~nvalchemiops.batch_utils.atom_ptr_to_batch_idx`, which handle ragged
batches as well as uniform ones.

.. autofunction:: nvalchemiops.jax.lbfgs.lbfgs_prepare_cell_state
.. autofunction:: nvalchemiops.jax.lbfgs.lbfgs_set_reference_cell
.. autofunction:: nvalchemiops.jax.lbfgs.lbfgs_cell_kappa
.. autofunction:: nvalchemiops.jax.lbfgs.lbfgs_step_coord_cell
