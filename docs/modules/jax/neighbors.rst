:mod:`nvalchemiops.jax.neighbors`: Neighbor Lists
==================================================

.. currentmodule:: nvalchemiops.jax.neighbors

The neighbors module provides JAX bindings for the GPU-accelerated
implementations of neighbor list algorithms.

.. tip::
    For the underlying framework-agnostic Warp kernels, see :doc:`../warp/neighbors`.

.. automodule:: nvalchemiops.jax.neighbors
    :no-members:
    :no-inherited-members:

High-Level Interface
--------------------

.. autofunction:: nvalchemiops.jax.neighbors.neighbor_list

Unbatched Algorithms
--------------------

Naive Algorithm
^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.jax.neighbors.naive_neighbor_list

Cell List Algorithm
^^^^^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.jax.neighbors.cell_list
.. autofunction:: nvalchemiops.jax.neighbors.cell_list.build_cell_list
.. autofunction:: nvalchemiops.jax.neighbors.cell_list.query_cell_list

Dual Cutoff Algorithm
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.jax.neighbors.naive_neighbor_list_dual_cutoff

Batched Algorithms
------------------

Batched Naive Algorithm
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.jax.neighbors.batch_naive_neighbor_list

Batched Cell List Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.jax.neighbors.batch_cell_list
.. autofunction:: nvalchemiops.jax.neighbors.batch_cell_list.batch_build_cell_list
.. autofunction:: nvalchemiops.jax.neighbors.batch_cell_list.batch_query_cell_list

Batched Dual Cutoff Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.jax.neighbors.batch_naive_neighbor_list_dual_cutoff

Rebuild Detection
-----------------

.. autofunction:: nvalchemiops.jax.neighbors.rebuild_detection.cell_list_needs_rebuild
.. autofunction:: nvalchemiops.jax.neighbors.rebuild_detection.neighbor_list_needs_rebuild
.. autofunction:: nvalchemiops.jax.neighbors.rebuild_detection.check_cell_list_rebuild_needed
.. autofunction:: nvalchemiops.jax.neighbors.rebuild_detection.check_neighbor_list_rebuild_needed

Utility Functions
-----------------

.. warning::

   The estimation and cell list building utilities are functional, however
   due to the dynamic nature of the two it is not possible to ``jax.jit``
   compile workflows that combine the two. Users expecting to ``jax.jit``
   end-to-end workflows should explicitly set ``max_total_cells`` to cell
   construction methods.

.. autofunction:: nvalchemiops.jax.neighbors.estimate_cell_list_sizes
.. autofunction:: nvalchemiops.jax.neighbors.estimate_batch_cell_list_sizes
.. autofunction:: nvalchemiops.jax.neighbors.neighbor_utils.allocate_cell_list
.. autofunction:: nvalchemiops.jax.neighbors.neighbor_utils.prepare_batch_idx_ptr
.. autofunction:: nvalchemiops.jax.neighbors.neighbor_utils.estimate_max_neighbors
.. autofunction:: nvalchemiops.jax.neighbors.neighbor_utils.get_neighbor_list_from_neighbor_matrix
.. autofunction:: nvalchemiops.jax.neighbors.neighbor_utils.compute_naive_num_shifts
