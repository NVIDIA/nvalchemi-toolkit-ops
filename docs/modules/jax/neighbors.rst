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

Prepared Execution
------------------

Prepared state packages fixed configuration and reusable storage in an
immutable PyTree threaded between calls. See the
:ref:`prepared neighbor-list guide <prepared_neighbor_lists>` for route
selection, lifecycle, and performance guidance.

Set ``fixed_cell=True`` during preparation to reuse cell-dependent search
geometry for an unchanged explicit cell. This is a caller promise: execution
does not compare cell values, and changing them requires a new state. The
read-only ``state.fixed_cell`` records the mode. Thread successor states as
usual; positions may still change, and pair geometry retains its existing
position and cell gradients.

.. autoclass:: nvalchemiops.jax.neighbors.NeighborListState()
   :members:
   :exclude-members: tree_flatten, tree_unflatten

.. autofunction:: nvalchemiops.jax.neighbors.prepare_neighbor_list
.. autofunction:: nvalchemiops.jax.neighbors.check_neighbor_list_state

Method Selection
----------------

.. autofunction:: nvalchemiops.jax.neighbors.estimate_neighbor_list_costs
.. autofunction:: nvalchemiops.jax.neighbors.suggest_neighbor_list_method

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

Cluster Tile Algorithm
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.jax.neighbors.cluster_tile_neighbor_list
.. autofunction:: nvalchemiops.jax.neighbors.cluster_tile.build_cluster_tile_list
.. autofunction:: nvalchemiops.jax.neighbors.cluster_tile.query_cluster_tile
.. autofunction:: nvalchemiops.jax.neighbors.cluster_tile.query_cluster_tile_coo
.. autofunction:: nvalchemiops.jax.neighbors.estimate_cluster_tile_list_sizes

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

Batched Cluster Tile Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.jax.neighbors.batch_cluster_tile_neighbor_list
.. autofunction:: nvalchemiops.jax.neighbors.batch_cluster_tile.batch_build_cluster_tile_list
.. autofunction:: nvalchemiops.jax.neighbors.batch_cluster_tile.batch_query_cluster_tile
.. autofunction:: nvalchemiops.jax.neighbors.batch_cluster_tile.batch_query_cluster_tile_coo
.. autofunction:: nvalchemiops.jax.neighbors.estimate_batch_max_tiles_per_group
.. autofunction:: nvalchemiops.jax.neighbors.estimate_batch_cluster_tile_list_sizes
.. autofunction:: nvalchemiops.jax.neighbors.estimate_batch_cluster_tile_segments
.. autofunction:: nvalchemiops.jax.neighbors.batch_cluster_tile.allocate_batch_cluster_tile_list

Batched Dual Cutoff Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.jax.neighbors.batch_naive_neighbor_list_dual_cutoff

Rebuild Detection
-----------------

.. autofunction:: nvalchemiops.jax.neighbors.rebuild_detection.cell_list_needs_rebuild
.. autofunction:: nvalchemiops.jax.neighbors.rebuild_detection.neighbor_list_needs_rebuild
.. autofunction:: nvalchemiops.jax.neighbors.rebuild_detection.check_cell_list_rebuild_needed
.. autofunction:: nvalchemiops.jax.neighbors.rebuild_detection.check_neighbor_list_rebuild_needed
.. autofunction:: nvalchemiops.jax.neighbors.rebuild_detection.batch_cell_list_needs_rebuild
.. autofunction:: nvalchemiops.jax.neighbors.rebuild_detection.batch_neighbor_list_needs_rebuild

Exceptions
----------

.. autoexception:: nvalchemiops.jax.neighbors.NeighborOverflowError
   :no-index:
   :show-inheritance:

.. autoexception:: nvalchemiops.jax.neighbors.TileBufferOverflow
   :no-index:
   :show-inheritance:

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
