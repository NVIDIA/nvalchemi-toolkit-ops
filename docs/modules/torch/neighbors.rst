:mod:`nvalchemiops.torch.neighbors`: Neighbor Lists
===================================================

.. currentmodule:: nvalchemiops.torch.neighbors

The neighbors module provides PyTorch-bindings for the GPU accelerated
implementations of neighbor list algorithms.

.. tip::
    For the underlying framework-agnostic Warp kernels, see :doc:`../warp/neighbors`.

.. automodule:: nvalchemiops.torch.neighbors
    :no-members:
    :no-inherited-members:


Pair Functions
--------------

Torch neighbor APIs that expose ``pair_fn`` use the same Warp callback API
as the low-level kernels.  See :ref:`the Warp neighbor-list pair function
API <warp-neighbor-pair-function-contract>` for the callback signature,
force convention, and Lennard-Jones example.

High-Level Interface
--------------------

.. autofunction:: nvalchemiops.torch.neighbors.neighbor_list

Method Selection
^^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.torch.neighbors.estimate_neighbor_list_costs
.. autofunction:: nvalchemiops.torch.neighbors.suggest_neighbor_list_method

Unbatched Algorithms
--------------------

Naive Algorithm
^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.torch.neighbors.naive_neighbor_list

Cell List Algorithm
^^^^^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.torch.neighbors.cell_list
.. autofunction:: nvalchemiops.torch.neighbors.cell_list.build_cell_list
.. autofunction:: nvalchemiops.torch.neighbors.cell_list.query_cell_list

Cluster Tile Algorithm
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.torch.neighbors.cluster_tile_neighbor_list
.. autofunction:: nvalchemiops.torch.neighbors.cluster_tile.build_cluster_tile_list
.. autofunction:: nvalchemiops.torch.neighbors.cluster_tile.query_cluster_tile
.. autofunction:: nvalchemiops.torch.neighbors.cluster_tile.query_cluster_tile_coo
.. autofunction:: nvalchemiops.torch.neighbors.cluster_tile.estimate_cluster_tile_list_sizes
.. autofunction:: nvalchemiops.torch.neighbors.cluster_tile.allocate_cluster_tile_list

Dual Cutoff Algorithm
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.torch.neighbors.naive_neighbor_list_dual_cutoff

Batched Algorithms
------------------

Batched Naive Algorithm
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.torch.neighbors.batch_naive_neighbor_list

Batched Cell List Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.torch.neighbors.batch_cell_list
.. autofunction:: nvalchemiops.torch.neighbors.batch_cell_list.batch_build_cell_list
.. autofunction:: nvalchemiops.torch.neighbors.batch_cell_list.batch_query_cell_list

Batched Cluster Tile Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.torch.neighbors.batch_cluster_tile_neighbor_list
.. autofunction:: nvalchemiops.torch.neighbors.batch_cluster_tile.batch_build_cluster_tile_list
.. autofunction:: nvalchemiops.torch.neighbors.batch_cluster_tile.batch_query_cluster_tile
.. autofunction:: nvalchemiops.torch.neighbors.batch_cluster_tile.batch_query_cluster_tile_coo
.. autofunction:: nvalchemiops.torch.neighbors.batch_cluster_tile.estimate_batch_cluster_tile_list_sizes
.. autofunction:: nvalchemiops.torch.neighbors.batch_cluster_tile.estimate_batch_cluster_tile_segments
.. autofunction:: nvalchemiops.torch.neighbors.batch_cluster_tile.allocate_batch_cluster_tile_list

Batched Dual Cutoff Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.torch.neighbors.batch_naive_neighbor_list_dual_cutoff

Rebuild Detection
-----------------

.. autofunction:: nvalchemiops.torch.neighbors.rebuild_detection.cell_list_needs_rebuild
.. autofunction:: nvalchemiops.torch.neighbors.rebuild_detection.neighbor_list_needs_rebuild
.. autofunction:: nvalchemiops.torch.neighbors.rebuild_detection.batch_cell_list_needs_rebuild
.. autofunction:: nvalchemiops.torch.neighbors.rebuild_detection.batch_neighbor_list_needs_rebuild

Utility Functions
-----------------

.. autofunction:: nvalchemiops.torch.neighbors.estimate_cell_list_sizes
.. autofunction:: nvalchemiops.torch.neighbors.estimate_batch_cell_list_sizes
.. autofunction:: nvalchemiops.torch.neighbors.neighbor_utils.allocate_cell_list
.. autofunction:: nvalchemiops.torch.neighbors.neighbor_utils.prepare_batch_idx_ptr
