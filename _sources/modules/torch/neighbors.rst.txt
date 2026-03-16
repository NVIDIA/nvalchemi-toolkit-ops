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

High-Level Interface
--------------------

.. autofunction:: nvalchemiops.torch.neighbors.neighbor_list

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

Batched Dual Cutoff Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.torch.neighbors.batch_naive_neighbor_list_dual_cutoff

Rebuild Detection
-----------------

.. autofunction:: nvalchemiops.torch.neighbors.rebuild_detection.cell_list_needs_rebuild
.. autofunction:: nvalchemiops.torch.neighbors.rebuild_detection.neighbor_list_needs_rebuild
.. autofunction:: nvalchemiops.torch.neighbors.rebuild_detection.check_cell_list_rebuild_needed
.. autofunction:: nvalchemiops.torch.neighbors.rebuild_detection.check_neighbor_list_rebuild_needed

Utility Functions
-----------------

.. autofunction:: nvalchemiops.torch.neighbors.estimate_cell_list_sizes
.. autofunction:: nvalchemiops.torch.neighbors.estimate_batch_cell_list_sizes
.. autofunction:: nvalchemiops.torch.neighbors.neighbor_utils.allocate_cell_list
.. autofunction:: nvalchemiops.torch.neighbors.neighbor_utils.prepare_batch_idx_ptr
