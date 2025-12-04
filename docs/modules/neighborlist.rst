:mod:`nvalchemiops.neighborlist`: Neighbor Lists
================================================

.. automodule:: nvalchemiops.neighborlist
    :no-members:
    :no-inherited-members:

High-Level Interface
--------------------
.. autofunction:: nvalchemiops.neighborlist.neighbor_list

Single System Algorithms
------------------------

.. autofunction:: nvalchemiops.neighborlist.naive_neighbor_list
.. autofunction:: nvalchemiops.neighborlist.cell_list

Cell List (Sub-)Algorithms
--------------------------

.. autofunction:: nvalchemiops.neighborlist.build_cell_list
.. autofunction:: nvalchemiops.neighborlist.query_cell_list

Batch Processing Algorithms
---------------------------

.. autofunction:: nvalchemiops.neighborlist.batch_naive_neighbor_list
.. autofunction:: nvalchemiops.neighborlist.batch_cell_list

Batch Cell List (Sub-)Algorithms
--------------------------------

.. autofunction:: nvalchemiops.neighborlist.batch_build_cell_list
.. autofunction:: nvalchemiops.neighborlist.batch_query_cell_list

Dual Cutoff Algorithms
----------------------

.. autofunction:: nvalchemiops.neighborlist.naive_neighbor_list_dual_cutoff
.. autofunction:: nvalchemiops.neighborlist.batch_naive_neighbor_list_dual_cutoff


Rebuild Detection
-----------------

.. autofunction:: nvalchemiops.neighborlist.cell_list_needs_rebuild
.. autofunction:: nvalchemiops.neighborlist.neighbor_list_needs_rebuild
.. autofunction:: nvalchemiops.neighborlist.check_cell_list_rebuild_needed
.. autofunction:: nvalchemiops.neighborlist.check_neighbor_list_rebuild_needed

Utility Functions
-----------------

.. autofunction:: nvalchemiops.neighborlist.estimate_cell_list_sizes
.. autofunction:: nvalchemiops.neighborlist.estimate_max_neighbors
.. autofunction:: nvalchemiops.neighborlist.estimate_batch_cell_list_sizes
.. autofunction:: nvalchemiops.neighborlist.allocate_cell_list
