:mod:`nvalchemiops.neighbors`: Neighbor Lists
===================================================

.. automodule:: nvalchemiops.neighbors
    :no-members:
    :no-inherited-members:

Warp-Level Interface
--------------------

.. tip::
   This is the low-level Warp interface that operates on ``warp.array`` objects.
   For PyTorch tensor support, see :doc:`../torch/neighbors`.

High-Level Compatibility
------------------------

.. py:function:: nvalchemiops.neighbors.neighbor_list(*args, **kwargs)

   Compatibility shim for the pre-0.3 PyTorch neighbor-list entry point.
   New code should import :func:`nvalchemiops.torch.neighbors.neighbor_list`
   directly. If PyTorch is unavailable, accessing this name raises
   :class:`RuntimeError`.

Method Selection
----------------

.. autofunction:: nvalchemiops.neighbors.estimate_neighbor_list_costs
.. autofunction:: nvalchemiops.neighbors.suggest_neighbor_list_method

.. _warp-neighbor-pair-function-contract:

Pair Function API
-----------------

Neighbor kernels that accept ``pair_fn`` invoke a module-scope ``@wp.func``
for each accepted pair, evaluate the user-supplied pair potential, and
accumulate the returned energy and force into the kernel's output buffers.

**Signature**

.. code-block:: text

    pair_fn(
        vector_ij: wp.vec3,
        distance_ij: scalar,
        pair_params: wp.array2d,
        i: int32,
        j: int32,
    ) -> (energy: scalar, force: wp.vec3)

where ``scalar`` is the position dtype (``wp.float32`` or ``wp.float64``)
and ``wp.vec3`` is the matching vector width (``wp.vec3f`` or ``wp.vec3d``).

**Parameters**

``vector_ij``
    Separation vector ``positions[j] - positions[i]`` plus any periodic
    image shift, following the project separation-vector convention.

``distance_ij``
    Euclidean norm of ``vector_ij``.  Precomputed by the kernel so callbacks
    can reuse it without recomputing the square root.

``pair_params``
    Two-dimensional per-atom parameter table with the same scalar dtype as
    positions.  Conventionally laid out as ``(num_atoms, num_param_cols)``;
    rows are indexed by ``i`` and ``j``.

``i``, ``j``
    Atom indices into ``positions`` and ``pair_params`` for the pair being
    evaluated.

**Returns**

``(energy, force)``
    ``energy`` is the scalar pair energy.  ``force`` is the Cartesian force
    on atom ``i`` due to atom ``j``; the kernel accumulates ``+force`` to
    atom ``i`` and ``-force`` to atom ``j``.

**Example: Lorentz-Berthelot Lennard-Jones**

.. code-block:: python

    import warp as wp


    @wp.func
    def lj_pair_fn(
        vector_ij: wp.vec3f,
        distance_ij: wp.float32,
        pair_params: wp.array2d(dtype=wp.float32),
        i: int,
        j: int,
    ):
        eps = wp.sqrt(pair_params[i, 0] * pair_params[j, 0])
        sigma = 0.5 * (pair_params[i, 1] + pair_params[j, 1])
        inv_r = 1.0 / distance_ij
        sr = sigma * inv_r
        sr6 = sr * sr * sr * sr * sr * sr
        sr12 = sr6 * sr6
        energy = 4.0 * eps * (sr12 - sr6)
        force = -(24.0 * eps * inv_r * inv_r * (2.0 * sr12 - sr6)) * vector_ij
        return energy, force

Naive Algorithm
^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.neighbors.naive.naive_neighbor_matrix
.. autofunction:: nvalchemiops.neighbors.naive.naive_neighbor_matrix_pbc

Cell List Algorithm
^^^^^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.neighbors.cell_list.build_cell_list
.. autofunction:: nvalchemiops.neighbors.cell_list.query_cell_list

Batched Naive Algorithm
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.neighbors.naive.batch_naive_neighbor_matrix
.. autofunction:: nvalchemiops.neighbors.naive.batch_naive_neighbor_matrix_pbc

Batched Cell List Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.neighbors.cell_list.batch_build_cell_list
.. autofunction:: nvalchemiops.neighbors.cell_list.batch_query_cell_list

Cluster Tile Algorithm
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.neighbors.cluster_tile.build_cluster_tile_list
.. autofunction:: nvalchemiops.neighbors.cluster_tile.query_cluster_tile
.. autofunction:: nvalchemiops.neighbors.cluster_tile.query_cluster_tile_coo
.. autofunction:: nvalchemiops.neighbors.cluster_tile.estimate_max_tiles_per_group
.. autofunction:: nvalchemiops.neighbors.cluster_tile.estimate_batch_max_tiles_per_group

Batched Cluster Tile Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.neighbors.cluster_tile.batch_build_cluster_tile_list
.. autofunction:: nvalchemiops.neighbors.cluster_tile.batch_query_cluster_tile
.. autofunction:: nvalchemiops.neighbors.cluster_tile.batch_query_cluster_tile_coo
.. autofunction:: nvalchemiops.neighbors.cluster_tile.estimate_batch_cluster_tile_segments

Naive Dual Cutoff Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.neighbors.naive.naive_neighbor_matrix_dual_cutoff
.. autofunction:: nvalchemiops.neighbors.naive.naive_neighbor_matrix_pbc_dual_cutoff

Batched Naive Dual Cutoff Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.neighbors.naive.batch_naive_neighbor_matrix_dual_cutoff
.. autofunction:: nvalchemiops.neighbors.naive.batch_naive_neighbor_matrix_pbc_dual_cutoff

Rebuild Detection
^^^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.neighbors.rebuild.check_cell_list_rebuild
.. autofunction:: nvalchemiops.neighbors.rebuild.check_neighbor_list_rebuild
.. autofunction:: nvalchemiops.neighbors.rebuild.check_batch_cell_list_rebuild
.. autofunction:: nvalchemiops.neighbors.rebuild.check_batch_neighbor_list_rebuild

Exceptions
^^^^^^^^^^

.. autoexception:: nvalchemiops.neighbors.NeighborOverflowError
   :show-inheritance:

Utility Functions
^^^^^^^^^^^^^^^^^

.. autofunction:: nvalchemiops.neighbors.neighbor_utils.estimate_max_neighbors
.. autofunction:: nvalchemiops.neighbors.neighbor_utils.compute_naive_num_shifts
