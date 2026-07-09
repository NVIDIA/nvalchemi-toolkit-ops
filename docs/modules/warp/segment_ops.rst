:mod:`nvalchemiops.segment_ops`: Segment Operations
===================================================

.. currentmodule:: nvalchemiops.segment_ops

The segment-ops module provides the framework-agnostic, Warp-level
implementation of GPU-accelerated segmented reductions and per-segment
algebra. All operations act on ``warp.array`` objects and expect the segment
indices ``idx`` to be sorted in non-decreasing order.

.. tip::
    This is the low-level Warp interface that operates on ``warp.array``
    objects. For tensor bindings see :doc:`../torch/segment_ops` and
    :doc:`../jax/segment_ops`. For usage guidance and CUDA graph capture
    patterns, see :ref:`segment_ops_userguide`.

.. automodule:: nvalchemiops.segment_ops
    :no-members:
    :no-inherited-members:

Reductions
----------

Reduce the elements of each segment to a single per-segment result.

.. autofunction:: segmented_sum
.. autofunction:: segmented_component_sum
.. autofunction:: segmented_dot
.. autofunction:: segmented_inner_products
.. autofunction:: segmented_max_norm
.. autofunction:: segmented_max
.. autofunction:: segmented_min
.. autofunction:: segmented_mean
.. autofunction:: segmented_rms_norm
.. autofunction:: segmented_count

Broadcasts and Element-wise Algebra
-----------------------------------

Apply a per-segment value across every element of its segment.

.. autofunction:: segmented_mul
.. autofunction:: segmented_add
.. autofunction:: segmented_matvec
.. autofunction:: segmented_broadcast
.. autofunction:: segment_div

Fused Updates
-------------

In-place broadcast linear-combination updates (AXPY / AXPBY).

.. autofunction:: segmented_axpy
.. autofunction:: segmented_axpby
