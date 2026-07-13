:mod:`nvalchemiops.torch.segment_ops`: Segment Operations
=========================================================

.. currentmodule:: nvalchemiops.torch.segment_ops

The segment-ops module provides differentiable PyTorch bindings for
GPU-accelerated segmented reductions and per-segment algebra.

.. tip::
    For usage guidance and CUDA graph capture patterns, see
    :ref:`segment_ops_userguide`.

.. automodule:: nvalchemiops.torch.segment_ops
    :no-members:
    :no-inherited-members:

Public Operations
-----------------

.. autofunction:: segmented_sum
.. autofunction:: segmented_dot
.. autofunction:: segmented_mul
.. autofunction:: segmented_mean
.. autofunction:: segmented_rms_norm
.. autofunction:: segmented_matvec
