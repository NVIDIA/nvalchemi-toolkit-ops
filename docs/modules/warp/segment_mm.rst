:mod:`nvalchemiops.segment_mm`: Segmented Matrix Multiplication
===============================================================

.. currentmodule:: nvalchemiops.segment_mm

The Warp interface multiplies packed segments by their per-segment matrices
using runtime device lengths and caller-provided output buffers.

.. tip::
    See :ref:`segment_mm_userguide` for the packed-row contract and framework
    usage. Automatic Warp differentiation is disabled; use the explicit
    gradient functions below or the :doc:`../torch/segment_mm` and
    :doc:`../jax/segment_mm` bindings.

.. automodule:: nvalchemiops.segment_mm
    :no-members:
    :no-inherited-members:

Public Operations
-----------------

.. autofunction:: segment_mm
.. autofunction:: segment_mm_backward
.. autofunction:: segment_mm_backward_b
