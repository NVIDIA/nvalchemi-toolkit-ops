:mod:`nvalchemiops.jax.segment_mm`: Segmented Matrix Multiplication
===================================================================

.. currentmodule:: nvalchemiops.jax.segment_mm

The JAX interface provides segmented matrix multiplication with composable
forward and reverse derivatives and explicit whole-system sharding.

.. tip::
    See :ref:`segment_mm_userguide` for runtime lengths, explicit mixed-precision
    casts, and the local operand contract inside ``jax.shard_map``.

.. automodule:: nvalchemiops.jax.segment_mm
    :no-members:
    :no-inherited-members:

Public Operations
-----------------

.. autofunction:: segment_mm
