:mod:`nvalchemiops.interactions.dispersion`: Dispersion Corrections
===================================================================

.. automodule:: nvalchemiops.interactions.dispersion
    :no-members:
    :no-inherited-members:

Warp-Level Interface
--------------------

DFT-D3(BJ) Dispersion Corrections
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tip::
   This is the low-level Warp interface that operates on ``warp.array`` objects.
   For PyTorch tensor support, see :doc:`../torch/dispersion`.

The DFT-D3 implementation supports two neighbor representation formats:

- **Neighbor matrix** (dense): ``[num_atoms, max_neighbors]`` with padding
- **Neighbor list** (sparse CSR): Compressed sparse row format with ``idx_j`` and ``neighbor_ptr``

Both formats produce identical results and support all features including periodic
boundary conditions, batching, and smooth cutoff functions.

Non-Periodic Systems
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: nvalchemiops.interactions.dispersion._dftd3.dftd3_matrix
.. autofunction:: nvalchemiops.interactions.dispersion._dftd3.dftd3

Periodic Boundary Conditions (PBC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: nvalchemiops.interactions.dispersion._dftd3.dftd3_matrix_pbc
.. autofunction:: nvalchemiops.interactions.dispersion._dftd3.dftd3_pbc

FourierD3: Particle-Mesh DFT-D3
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tip::
   For PyTorch and JAX tensor support, see :doc:`../torch/dispersion` and
   :doc:`../jax/dispersion`.

FourierD3 evaluates the same DFT-D3(BJ) correction on a particle mesh, in
:math:`O(N \log N)` and with no real-space cutoff on the dispersion sum. The only
real-space cutoff that remains is the short coordination-number list.

Unlike the launchers above, these are **component** launchers rather than one end-to-end
call. Warp has no full-mesh FFT, so the two transforms belong to the calling framework and
the launchers are driven around them, in the order given by the table in the module
documentation.

.. automodule:: nvalchemiops.interactions.dispersion._fourier_dftd3
    :no-members:
    :no-inherited-members:

Real-Space Passes
~~~~~~~~~~~~~~~~~

.. autofunction:: nvalchemiops.interactions.dispersion._fourier_dftd3.fd3_coordination_numbers
.. autofunction:: nvalchemiops.interactions.dispersion._fourier_dftd3.fd3_coordination_numbers_matrix
.. autofunction:: nvalchemiops.interactions.dispersion._fourier_dftd3.fd3_coefficients

Mesh Spread
~~~~~~~~~~~

Pass 3, between the real-space passes and the forward transform. Unlike the general channel
spread in :mod:`nvalchemiops.math.spline`, this one applies no stencil-weight threshold, so
the interpolation is the exact B-spline that :func:`fd3_gather_and_force` differentiates.

.. autofunction:: nvalchemiops.interactions.dispersion._fourier_dftd3.fd3_spread

Reciprocal-Space Pass
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: nvalchemiops.interactions.dispersion._fourier_dftd3.fd3_kspace

Gradient Passes
~~~~~~~~~~~~~~~

.. autofunction:: nvalchemiops.interactions.dispersion._fourier_dftd3.fd3_gather_and_force
.. autofunction:: nvalchemiops.interactions.dispersion._fourier_dftd3.fd3_self_energy
.. autofunction:: nvalchemiops.interactions.dispersion._fourier_dftd3.fd3_cn_chain
.. autofunction:: nvalchemiops.interactions.dispersion._fourier_dftd3.fd3_cn_chain_matrix

Reference Tensor Decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Computed once on the host, not on the GPU. The rank it selects sets the number of mesh
channels and so the cost of every evaluation that follows.

.. automodule:: nvalchemiops.interactions.dispersion._c6_decomposition
    :no-members:
    :no-inherited-members:

.. autofunction:: nvalchemiops.interactions.dispersion._c6_decomposition.decompose_c6_reference
.. autofunction:: nvalchemiops.interactions.dispersion._c6_decomposition.extract_species_reference_cn
.. autofunction:: nvalchemiops.interactions.dispersion._c6_decomposition.clear_decomposition_cache

.. autoclass:: nvalchemiops.interactions.dispersion._c6_decomposition.C6Decomposition
    :members:
    :undoc-members:
