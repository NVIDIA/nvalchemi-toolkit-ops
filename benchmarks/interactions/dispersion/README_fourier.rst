FourierD3 Benchmarks
====================

Measures the particle-mesh dispersion correction against the real-space ``dftd3``.

Running
-------

.. code-block:: bash

    python benchmarks/interactions/dispersion/benchmark_fourier_dftd3.py \
        --atom-counts 500 2000 8000 20000 \
        --output-dir benchmarks/results

Reading the results
-------------------

Timings include the neighbour-list build, since that is what a caller pays. A separate
``time_eval_seconds`` column reports the evaluation alone, which is the marginal cost when
the coordination-number list is already available --- the situation FourierD3 is designed
for, where a machine-learned force field has built that list anyway.

Compare **only at matched accuracy**. A ``1/r^6`` interaction summed over three dimensions
leaves a truncation error decaying as ``1/r^3``, so a 6 Angstrom dispersion cutoff is not a
converged calculation. It is reported here so that the distinction is visible, not as a
target to beat.

Systems are CsCl (B2) supercells from the shared benchmark builders, the same periodic
crystal the electrostatics suite uses, rather than a uniform random cell. A random cell at a
chosen density places atoms at arbitrary separations, including overlaps, and gives
coordination numbers far outside anything the reference tables cover; a real lattice has the
near-neighbour structure and the density that the neighbour list and the coordination-number
pass actually cost.

Density therefore comes from the lattice rather than being chosen, and is reported with the
results. It still matters to the comparison: the real-space neighbour count grows as density
times cutoff cubed, while the mesh cost does not depend on density at all.
