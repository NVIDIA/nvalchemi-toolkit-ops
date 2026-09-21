FourierD3 Benchmarks
====================

Fixed-configuration throughput for the particle-mesh dispersion correction, alongside the
real-space ``dftd3`` at several cutoffs.

Running
-------

.. code-block:: bash

    python benchmarks/interactions/dispersion/benchmark_fourier_dftd3.py \
        --config benchmarks/interactions/dispersion/benchmark_fourier_config.yaml \
        --output-dir benchmarks/results

The matrix matches the DFT-D3 benchmark: two systems (CsCl, NH3) crossed with three scaling
modes (system size, constant workload, batch), one standardized
``fd3-{system}-{mode}-scaling.csv`` per pair. Methods are ``fourier_dftd3``,
``fourier_dftd3_setup`` and ``dftd3_cutoff_{r}`` for each configured real-space cutoff.

FourierD3 is periodic only and the runner is Torch-only; the JAX binding is covered by the
test suite rather than here.

Reading the results
-------------------

``time_eval_seconds`` is the canonical metric, as in the other kernel benchmarks: the
evaluation alone, which is the marginal cost when the coordination-number list is already
available --- the situation FourierD3 is designed for, where a machine-learned force field
has built that list anyway. ``time_neighbor_seconds`` and ``time_total_seconds`` are
supplementary columns for the list build and the sum of the two.

.. warning::
    The rows are **not matched in accuracy**. The mesh follows the paper's atom-count
    schedule and the ``dftd3`` rows use fixed cutoffs; no accuracy relationship between the
    two has been established for this system. A ``1/r^6`` interaction summed over three
    dimensions leaves a truncation error decaying as ``1/r^3``, so a 6 Angstrom dispersion
    cutoff is not a converged calculation --- but nor is it established what mesh would match
    a given cutoff here. Read each row as the cost of one stated configuration.

Systems come from the shared benchmark builders, rather than a uniform random cell. A random cell at a
chosen density places atoms at arbitrary separations, including overlaps, and gives
coordination numbers far outside anything the reference tables cover; a real lattice has the
near-neighbour structure and the density that the neighbour list and the coordination-number
pass actually cost.

Density therefore comes from the lattice rather than being chosen, and is reported with the
results. It still matters to the comparison: the real-space neighbour count grows as density
times cutoff cubed, while the mesh cost does not depend on density at all.
