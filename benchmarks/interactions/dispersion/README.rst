Dispersion Benchmarks
=====================

Performance benchmarks for DFT-D3 dispersion correction kernels.

Dispersion PME (LJ-PME) Benchmarks
==================================

``benchmark_dispersion.py`` benchmarks the long-range dispersion (:math:`r^{-6}`)
PME — both the reciprocal-space term and the full real + reciprocal calculation —
with a size-scaling sweep and a batched sweep, writing CSV results. It mirrors
``benchmarks/interactions/electrostatics/benchmark_electrostatics.py``.

Examples::

    # PyTorch backend, custom sizes
    python -m benchmarks.interactions.dispersion.benchmark_dispersion \
        --backend torch --sizes 1000 4000 16000

    # JAX backend
    python -m benchmarks.interactions.dispersion.benchmark_dispersion \
        --backend jax --output dispersion_jax.csv
