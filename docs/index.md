# NVIDIA ALCHEMI Toolkit-Ops

GPU-accelerated computational kernels for molecular dynamics simulations and
atomic-scale modeling. Built on NVIDIA Warp, this toolkit provides high-performance
primitives for neighbor list construction, dispersion corrections, and other
operations critical to atomistic workflows.

## Key Capabilities

- O(N) cell list algorithms for neighbor list construction
- DFT-D3(BJ) dispersion corrections with environment-dependent C6 coefficients
- Ewald and particle mesh Ewald (PME) methods for electrostatic calculations
- Batch processing for multiple systems with heterogeneous parameters
- Native PyTorch tensor support with `torch.compile` compatibility
- Dense or sparse COO output formats for graph neural networks

## Who Is This For?

ML Researchers
: Integrate high-performance neighbor lists and energy corrections into graph
neural network pipelines.

Method Developers
: Access low-level Warp kernels to build custom atomistic workflows.

Computational Chemists
: Add GPU-accelerated dispersion corrections to DFT calculations.

[Get started →](userguide/about/install)

## User Guide

```{toctree}
:maxdepth: 2

userguide/index
```

## Examples

```{toctree}
:maxdepth: 2

examples/index
```

## Benchmarks

```{toctree}
:maxdepth: 2

benchmarks/index
```

## API

```{toctree}
:maxdepth: 2

API <modules/index>
```
