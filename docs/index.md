# NVIDIA ALCHEMI Toolkit-Ops

GPU-accelerated primitives for atomistic simulations and atomic-scale modeling.

**[Get started →](userguide/index.md#quick-start)**

## Key Capabilities

- **[Neighbor lists](userguide/components/neighborlist.md):** naive, cell-list,
  and cluster-tile searches, with batching, selected-atom queries,
  differentiable distances and vectors, and matrix or COO outputs.
- **[Dispersion corrections](userguide/components/dispersion.md):** two-body
  DFT-D3(BJ) with environment-dependent coefficients, including real-space
  and particle-mesh Fourier implementations.
- **[Electrostatics](userguide/components/electrostatics.md):** direct Coulomb,
  Ewald, PME, and DSF calculations, with slab corrections and PyTorch multipole
  electrostatics.
- **[Molecular dynamics](userguide/components/dynamics.md):** velocity Verlet
  and Langevin integration, thermostats, and constant-pressure integration.
- **[Geometry optimization](userguide/components/dynamics.md#geometry-optimization):**
  FIRE, FIRE2, and L-BFGS methods for atomic relaxation and supported variable-cell
  calculations.
- **[Segment operations](userguide/components/segment_ops.md):** reductions and
  transformations of grouped data, including sums, means, dot products, and
  matrix-vector products.
- **Framework integration:** NVIDIA Warp kernels with PyTorch and JAX interfaces
  for supported operations.
- **[Repeated calculations](userguide/about/performance.md):**
  batching, reusable setup and storage, and compilation or CUDA Graph execution
  on supported routes.

Backend, differentiation, and compilation support varies by operation. See the
component guides for details.

## Explore the toolkit

- [User Guide](userguide/index.md) — installation and component guides.
- [Examples](examples/index.rst) — runnable examples of toolkit operations.
- [Benchmarks](benchmarks/index.md) — performance results and methodology.
- [API](modules/index.md) — function and class reference.
- [Changelog](changes.md) — release history.

```{toctree}
:maxdepth: 2
:hidden:

userguide/index
```

```{toctree}
:maxdepth: 2
:hidden:

examples/index
```

```{toctree}
:maxdepth: 2
:hidden:

benchmarks/index
```

```{toctree}
:maxdepth: 1
:hidden:

changes
```

```{toctree}
:maxdepth: 2
:hidden:

API <modules/index>
```
