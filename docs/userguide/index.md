<!-- markdownlint-disable MD014 -->

(userguide)=

# User Guide

Welcome to the ALCHEMI Toolkit-Ops user guide. Start with installation below,
then choose a component to learn its features, make your first call, and find
guidance for your workload.

## Quick Start

The quickest way to install ALCHEMI Toolkit-Ops:

```bash
$ pip install nvalchemi-toolkit-ops
```

To install ALCHEMI Toolkit-Ops with a deep-learning backend:

::::{tab-set}

:::{tab-item} PyTorch
:sync: torch

```bash
$ pip install 'nvalchemi-toolkit-ops[torch]'
```

:::

:::{tab-item} JAX
:sync: jax

```bash
$ pip install 'nvalchemi-toolkit-ops[jax]'
```

:::

::::

```{tip}
Running on **NVIDIA DGX Spark**? The Blackwell GPU requires CUDA 13 wheels for
PyTorch. See the [CUDA 13 installation notes](about/install.md#cuda-13-installation)
before proceeding.
```

## About

- [Install](about/install)
- [Introduction](about/intro)
- [Conventions](about/conventions)
- [Performance Guide](about/performance)
- [Migration Guide](about/migration)
- [FAQ](about/faq)

## Core Components

- [Neighbor Lists](components/neighborlist) — find nearby atom pairs and use
  their distances, vectors, or pair-function outputs.
- [Electrostatics](components/electrostatics) — calculate Coulomb interactions
  with direct, Ewald, PME, or DSF methods.
- [Dispersion Corrections](components/dispersion) — add two-body DFT-D3(BJ)
  dispersion with environment-dependent coefficients.
- [Dynamics](components/dynamics) — integrate molecular dynamics trajectories
  and relax structures with [geometry optimization](components/dynamics.md#geometry-optimization).
- [Segment Operations](components/segment_ops) — reduce and transform data
  grouped by system or segment.

## Developer Resources

- [Contributing](about/contributing)
- [Kernel Style Guide](about/kernel-style-guide)

```{toctree}
:caption: About
:maxdepth: 1
:hidden:

about/install
about/intro
about/conventions
about/performance
about/migration
about/faq

```

```{toctree}
:caption: Core Components
:maxdepth: 2
:hidden:

components/neighborlist
components/electrostatics
components/dispersion
components/dynamics
components/segment_ops
```

```{toctree}
:caption: Developer Resources
:maxdepth: 1
:hidden:

about/contributing
about/kernel-style-guide
```
