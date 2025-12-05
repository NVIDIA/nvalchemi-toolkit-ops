<!-- markdownlint-disable MD025 MD033 MD014 -->

(install_guide)=

# Installation Guide

As ALCHEMI Toolkit-Ops is intended to be a low footprint library of lower level,
high-performance kernels, the number of external dependencies is deliberately
kept low as to keep the package lightweight and modular.

## Prerequisites

For the most part, ALCHEMI Toolkit-Ops shares the minimum prerequisites with
[NVIDIA Warp](https://nvidia.github.io/warp/installation.html): the kernels
**can** be run on a variety of CPU platforms (x86, ARM including Apple Silicon),
with best performance provided on CUDA-capable NVIDIA GPUs running on the following
operating systems:

- Linux-based distributions with recent CUDA versions, drivers, and firmware,
and Linux kernels
- Windows, through WSL2
- macOS (Apple Silicon only)

When running on CUDA-capable NVIDIA GPUs, we recommend:

- CUDA Toolkit: 12 or higher
- GPU Compute Capability: 8.0 or higher (A100 and newer)
- Driver: NVIDIA driver 570.xx.xx or newer

## Installation Methods

### From PyPI

The most straightforward way to install ALCHEMI Toolkit-Ops is via PyPI:

```bash
$ pip install nvalchemi-toolkit-ops
```

```{note}
We recommend using `uv` for virtual environment, package management, and
dependency resolution. `uv` can be obtained through their installation
page found [here](https://docs.astral.sh/uv/getting-started/installation/).
```

### From Github Source

This approach is useful for obtain nightly builds by installing directly
from the source repository:

```bash
$ pip install git+https://www.github.com/NVIDIA/nvalchemi-toolkit-ops.git
```

### Installation via `uv`

Maintainers generally use `uv`, and is the most reliable (and fastest) way
to spin up a virtual environment to use ALCHEMI Toolkit-Ops. Assuming `uv`
is in your path, here are a few ways to get started:

<details>
    <summary><b>Stable</b>, without cloning</summary>

This method is recommended for production use-cases, and when using
ALCHEMI Toolkit-Ops as a dependency for your project. The Python version
can be substituted for any other version supported by ALCHEMI Toolkit-Ops.

```bash
$ uv venv --seed --python 3.12
$ uv pip install nvalchemi-toolkit-ops
```

</details>

<details>
    <summary><b>Nightly</b>, with cloning</summary>

This method is recommended for local development and testing.

```bash
$ git clone git@github.com/NVIDIA/nvalchemi-toolkit-ops.git
$ cd nvalchemi-toolkit-ops
$ uv sync
```

</details>

<details>
    <summary><b>Nightly</b>, without cloning</summary>

```{warning}
Installing nightly versions without cloning the codebase is not recommended
for production settings!
```

```bash
$ uv venv --seed --python 3.12
$ uv pip install git+https://www.github.com/NVIDIA/nvalchemi-toolkit-ops.git
```

</details>

Includes Sphinx and related tools for building documentation.

## Installation with Conda & Mamba

The installation procedure should be similar to other environment management tools
when using either `conda` or `mamba` managers; assuming installation from a fresh
environment:

```bash
# create a new environment named nvalchemi if needed
mamba create -n nvalchemi python=3.12 pip
mamba activate nvalchemi
pip install nvalchemi-toolkit-ops
```

## Docker Usage

Given the modular nature of `nvalchemiops`, we do not provide a base Docker image.
Instead, the snippet below is a suggested base image that follows the requirements
of NVIDIA `warp-lang`, and installs `uv` for Python management:

```docker
# uses a lightweight Ubuntu-based image with CUDA 13
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

# grab package updates and other system dependencies here
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*
# copy uv for venv management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN uv venv --seed --python 3.12 /opt/venv
# this sets the default virtual environment to use
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# install ALCHEMI Toolkit-Ops
RUN uv pip install nvalchemi-toolkit-ops
```

This image can potentially be used as a basis for your application and/or development
environment. Your host system should have the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html)
installed, and at runtime, include `--gpus all` as a flag to container run statements to
ensure that GPUs are exposed to the container.

## Next Steps

You should now have a local installation of `nvalchemiops` ready for whatever
your use case might be! To verify, you can always run:

```bash
$ python -c "import nvalchemiops; print(nvalchemiops.__version__)"
```

If that doesn't resolve, make sure you've activated your virtual environment. Once
you've verified your installation, you can:

1. **Explore examples & benchmarks**: Check the `examples/` directory for tutorials
2. **Read Documentation**: Browse the user and API documentation to determine how to
integrate ALCHEMI Toolkit-Ops into your application.
