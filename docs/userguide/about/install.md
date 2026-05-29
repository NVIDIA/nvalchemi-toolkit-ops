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

### CUDA 13

Blackwell GPUs require **CUDA 13**. The default PyPI `warp-lang` package ships
with CUDA 12 and needs a CUDA 13 variant. `torch>=2.11.0` and `jax[cuda13]`
publish CUDA 13 wheels on the default PyPI index for Linux x86_64 and aarch64.
See [CUDA 13 Installation](#cuda-13-installation) below for detailed steps.

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

### Backend Extras

ALCHEMI Toolkit-Ops provides optional extras for framework-specific bindings.
Install the extra matching your deep learning backend. The plain `torch` and
`jax` extras use CUDA 13 by default; use `torch-cu12` or `jax-cu12` when a
CUDA 12 environment is required.

::::{tab-set}

:::{tab-item} PyTorch
:sync: torch

```bash
$ pip install 'nvalchemi-toolkit-ops[torch]'
```

Verify the PyTorch bindings are available:

```bash
$ python -c "from nvalchemiops.torch import neighbors; print('PyTorch bindings available')"
```

:::

:::{tab-item} JAX
:sync: jax

```bash
$ pip install 'nvalchemi-toolkit-ops[jax]'
```

This installs JAX with CUDA 13 support. Verify the JAX bindings are available:

```bash
$ python -c "from nvalchemiops.jax import neighbors; print('JAX bindings available')"
```

:::

::::

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
# include torch backend with the default CUDA 13 version
$ uv sync --extra torch
# include jax backend with the default CUDA 13 version
$ uv sync --extra jax
# include both backends with the default CUDA 13 version
$ uv sync --extra torch --extra jax
# include both backends with CUDA 12
$ uv sync --extra torch-cu12 --extra jax-cu12
# equivalently, request CUDA 13 explicitly
$ uv sync --extra torch-cu13 --extra jax-cu13
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

### Adding `nvalchemi-toolkit-ops` as a dependency

<details>
    <summary><b>Nightly</b></summary>

```{warning}
Installing nightly versions without cloning the codebase is not recommended
for production settings! We recommend pinning this to a release tag or
commit hash.
```

```bash
$ uv add "nvalchemi-toolkit-ops @ git+https://www.github.com/NVIDIA/nvalchemi-toolkit-ops.git"
```

</details>

<details>
    <summary><b>Stable</b></summary>

```bash
$ uv add nvalchemi-toolkit-ops
```

</details>

(cuda-13-installation)=

## CUDA 13 Installation

Blackwell GPUs require packages/dependencies that are built for CUDA 13, which
includes `warp-lang`, `jax`, and `torch`. `torch>=2.11.0` and `jax[cuda13]`
provide CUDA 13 wheels from the default PyPI index on Linux x86_64 and aarch64.

### Warp

The PyPI `warp-lang` package ships with CUDA 12, and as of writing CUDA 13
wheels can be obtained from the [Warp GitHub Releases](https://github.com/NVIDIA/warp/releases)
page. Copy the URL of the appropriate `+cu13` wheel for your platform and pass it to
`pip install`. Select the wheel matching your architecture:

- **x86**: `manylinux_2_34_x86_64` variant
- **Arm** (e.g. DGX Spark): `manylinux_2_34_aarch64` variant

```bash
$ uv pip install https://github.com/NVIDIA/warp/releases/download/v1.12.1/warp_lang-1.12.1+cu13-py3-none-manylinux_2_34_aarch64.whl
```

```{tip}
Check the [Warp releases page](https://github.com/NVIDIA/warp/releases) for
newer versions. See the
[Warp installation guide](https://nvidia.github.io/warp/user_guide/installation.html#installing-from-github-releases)
for full details on installing specific versions from GitHub releases. The
`--force-reinstall` flag may be needed to overwrite a previous installation.
```

### PyTorch

Starting with version **2.11.0**, PyTorch publishes CUDA 13 (`cu130`) wheels on
the default PyPI index for Linux x86_64 and aarch64:

```bash
$ uv pip install torch==2.11.0
```

Use the PyTorch CUDA index directly when installing an explicit `+cu130` wheel:

```bash
$ uv pip install torch==2.11.0+cu130 \
    --extra-index-url https://download.pytorch.org/whl/cu130
```

### JAX

`jax[cuda13]` resolves from the default PyPI index on Linux x86_64 and
aarch64:

```bash
$ uv pip install 'jax[cuda13]'
```

### Full installation examples

#### Without cloning (recommended for most users)

::::{tab-set}

:::{tab-item} x86

```bash
$ uv venv --seed --python 3.12
$ WARP_CU13_WHEEL="https://github.com/NVIDIA/warp/releases/download/v1.12.1/\
warp_lang-1.12.1+cu13-py3-none-manylinux_2_34_x86_64.whl"
$ uv pip install nvalchemi-toolkit-ops \
    "$WARP_CU13_WHEEL" \
    torch==2.11.0 \
    'jax[cuda13]'
```

:::

:::{tab-item} Arm (e.g. DGX Spark)

```bash
$ uv venv --seed --python 3.12
$ WARP_CU13_WHEEL="https://github.com/NVIDIA/warp/releases/download/v1.12.1/\
warp_lang-1.12.1+cu13-py3-none-manylinux_2_34_aarch64.whl"
$ uv pip install nvalchemi-toolkit-ops \
    "$WARP_CU13_WHEEL" \
    torch==2.11.0 \
    'jax[cuda13]'
```

:::

::::

#### With cloning (for developers)

::::{tab-set}

:::{tab-item} x86

```bash
$ git clone git@github.com:NVIDIA/nvalchemi-toolkit-ops.git
$ cd nvalchemi-toolkit-ops
$ uv sync --group dev
$ WARP_CU13_WHEEL="https://github.com/NVIDIA/warp/releases/download/v1.12.1/\
warp_lang-1.12.1+cu13-py3-none-manylinux_2_34_x86_64.whl"
$ uv pip install \
    "$WARP_CU13_WHEEL" \
    torch==2.11.0 \
    'jax[cuda13]' \
    --force-reinstall
```

:::

:::{tab-item} Arm (e.g. DGX Spark)

```bash
$ git clone git@github.com:NVIDIA/nvalchemi-toolkit-ops.git
$ cd nvalchemi-toolkit-ops
$ uv sync --group dev
$ WARP_CU13_WHEEL="https://github.com/NVIDIA/warp/releases/download/v1.12.1/\
warp_lang-1.12.1+cu13-py3-none-manylinux_2_34_aarch64.whl"
$ uv pip install \
    "$WARP_CU13_WHEEL" \
    torch==2.11.0 \
    'jax[cuda13]' \
    --force-reinstall
```

:::

::::

```{note}
The `--force-reinstall` flag is needed in the developer flow because `uv sync`
will have already installed the default PyPI `warp-lang` wheel. `torch>=2.11.0`
and `jax[cuda13]` resolve from the default PyPI index on Linux x86_64 and
aarch64.

The `jax` and `torch` extras in `pyproject.toml` target CUDA 13 and are
equivalent to the explicit `jax-cu13` and `torch-cu13` extras. PyTorch CUDA 13
uses the `cu130` index; the CUDA 12 fallback uses the `cu126` index. Use
`jax-cu12`/`torch-cu12` when a CUDA 12 environment is required. Do not sync CUDA
12 and CUDA 13 extras into the same environment; JAX will error with
`ALREADY_EXISTS: PJRT_Api already exists for device type cuda` when both CUDA
plugins are present.
```

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
