<!-- markdownlint-disable MD025 MD026 -->

# Frequently Asked Questions

## General

### How do I get started?

For installation instructions, check the [installation guide](install). For a
quick working example, see the [User Guide](../index). For detailed API usage,
refer to the [API documentation](../../modules/index).

If your question is not answered here, please submit a Github [Issue][issues_].

### What hardware does this support?

ALCHEMI Toolkit-Ops runs on:

- CUDA-capable NVIDIA GPUs (Compute Capability 8.0+, i.e. A100 and newer)
- CPU execution via NVIDIA Warp (x86 and ARM, including Apple Silicon)

For best performance, we recommend CUDA 12+ with driver version 570.xx or newer.
See the [installation guide](install) for full prerequisites.

### I need a kernel that does not exist yet

If the existing API is missing functionality you need and you think it would
benefit the community, please start a discussion on Github [Issues][issues_].

## Neighbor Lists

### What is the difference between cell_list and naive algorithms?

The two algorithm families have different computational complexity:

- `cell_list()` uses spatial decomposition for O(N) scaling. It is optimized
  for large systems (roughly >2000 atoms) where the cutoff is small relative
  to the simulation box.
- `naive_neighbor_list()` computes all pairwise distances for O(N²) scaling.
  It has lower overhead and can be faster for smaller systems.

The crossover point depends on hardware, system density, and cutoff radius.
We recommend benchmarking both on your specific workload.

### How does this compare to ASE neighbor lists?

[ASE](https://wiki.fysik.dtu.dk/ase/) provides CPU-based neighbor list
implementations. ALCHEMI Toolkit-Ops differs in several ways:

- GPU acceleration via NVIDIA Warp kernels
- Native batch processing for multiple systems
- `torch.compile` compatibility for ML training loops
- Both dense (neighbor matrix) and sparse (COO) output formats

The acceleration is substantial, particularly for larger system sizes
where GPU utilization is amortized.

## Troubleshooting

### Using `torch.compile`

Select kernels support `torch.compile`; those that do will say so in their
docstrings. For `torch.compile` to work without graph breaks, you typically
need to pre-allocate output tensors. See the
[neighbor list documentation](../components/neighborlist) for details on
pre-allocation patterns.

We recommend reading the general
[`torch.compile` troubleshooting guide](https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html)
and the
[PhysicsNeMo performance tuning guide](https://docs.nvidia.com/physicsnemo/latest/user-guide/performance_docs/torch_compile_support.html#torch-compile).

If a kernel is expected to be `torch.compile` compatible but is not working,
please open a Github [Issue][issues_].

[issues_]: https://www.github.com/NVIDIA/nvalchemi-toolkit-ops/issues/new/choose
