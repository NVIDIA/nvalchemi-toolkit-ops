# Neighbor List Benchmarks

Performance benchmarks for neighbor list algorithms in ALCHEMI Toolkit-Ops.
Results show scaling behaviour across system sizes for multiple cutoff radii
and algorithms.

```{warning}
These results are intended to be indicative _only_: your actual performance may
vary depending on the atomic system topology, software and hardware configuration
and we encourage users to benchmark on their own systems of interest.
```

## How to Read These Charts

Time Scaling
: Mean execution time (µs/atom) vs. system size. Lower is better. The naive
  strategies enumerate all pairs and have $O(N^2)$ work. Cell-list strategies
  are expected to approach $O(N)$ at fixed density and cutoff; finite-size
  timings also include launch, build/query, and output costs.

Throughput
: Atoms processed per second (plotted as 10⁶ atoms/s). Higher is better.
  This metric helps compare efficiency across different system sizes.

Memory
: Peak memory reported by the Torch CUDA allocator vs. system size. Units
  switch between MB and GB automatically on the y-axis. JAX memory is not
  measured by this suite.

Batch Scaling
: Compares the concrete batch APIs. A batch size of one resolves to the
  corresponding single-system API and is therefore kept out of these curves.
  The smaller fixed systems contain 250 atoms for CsCl and 256 atoms for NH₃,
  so their first plotted batch points contain 500 and 512 total atoms,
  respectively. The system-size panels include the 128-atom cases.

## Performance Results

<!-- markdownlint-disable MD013 -->

```{raw} html
<div id="nl-cutoff-switch" role="group" aria-label="Neighbor-list cutoff selection">
  <span class="nl-cutoff-switch__label">Cutoffs</span>
  <span class="nl-cutoff-switch__controls">
    <label class="nl-cutoff-toggle">
      <input type="checkbox" data-cutoff="6A" role="switch" aria-label="Show 6 Angstrom cutoff">
      <span class="nl-cutoff-toggle__control" aria-hidden="true"></span>
      <span class="nl-cutoff-toggle__value">6 Å</span>
    </label>
    <label class="nl-cutoff-toggle">
      <input type="checkbox" data-cutoff="15A" role="switch" aria-label="Show 15 Angstrom cutoff" checked>
      <span class="nl-cutoff-toggle__control" aria-hidden="true"></span>
      <span class="nl-cutoff-toggle__value">15 Å</span>
    </label>
    <label class="nl-cutoff-toggle">
      <input type="checkbox" data-cutoff="25A" role="switch" aria-label="Show 25 Angstrom cutoff">
      <span class="nl-cutoff-toggle__control" aria-hidden="true"></span>
      <span class="nl-cutoff-toggle__value">25 Å</span>
    </label>
  </span>
</div>
<script>
(function () {
  const root = document.getElementById("nl-cutoff-switch");
  if (!root) {
    return;
  }
  const cutoffOrder = ["6A", "15A", "25A"];
  const inputs = Array.from(root.querySelectorAll("input[data-cutoff]"));

  function selectedCutoffs() {
    return cutoffOrder.filter(function (cutoff) {
      const input = root.querySelector('input[data-cutoff="' + cutoff + '"]');
      return input && input.checked;
    });
  }

  function applySelection(plot) {
    const selected = selectedCutoffs();
    cutoffOrder.forEach(function (cutoff) {
      const visible = selected.indexOf(cutoff) !== -1;
      const groups = plot.querySelectorAll('[data-nl-cutoff="' + cutoff + '"]');
      groups.forEach(function (group) {
        group.style.opacity = visible ? "1" : "0";
      });
    });
  }

  function updatePlots() {
    document.querySelectorAll("#performance-results svg.nl-cutoff-plot")
      .forEach(function (plot) {
        applySelection(plot);
      });
  }

  function initialize() {
    updatePlots();
  }

  inputs.forEach(function (input) {
    input.addEventListener("change", function () {
      if (!input.checked && selectedCutoffs().length === 0) {
        input.checked = true;
        return;
      }
      updatePlots();
    });
  });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize, {once: true});
  } else {
    initialize();
  }
})();
</script>
```

<!-- markdownlint-enable MD013 -->

::::{tab-set}

:::{tab-item} Torch
:selected:

`````{tab-set}

````{tab-item} CsCl
:selected:

```{eval-rst}
.. tab-set::

    .. tab-item:: System Size Scaling

        .. figure:: _static/nl-cscl-system-size-scaling-time.png
           :width: 90%
           :align: center

           Mean execution time vs. system size for naive and cell list algorithms.

        .. figure:: _static/nl-cscl-system-size-scaling-throughput.png
           :width: 90%
           :align: center

           Throughput (10⁶ atoms/s) vs. system size.

        .. figure:: _static/nl-cscl-system-size-scaling-memory.png
           :width: 90%
           :align: center

           Peak GPU memory vs. system size.

    .. tab-item:: Constant Workload

        .. figure:: _static/nl-cscl-constant-workload-scaling-time.png
           :width: 90%
           :align: center

           Execution time near the configured total-atom target, varying batch size.

        .. figure:: _static/nl-cscl-constant-workload-scaling-throughput.png
           :width: 90%
           :align: center

           Throughput near the configured total-atom target.

        .. figure:: _static/nl-cscl-constant-workload-scaling-memory.png
           :width: 90%
           :align: center

           Peak GPU memory near the configured total-atom target.

    .. tab-item:: Batch Scaling

        .. figure:: _static/nl-cscl-batch-scaling-time.png
           :width: 90%
           :align: center

           Execution time vs. batch size (fixed atoms per system).

        .. figure:: _static/nl-cscl-batch-scaling-throughput.png
           :width: 90%
           :align: center

           Throughput vs. batch size.

        .. figure:: _static/nl-cscl-batch-scaling-memory.png
           :width: 90%
           :align: center

           Peak GPU memory vs. batch size.
```

````

````{tab-item} NH₃

```{eval-rst}
.. tab-set::

    .. tab-item:: System Size Scaling

        .. figure:: _static/nl-nh3-system-size-scaling-time.png
           :width: 90%
           :align: center

           Mean execution time vs. system size for naive and cell list algorithms (NH₃).

        .. figure:: _static/nl-nh3-system-size-scaling-throughput.png
           :width: 90%
           :align: center

           Throughput (10⁶ atoms/s) vs. system size (NH₃).

        .. figure:: _static/nl-nh3-system-size-scaling-memory.png
           :width: 90%
           :align: center

           Peak GPU memory vs. system size (NH₃).

    .. tab-item:: Constant Workload

        .. figure:: _static/nl-nh3-constant-workload-scaling-time.png
           :width: 90%
           :align: center

           Execution time near the configured total-atom target (NH₃).

        .. figure:: _static/nl-nh3-constant-workload-scaling-throughput.png
           :width: 90%
           :align: center

           Throughput near the configured total-atom target (NH₃).

        .. figure:: _static/nl-nh3-constant-workload-scaling-memory.png
           :width: 90%
           :align: center

           Peak GPU memory near the configured total-atom target (NH₃).

    .. tab-item:: Batch Scaling

        .. figure:: _static/nl-nh3-batch-scaling-time.png
           :width: 90%
           :align: center

           Execution time vs. batch size (NH₃).

        .. figure:: _static/nl-nh3-batch-scaling-throughput.png
           :width: 90%
           :align: center

           Throughput vs. batch size (NH₃).

        .. figure:: _static/nl-nh3-batch-scaling-memory.png
           :width: 90%
           :align: center

           Peak GPU memory vs. batch size (NH₃).
```

````

`````

:::

:::{tab-item} JAX

The default JAX reportable pass covers ``naive_scalar``, ``naive_tile``,
``cell_list_atom_centric``, and ``cluster_tile`` (plus their batched forms).
Pair-centric cell-list timing remains available as an explicit coverage-only
request. The cluster-tile path requires CUDA, ``float32`` positions, and fully
periodic cells; unsupported inputs are retained as explicit policy rows.

`````{tab-set}

````{tab-item} CsCl
:selected:

```{eval-rst}
.. tab-set::

    .. tab-item:: System Size Scaling

        .. figure:: _static/nl-cscl-system-size-scaling-jax-time.png
           :width: 90%
           :align: center

           Mean execution time vs. system size (JAX).

        .. figure:: _static/nl-cscl-system-size-scaling-jax-throughput.png
           :width: 90%
           :align: center

           Throughput (10⁶ atoms/s) vs. system size (JAX).

    .. tab-item:: Constant Workload

        .. figure:: _static/nl-cscl-constant-workload-scaling-jax-time.png
           :width: 90%
           :align: center

           Execution time near the configured total-atom target (JAX).

        .. figure:: _static/nl-cscl-constant-workload-scaling-jax-throughput.png
           :width: 90%
           :align: center

           Throughput near the configured total-atom target (JAX).

    .. tab-item:: Batch Scaling

        .. figure:: _static/nl-cscl-batch-scaling-jax-time.png
           :width: 90%
           :align: center

           Execution time vs. batch size (JAX).

        .. figure:: _static/nl-cscl-batch-scaling-jax-throughput.png
           :width: 90%
           :align: center

           Throughput vs. batch size (JAX).

```

```{note}
JAX memory plots are omitted. The suite does not measure JAX memory:
XLA's allocator pool and buffer reuse make per-call allocation deltas
unreliable. The Torch panels report the Torch allocator only; they are not a
proxy for JAX memory because the framework wrappers, buffer lifetimes, and
allocators differ.
```

````

````{tab-item} NH₃

```{eval-rst}
.. tab-set::

    .. tab-item:: System Size Scaling

        .. figure:: _static/nl-nh3-system-size-scaling-jax-time.png
           :width: 90%
           :align: center

           Mean execution time vs. system size (JAX, NH₃).

        .. figure:: _static/nl-nh3-system-size-scaling-jax-throughput.png
           :width: 90%
           :align: center

           Throughput (10⁶ atoms/s) vs. system size (JAX, NH₃).

    .. tab-item:: Constant Workload

        .. figure:: _static/nl-nh3-constant-workload-scaling-jax-time.png
           :width: 90%
           :align: center

           Execution time near the configured total-atom target (JAX, NH₃).

        .. figure:: _static/nl-nh3-constant-workload-scaling-jax-throughput.png
           :width: 90%
           :align: center

           Throughput near the configured total-atom target (JAX, NH₃).

    .. tab-item:: Batch Scaling

        .. figure:: _static/nl-nh3-batch-scaling-jax-time.png
           :width: 90%
           :align: center

           Execution time vs. batch size (JAX, NH₃).

        .. figure:: _static/nl-nh3-batch-scaling-jax-throughput.png
           :width: 90%
           :align: center

           Throughput vs. batch size (JAX, NH₃).

```

```{note}
JAX memory plots are omitted. The suite does not measure JAX memory:
XLA's allocator pool and buffer reuse make per-call allocation deltas
unreliable. The Torch panels report the Torch allocator only; they are not a
proxy for JAX memory because the framework wrappers, buffer lifetimes, and
allocators differ.
```

````

`````

:::

:::{tab-item} Backend Comparison

These overlays use only matched, comparison-eligible Torch/JAX rows. The cutoff
switches above control these panels as well as the individual backend panels.
JAX allocator memory is not shown.

```{note}
In the two Constant Workload comparison tabs, selecting only 25 Å leaves the
chart empty. Those JAX measurements had to wait after every timed call to fit
their output storage, so they are not timed the same way as Torch and are
deliberately excluded from the overlay. The individual Torch and JAX tabs still
show the 25 Å measurements.
```

`````{tab-set}

````{tab-item} CsCl
:selected:

```{eval-rst}
.. tab-set::

    .. tab-item:: System Size Scaling

        .. figure:: _static/nl-backend-cscl-system-size-scaling-time.png
           :width: 90%
           :align: center

           Torch vs. JAX execution time comparison.

        .. figure:: _static/nl-backend-cscl-system-size-scaling-throughput.png
           :width: 90%
           :align: center

           Torch vs. JAX throughput comparison.

    .. tab-item:: Constant Workload

        .. figure:: _static/nl-backend-cscl-constant-workload-scaling-time.png
           :width: 90%
           :align: center

           Torch vs. JAX execution time at constant workload.

        .. figure:: _static/nl-backend-cscl-constant-workload-scaling-throughput.png
           :width: 90%
           :align: center

           Torch vs. JAX throughput at constant workload.

    .. tab-item:: Batch Scaling

        .. figure:: _static/nl-backend-cscl-batch-scaling-time.png
           :width: 90%
           :align: center

           Torch vs. JAX execution time vs. total atoms.

        .. figure:: _static/nl-backend-cscl-batch-scaling-throughput.png
           :width: 90%
           :align: center

           Torch vs. JAX throughput vs. total atoms.
```

````

````{tab-item} NH₃

```{eval-rst}
.. tab-set::

    .. tab-item:: System Size Scaling

        .. figure:: _static/nl-backend-nh3-system-size-scaling-time.png
           :width: 90%
           :align: center

           Torch vs. JAX execution time comparison (NH₃).

        .. figure:: _static/nl-backend-nh3-system-size-scaling-throughput.png
           :width: 90%
           :align: center

           Torch vs. JAX throughput comparison (NH₃).

    .. tab-item:: Constant Workload

        .. figure:: _static/nl-backend-nh3-constant-workload-scaling-time.png
           :width: 90%
           :align: center

           Torch vs. JAX execution time at constant workload (NH₃).

        .. figure:: _static/nl-backend-nh3-constant-workload-scaling-throughput.png
           :width: 90%
           :align: center

           Torch vs. JAX throughput at constant workload (NH₃).

    .. tab-item:: Batch Scaling

        .. figure:: _static/nl-backend-nh3-batch-scaling-time.png
           :width: 90%
           :align: center

           Torch vs. JAX execution time vs. total atoms (NH₃).

        .. figure:: _static/nl-backend-nh3-batch-scaling-throughput.png
           :width: 90%
           :align: center

           Torch vs. JAX throughput vs. total atoms (NH₃).
```

````

`````

::::{dropdown} Comparison contract
Torch/JAX comparison eligibility is limited to ``naive_scalar`` and
``batch_naive_scalar``. Both backends must succeed at the same x-value, and
JAX serial-fallback rows are excluded.

The other strategies remain in the individual backend panels. They are not
overlaid because the current public APIs use different execution contracts:
JAX naive-tile runs through an eager public call, the cell-list wrappers can
choose different cell grids or query paths, pair-centric has backend-specific
launch guards, and cluster-tile has different traceability between single and
batched APIs.

JAX normally dispatches timed calls back-to-back and blocks on the last result.
If output storage cannot fit that queue, it blocks after each call and records
``timing_method=jax_wall_block_each``; no atom-count or hardware-specific case
limit is applied.

This is a framework-facing steady-state comparison, not a raw-kernel
microbenchmark. Each CSV row records its caller-, API-, or JIT-managed
``allocation_boundary`` and its ``timing_scope`` so differing execution
contracts remain explicit.
::::

:::

::::

## Method Variants

The NL suite treats each user-facing neighbor-list strategy as a separate
method. The naive family contains `naive_scalar` and `naive_tile`; the cell-list
family contains `cell_list_atom_centric` and `cell_list_pair_centric`.
`cluster_tile` is the CUDA/float32, fully periodic path that Morton-sorts atoms
into 32-atom clusters and evaluates candidate cluster pairs cooperatively.
Batched inputs use the matching batch API where needed, but the CSV method
column keeps the concrete strategy visible for plotting and comparison.

The shipped YAML config enables all five strategies. Torch runs all five by
default. JAX omits pair-centric from its default expansion; an explicit
pair-centric selection runs as coverage-only. JAX includes cluster-tile by
default when its CUDA, ``float32``, and fully periodic requirements are met.
The bundled H100 snapshot includes current JAX cluster-tile coverage at the
6, 15, and 25 Angstrom cutoffs.

## Benchmark Configuration

| Parameter | Value |
| --------- | ----- |
| Cutoffs | 6.0, 15.0, 25.0 Å |
| Configured Methods | `naive_scalar`, `naive_tile`, `cell_list_atom_centric`, `cell_list_pair_centric`, `cluster_tile` |
| System Type | CsCl (programmatic), NH₃ (PDB) |
| Warmup Iterations | 3 |
| Timing Iterations | 10 |
| Dtype | `float32` |

## Running Your Own Benchmarks

Run from the repository root:

```bash
RESULT_DIR="$BENCHMARK_SCRATCH/results/manual-nl-run"
python -m benchmarks.neighborlist.benchmark_neighborlist \
    --config benchmarks/neighborlist/benchmark_config.yaml \
    --output-dir "$RESULT_DIR"
```

For the JAX backend:

```bash
RESULT_DIR="$BENCHMARK_SCRATCH/results/manual-nl-run"
python -m benchmarks.neighborlist.benchmark_neighborlist \
    --config benchmarks/neighborlist/benchmark_config.yaml \
    --backend jax \
    --output-dir "$RESULT_DIR"
```

For direct Warp API timing:

```bash
RESULT_DIR="$BENCHMARK_SCRATCH/results/manual-nl-run"
python -m benchmarks.neighborlist.benchmark_neighborlist \
    --config benchmarks/neighborlist/benchmark_config.yaml \
    --backend warp \
    --method cell_list_atom_centric \
    --output-dir "$RESULT_DIR"
```

Use `--method` / `--methods` to restrict the benchmark to particular APIs, for
example `--method naive_tile cell_list_pair_centric`, and `--dry-run` to inspect
the expanded `(system, mode, method, cutoff)` plan before allocating GPU memory.
