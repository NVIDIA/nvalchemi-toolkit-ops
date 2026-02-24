Segment Operations Benchmarks
==============================

Performance benchmarks for segmented operations (Warp kernels) against
PyTorch equivalents, including:

* Reductions: ``segmented_sum``, ``segmented_component_sum``,
  ``segmented_dot``, ``segmented_max_norm``
* Broadcasts: ``segmented_mul``, ``segmented_add``, ``segmented_matvec``
* Dtype coverage: float32, float64, vec3f, vec3d

Configuration
-------------

All parameters are controlled via ``benchmark_config.yaml``:

* ``parameters``: warmup and timing run counts
* ``sweep``: total element counts and average segment lengths
* ``operations``: enable/disable individual operations and dtype variants

Usage
-----

.. code-block:: bash

   python -m benchmarks.segment_ops.benchmark_segment_ops \
       --config benchmark_config.yaml \
       --output-dir ./benchmark_results \
       --device cuda:0

Output
------

CSV file: ``segment_ops_benchmark_{gpu_sku}.csv``

Schema::

    operation,dtype,total_elements,num_segments,avg_segment_length,
    warp_median_ms,torch_median_ms,speedup
