Neighbor Lists
==============

Examples demonstrating efficient O(N) neighbor list construction using GPU-accelerated
cell list algorithms.

These examples show how to:

* Build neighbor lists for single and batched systems
* Use dense or sparse COO output formats
* Inspect Torch cost-model dispatch estimates
* Request compact source rows with ``target_indices``
* Return per-neighbor vectors and distances, including differentiable geometry
* Compute inline Lennard-Jones pair energies and forces with ``pair_fn``
* Detect when neighbor lists need rebuilding
* Optimize performance with ``torch.compile``
* Integrate with molecular dynamics workflows
