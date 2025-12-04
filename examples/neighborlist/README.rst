Neighbor Lists
==============

Examples demonstrating efficient O(N) neighbor list construction using GPU-accelerated
cell list algorithms.

These examples show how to:

* Build neighbor lists for single and batched systems
* Use dense or sparse COO output formats
* Detect when neighbor lists need rebuilding
* Optimize performance with ``torch.compile``
* Integrate with molecular dynamics workflows
