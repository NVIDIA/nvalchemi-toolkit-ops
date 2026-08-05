# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sphinx-Gallery hooks, and the entry point for the process an example runs in.

Everything here is reached by import path rather than by passing an object,
because each example runs in a spawned process and what crosses that boundary
has to pickle. ``conf.py`` names the reset hook as ``"sphinxext.reset_seeds"``
for that reason, and submits ``run_example`` rather than the Sphinx-Gallery
function it wraps -- that one is monkeypatched in the parent, so pickling it by
reference fails its own identity check.

The module is addressed as top-level ``sphinxext``, not ``docs.sphinxext``. That
resolves it out of the Sphinx conf directory, which sphinx-multiversion takes
from the branch driving the build. ``docs.sphinxext`` would instead resolve
against whichever ref is being rendered, so the historical tags would silently
run their own copy of this file.
"""


def run_example(*args, **kwargs):
    """Render one gallery example, in the process created to hold it.

    Imports Sphinx-Gallery here rather than at module scope: this runs in a
    fresh interpreter, where the function is the unpatched original.
    """
    from sphinx_gallery.gen_rst import generate_file_rst

    return generate_file_rst(*args, **kwargs)


def reset_seeds(gallery_conf, fname):
    """Seed NumPy and Torch so example output is stable across rebuilds.

    Every example gets a fresh interpreter, so each one would otherwise start
    from OS entropy and the examples that draw random numbers would rewrite
    their own output on every build.
    """
    import numpy
    import torch

    numpy.random.seed(42)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
