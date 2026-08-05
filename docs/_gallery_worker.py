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
"""Entry points for the isolated processes that execute gallery examples.

The private module name is deliberately absent from historical documentation
trees. Spawned processes unpickle functions by module path, so a generic name
such as ``sphinxext`` can resolve to an older ref's incompatible helper module.
"""


def run_example(*args, **kwargs):
    """Render one gallery example in its dedicated process.

    This seems roundabout, but the main issue is to try and get around
    `sphinx` needing to serialize functions for multiprocessing.
    """
    from sphinx_gallery.gen_rst import generate_file_rst

    return generate_file_rst(*args, **kwargs)


def reset_seeds(gallery_conf, fname):
    """Seed NumPy and Torch so example output is stable across rebuilds."""
    import numpy
    import torch

    numpy.random.seed(42)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
