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

"""Compatibility facade for rebuild-detection launchers.

Import from :mod:`nvalchemiops.neighbors.rebuild` instead. This module
preserves the retained public 0.3.1 import path without re-exporting private
kernels or private factory internals."""

import warnings

from nvalchemiops.neighbors.rebuild.launchers import (
    check_batch_cell_list_rebuild,
    check_batch_neighbor_list_rebuild,
    check_cell_list_rebuild,
    check_neighbor_list_rebuild,
    get_cell_list_rebuild_kernel,
    get_neighbor_list_rebuild_kernel,
)

warnings.warn(
    "nvalchemiops.neighbors.rebuild_detection is deprecated; import rebuild-detection launchers from nvalchemiops.neighbors.rebuild instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "check_batch_cell_list_rebuild",
    "check_batch_neighbor_list_rebuild",
    "check_cell_list_rebuild",
    "check_neighbor_list_rebuild",
    "get_cell_list_rebuild_kernel",
    "get_neighbor_list_rebuild_kernel",
]
