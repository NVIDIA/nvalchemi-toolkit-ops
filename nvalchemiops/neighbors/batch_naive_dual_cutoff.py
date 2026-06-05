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

"""Compatibility facade for batched naive dual-cutoff launchers.

Import from :mod:`nvalchemiops.neighbors.naive` instead."""

import warnings

from nvalchemiops.neighbors.naive.launchers import (
    batch_naive_neighbor_matrix_dual_cutoff,
    batch_naive_neighbor_matrix_pbc_dual_cutoff,
)

warnings.warn(
    "nvalchemiops.neighbors.batch_naive_dual_cutoff is deprecated; import batched naive dual-cutoff launchers from nvalchemiops.neighbors.naive instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "batch_naive_neighbor_matrix_dual_cutoff",
    "batch_naive_neighbor_matrix_pbc_dual_cutoff",
]
