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

"""Electrostatics aliases for the shared lazy ``jax_kernel`` wrappers.

The implementation lives in :mod:`nvalchemiops.jax._lazy_jax_kernels`, which the dispersion
bindings share. Kept as a module so the existing import sites here stay unchanged.
"""

from __future__ import annotations

from nvalchemiops.jax._lazy_jax_kernels import (
    make_jax_kernel_factory as _make_jax_kernel_factory,
)
from nvalchemiops.jax._lazy_jax_kernels import (
    make_jax_kernels as _make_jax_kernels,
)

__all__ = ["_make_jax_kernel_factory", "_make_jax_kernels"]
