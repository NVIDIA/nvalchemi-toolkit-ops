# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""JAX electrostatics interactions API.

This module provides JAX bindings for electrostatic calculations (Coulomb, Ewald, PME).
"""

from __future__ import annotations

from warnings import warn

import jax

if not getattr(jax.config, "jax_enable_x64", False):
    warn(
        "Electrostatics kernels rely on FP64, and `jax_enable_x64` is set to False."
        " `nvalchemiops` will set this value to True by default."
    )
    jax.config.update("jax_enable_x64", True)
