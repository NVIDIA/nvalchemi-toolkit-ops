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

"""Fixtures for the JAX dynamics binding tests."""

import pytest

pytest.importorskip("jax", reason="No JAX installed.")

import jax  # noqa: E402

# x64 is enabled so this suite can exercise the fp64 path; the fp32 path
# deliberately does not need it, which ``test_fp32_runs_without_x64`` checks in
# a subprocess.
jax.config.update("jax_enable_x64", True)


requires_gpu = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not any(d.platform == "gpu" for d in jax.devices()),
        reason="JAX L-BFGS bindings require a JAX GPU device",
    ),
]
