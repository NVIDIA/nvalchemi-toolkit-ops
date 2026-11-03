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

"""Tests for the device-aware real-space core precision resolver."""

from __future__ import annotations

import pytest
import warp as wp

from nvalchemiops.interactions.electrostatics._factory_common import (
    _read_legacy_fp32_env,
    _resolve_core_scalar,
)


@pytest.fixture(autouse=True)
def _reset_legacy_fp32_cache(monkeypatch):
    """Start every test from the default (non-legacy) mode with a fresh env read."""
    monkeypatch.delenv("NVALCHEMIOPS_ELECTROSTATICS_LEGACY_FP32", raising=False)
    _read_legacy_fp32_env.cache_clear()
    yield
    _read_legacy_fp32_env.cache_clear()


class TestResolveCoreScalar:
    """Float32 evaluation is CUDA-only; every other case stays float64."""

    def test_float64_always_float64(self):
        """Float64 inputs evaluate in float64 on any device."""
        assert _resolve_core_scalar(wp.float64, "cpu") is wp.float64

    def test_float32_cpu_stays_float64(self):
        """Float32 CPU inputs keep the float64 core, so CPU results are unchanged."""
        assert _resolve_core_scalar(wp.float32, "cpu") is wp.float64

    @pytest.mark.gpu
    @pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA is required")
    def test_float32_cuda_uses_float32_core(self):
        """Float32 CUDA inputs use the fast float32 core by default."""
        assert _resolve_core_scalar(wp.float32, "cuda:0") is wp.float32

    @pytest.mark.gpu
    @pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA is required")
    def test_legacy_flag_forces_float64_on_cuda(self, monkeypatch):
        """The legacy flag restores the float64 core for float32 CUDA inputs."""
        monkeypatch.setenv("NVALCHEMIOPS_ELECTROSTATICS_LEGACY_FP32", "1")
        _read_legacy_fp32_env.cache_clear()
        assert _resolve_core_scalar(wp.float32, "cuda:0") is wp.float64
