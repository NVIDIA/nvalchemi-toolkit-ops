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

"""Regression tests for electrostatics pytest cleanup helpers."""

from __future__ import annotations

import sys
from types import SimpleNamespace

from test.interactions.electrostatics import conftest as electrostatics_conftest


def test_release_optional_gpu_memory_runs_gc_and_torch_cache(monkeypatch):
    """Cleanup helper runs GC and clears Torch CUDA cache when available."""
    calls: list[str] = []

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            calls.append("cuda_available")
            return True

        @staticmethod
        def empty_cache() -> None:
            calls.append("empty_cache")

    monkeypatch.setattr(
        electrostatics_conftest.gc,
        "collect",
        lambda: calls.append("gc") or 0,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=_Cuda))

    electrostatics_conftest._release_optional_gpu_memory()

    assert calls == ["gc", "cuda_available", "empty_cache"]
