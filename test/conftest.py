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
import gc
import os
import sys

import pytest

# These must be set before any test module imports JAX. The previous autouse
# fixture set them after collection-time imports, which is too late for XLA's
# allocator configuration.
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def _framework_sort_key(item: pytest.Item) -> tuple[int, str, str]:
    """Sort framework binding suites so JAX runs before Torch."""
    item_path = str(getattr(item, "path", None) or getattr(item, "fspath", ""))
    item_path = item_path.replace(os.sep, "/")

    if "/bindings/jax/" in item_path:
        framework_rank = 1
    elif "/bindings/torch/" in item_path:
        framework_rank = 2
    else:
        framework_rank = 0

    return framework_rank, item_path, item.name


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    """Group JAX binding tests before Torch binding tests within each run."""
    items.sort(key=_framework_sort_key)


@pytest.fixture(autouse=True)
def _release_gpu_memory():
    """Release framework-owned GPU caches after each test."""
    yield
    _release_framework_gpu_memory()


def _release_framework_gpu_memory() -> None:
    """Drop Python references and framework caches between tests."""
    gc.collect()
    _clear_jax_caches()
    _clear_torch_cuda_cache()
    gc.collect()


def _clear_jax_caches() -> None:
    """Clear JAX compilation caches if JAX was imported by the test."""
    jax = sys.modules.get("jax")
    if jax is None:
        return

    clear_caches = getattr(jax, "clear_caches", None)
    if clear_caches is not None:
        clear_caches()


def _clear_torch_cuda_cache() -> None:
    """Clear PyTorch CUDA allocator caches if Torch was imported by the test."""
    torch = sys.modules.get("torch")
    if torch is None:
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
