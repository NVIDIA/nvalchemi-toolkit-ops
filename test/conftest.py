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


def _item_path(item: pytest.Item) -> str:
    """Return a normalized path for a pytest item."""
    item_path = str(getattr(item, "path", None) or getattr(item, "fspath", ""))
    return item_path.replace(os.sep, "/")


def _item_framework(item: pytest.Item | None) -> str | None:
    """Classify framework binding tests by path."""
    if item is None:
        return None

    item_path = _item_path(item)
    if "/bindings/jax/" in item_path:
        return "jax"
    if "/bindings/torch/" in item_path:
        return "torch"
    return None


def _framework_sort_key(item: pytest.Item) -> tuple[int, str, str]:
    """Sort framework suites so JAX runs before Warp and Torch CUDA tests."""
    item_path = _item_path(item)
    framework = _item_framework(item)
    if framework == "jax":
        framework_rank = 0
    elif framework == "torch":
        framework_rank = 2
    else:
        framework_rank = 1

    return framework_rank, item_path, item.name


def _sort_items_by_framework(items: list[pytest.Item]) -> None:
    """Sort pytest items in place by framework execution order."""
    items.sort(key=_framework_sort_key)


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    """Group JAX binding tests before Torch binding tests during collection."""
    _sort_items_by_framework(items)


@pytest.hookimpl(trylast=True)
def pytest_collection_finish(session):
    """Re-apply framework ordering after collection plugins finish deselection."""
    _sort_items_by_framework(session.items)


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item, nextitem):
    """Release framework GPU memory only after leaving framework blocks."""
    framework = _item_framework(item)
    if framework is None:
        return

    next_framework = _item_framework(nextitem)
    if framework == next_framework:
        return

    _write_cleanup_log(
        item.config,
        f"GPU cleanup: leaving {framework} before {_cleanup_destination(nextitem)}",
    )
    _release_framework_gpu_memory(framework, item.config)


def _cleanup_destination(nextitem: pytest.Item | None) -> str:
    """Describe where execution is headed after a cleanup boundary."""
    if nextitem is None:
        return "end of pytest run"

    next_framework = _item_framework(nextitem)
    if next_framework is None:
        return "base tests"
    return f"{next_framework} tests"


def _release_framework_gpu_memory(framework: str, config: pytest.Config) -> None:
    """Drop caches for the framework block that just finished."""
    if framework == "jax":
        _call_cleanup(_synchronize_jax_devices, framework, config)
        _call_cleanup(_clear_jax_caches, framework, config)
    elif framework == "torch":
        _call_cleanup(_synchronize_torch_cuda, framework, config)
        _call_cleanup(_clear_torch_cuda_cache, framework, config)
    gc.collect()


def _call_cleanup(cleanup, framework: str, config: pytest.Config) -> None:
    """Run cleanup best-effort so teardown does not mask test failures."""
    try:
        cleanup()
    except Exception as exc:
        message = str(exc).splitlines()[0]
        _write_cleanup_log(
            config,
            f"GPU cleanup: {framework} {cleanup.__name__} failed: "
            f"{type(exc).__name__}: {message}",
        )


def _write_cleanup_log(config: pytest.Config, message: str) -> None:
    """Write a concise cleanup message to pytest terminal output."""
    terminal_reporter = config.pluginmanager.get_plugin("terminalreporter")
    if terminal_reporter is not None:
        terminal_reporter.write_line(message)


def _synchronize_jax_devices() -> None:
    """Wait for pending JAX work before clearing JAX caches."""
    jax = sys.modules.get("jax")
    if jax is None:
        return

    effects_barrier = getattr(jax, "effects_barrier", None)
    if effects_barrier is not None:
        effects_barrier()

    for device in jax.devices():
        synchronize = getattr(device, "synchronize_all_activity", None)
        if synchronize is not None:
            synchronize()


def _clear_jax_caches() -> None:
    """Clear JAX compilation caches if JAX was imported by the test."""
    jax = sys.modules.get("jax")
    if jax is None:
        return

    clear_caches = getattr(jax, "clear_caches", None)
    if clear_caches is not None:
        clear_caches()


def _synchronize_torch_cuda() -> None:
    """Wait for pending PyTorch CUDA work before releasing allocator caches."""
    torch = sys.modules.get("torch")
    if torch is None:
        return

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _clear_torch_cuda_cache() -> None:
    """Clear PyTorch CUDA allocator caches if Torch was imported by the test."""
    torch = sys.modules.get("torch")
    if torch is None:
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
