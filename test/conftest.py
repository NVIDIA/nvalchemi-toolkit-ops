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
"""Top-level pytest coordination for CI GPU memory stability.

This file owns test-session behavior that must apply to every pytest process.
CI runs pytest once per top-level test module group via ``make testmon-coverage``,
so each process imports this file before collecting one of ``test/test_types.py``,
``test/math``, ``test/neighbors``, or ``test/interactions``.

The hooks here do three things:

* configure JAX's allocator before any test imports JAX,
* order framework binding tests as JAX -> framework-neutral Warp -> Torch, and
* release framework-owned GPU caches when leaving a JAX or Torch test block.

Framework detection is intentionally path based. Collection ordering and teardown
must not import JAX or Torch just to decide where a test belongs. Cleanup is also
best effort: a cache-release failure should be visible in the log, but should not
mask the original test result.

Keep broad pytest/session plumbing here. Put domain-specific fixtures near their
tests unless they truly need to affect every pytest invocation.
"""

import gc
import hashlib
import os
import sys
import tempfile
import warnings
from collections.abc import Callable
from typing import Literal

import pytest

FrameworkName = Literal["jax", "torch"]

_JAX_IMPORT_ENVIRONMENT = {
    "JAX_ENABLE_X64": "True",
    "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
}

warnings.filterwarnings(
    "ignore",
    message="`torch.jit.script_method` is deprecated.*",
    category=DeprecationWarning,
)

_FRAMEWORK_PATH_MARKERS: tuple[tuple[str, FrameworkName], ...] = (
    ("/bindings/jax/", "jax"),
    ("/bindings/torch/", "torch"),
)

_FRAMEWORK_SORT_RANK: dict[FrameworkName | None, int] = {
    "jax": 0,
    None: 1,
    "torch": 2,
}

_WORKTREE_CACHE_TOKEN = hashlib.sha256(os.getcwd().encode("utf-8")).hexdigest()[:12]
_WARP_CACHE_DEFAULT = os.path.join(
    os.environ.get("TMPDIR", tempfile.gettempdir()),
    f"warp_test_cache_{os.path.basename(os.getcwd())}_{_WORKTREE_CACHE_TOKEN}",
)
os.environ.setdefault("WARP_CACHE_PATH", _WARP_CACHE_DEFAULT)
os.makedirs(os.environ["WARP_CACHE_PATH"], exist_ok=True)


# ==============================================================================
# Import-Time Environment
# ==============================================================================


def _configure_jax_allocator_environment() -> None:
    """Set XLA allocator variables before collection-time imports can load JAX."""
    for name, value in _JAX_IMPORT_ENVIRONMENT.items():
        os.environ[name] = value


_configure_jax_allocator_environment()


# ==============================================================================
# Framework Classification
# ==============================================================================


def _item_path(item: pytest.Item) -> str:
    """Return a normalized path for a pytest item."""
    item_path = str(getattr(item, "path", None) or getattr(item, "fspath", ""))
    return item_path.replace(os.sep, "/")


def _item_framework(item: pytest.Item | None) -> FrameworkName | None:
    """Classify framework binding tests by path; None means framework-neutral."""
    if item is None:
        return None

    item_path = _item_path(item)
    for path_marker, framework in _FRAMEWORK_PATH_MARKERS:
        if path_marker in item_path:
            return framework
    return None


# ==============================================================================
# Collection Ordering
# ==============================================================================


def _framework_sort_key(item: pytest.Item) -> tuple[int, str, str]:
    """Return a stable JAX -> neutral -> Torch ordering key."""
    item_path = _item_path(item)
    return _FRAMEWORK_SORT_RANK[_item_framework(item)], item_path, item.name


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


# ==============================================================================
# Framework Boundary Cleanup
# ==============================================================================


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item: pytest.Item, nextitem: pytest.Item | None) -> None:
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


def _release_framework_gpu_memory(
    framework: FrameworkName,
    config: pytest.Config,
) -> None:
    """Synchronize and drop caches for the framework block that just finished."""
    if framework == "jax":
        _call_cleanup(_synchronize_jax_devices, framework, config)
        _call_cleanup(_clear_jax_caches, framework, config)
    elif framework == "torch":
        _call_cleanup(_synchronize_torch_cuda, framework, config)
        _call_cleanup(_clear_torch_cuda_cache, framework, config)
    gc.collect()


def _call_cleanup(
    cleanup: Callable[[], None],
    framework: FrameworkName,
    config: pytest.Config,
) -> None:
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


# ==============================================================================
# Framework-Specific Cleanup Helpers
# ==============================================================================


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
