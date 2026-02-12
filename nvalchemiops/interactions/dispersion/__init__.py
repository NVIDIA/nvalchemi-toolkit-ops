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
"""
Core warp-based dispersion correction implementations.

This module provides framework-agnostic warp launchers for dispersion corrections.
For PyTorch bindings, use `nvalchemiops.torch.interactions.dispersion` instead.
"""

from __future__ import annotations

import importlib
import warnings

from nvalchemiops.interactions.dispersion._dftd3 import dftd3 as wp_dftd3
from nvalchemiops.interactions.dispersion._dftd3 import (
    dftd3_matrix,
    dftd3_matrix_pbc,
    dftd3_pbc,
)


def __getattr__(name: str):  # pragma: no cover
    """Lazy import for backward compatibility with the old API.

    This avoids circular imports by deferring the import of `dftd3`
    from `nvalchemiops.torch.interactions.dispersion` until it is actually accessed.
    """
    match name:
        case "dftd3":
            if importlib.util.find_spec("torch") is None:
                warnings.warn(
                    "From version 0.3.0 onwards, PyTorch is now an optional dependency"
                    " and a PyTorch installation was not detected. This namespace is"
                    " reserved for `warp` kernels directly. For end-users, import from"
                    " `nvalchemiops.torch.interactions.dispersion` instead.",
                    category=DeprecationWarning,
                    stacklevel=2,
                )

                def dftd3(*args, **kwargs):
                    """Raise a `RuntimeError` if we can't use the new API with torch."""
                    raise RuntimeError(
                        "PyTorch is required to use the previous `dftd3` API."
                        " Please install via `pip install 'nvalchemiops[torch]'`"
                        " and import from `nvalchemiops.torch.interactions.dispersion.dftd3` instead."
                    )

                return dftd3
            else:
                warnings.warn(
                    "From version 0.3.0 onwards, PyTorch is now an optional dependency"
                    " and the `nvalchemiops.interactions.dispersion` namespace is"
                    " reserved for `warp` kernels directly. For end-users, import from"
                    " `nvalchemiops.torch.interactions.dispersion` instead.",
                    category=DeprecationWarning,
                    stacklevel=2,
                )
                from nvalchemiops.torch.interactions.dispersion import dftd3

                return dftd3
        case "D3Parameters":
            if importlib.util.find_spec("torch") is None:
                raise RuntimeError(
                    "PyTorch is required to use the previous `D3Parameters` API."
                    " Please install via `pip install 'nvalchemiops[torch]'`"
                    " and import from `nvalchemiops.torch.interactions.dispersion.D3Parameters` instead."
                )
            else:
                warnings.warn(
                    "From version 0.3.0 onwards, PyTorch is now an optional dependency"
                    " and the `nvalchemiops.interactions.dispersion` namespace is"
                    " reserved for `warp` kernels directly. For end-users, import from"
                    " `nvalchemiops.torch.interactions.dispersion` instead.",
                    category=DeprecationWarning,
                    stacklevel=2,
                )
                from nvalchemiops.torch.interactions.dispersion import D3Parameters

                return D3Parameters
        case _:
            pass

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "dftd3_matrix",
    "dftd3_matrix_pbc",
    "dftd3",
    "wp_dftd3",
    "dftd3_pbc",
]
