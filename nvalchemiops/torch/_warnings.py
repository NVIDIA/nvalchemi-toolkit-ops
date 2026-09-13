# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Shared warning helpers for PyTorch bindings."""

from __future__ import annotations

import warnings

import torch

__all__: list[str] = []


def _warn_compile_missing_argument_inference(
    missing: str,
    inference: str,
    *,
    stacklevel: int = 3,
) -> None:
    """Warn when compiled execution falls back to host-side inference.

    Examples
    --------
    Warn when ``max_atoms_per_system`` is inferred from ``batch_ptr``:

    >>> _warn_compile_missing_argument_inference(
    ...     missing="`max_atoms_per_system`",
    ...     inference="inferring it from `batch_ptr`",
    ... )
    """
    if torch.compiler.is_compiling():
        warnings.warn(
            f"Missing {missing}; {inference} introduces a graph break under "
            "torch.compile. This will become an error in a future release.",
            FutureWarning,
            stacklevel=stacklevel,
        )
