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

"""Private prepared-state periodic-image coverage guard."""

from __future__ import annotations

import torch
import warp as wp

from nvalchemiops.neighbors.neighbor_utils import check_naive_shift_coverage
from nvalchemiops.torch._warp_op_helpers import (
    scoped_torch_warp_stream,
    scoped_warp_stream,
)
from nvalchemiops.torch.types import get_wp_dtype, get_wp_mat_dtype

__all__ = ["check_prepared_naive_shift_coverage"]


@torch.library.custom_op(
    "nvalchemiops::_check_prepared_naive_shift_coverage",
    mutates_args=("status",),
)
@scoped_torch_warp_stream
def _check_prepared_naive_shift_coverage(
    cell: torch.Tensor,
    cutoff: float,
    pbc: torch.Tensor,
    prepared_range: torch.Tensor,
    rebuild_flags: torch.Tensor,
    status: torch.Tensor,
) -> None:
    """Run the device-side prepared periodic-image coverage check."""
    device = cell.device
    with scoped_warp_stream(device):
        check_naive_shift_coverage(
            cell=wp.from_torch(
                cell, dtype=get_wp_mat_dtype(cell.dtype), requires_grad=False
            ),
            cutoff=cutoff,
            pbc=wp.from_torch(pbc, dtype=wp.bool, requires_grad=False),
            prepared_range=wp.from_torch(
                prepared_range, dtype=wp.vec3i, requires_grad=False
            ),
            rebuild_flags=wp.from_torch(
                rebuild_flags, dtype=wp.bool, requires_grad=False
            ),
            status=wp.from_torch(status, dtype=wp.int32, requires_grad=False),
            wp_dtype=get_wp_dtype(cell.dtype),
            device=str(device),
        )


@_check_prepared_naive_shift_coverage.register_fake
def _(*args, **kwargs) -> None:
    return None


def check_prepared_naive_shift_coverage(
    cell: torch.Tensor,
    cutoff: float,
    pbc: torch.Tensor,
    prepared_range: torch.Tensor,
    rebuild_flags: torch.Tensor,
    status: torch.Tensor,
) -> None:
    """Launch the private guard and report eager failures clearly."""
    if not torch.compiler.is_compiling():
        if not bool(torch.isfinite(cell).all().item()):
            raise ValueError(
                "runtime cell contains non-finite values; re-prepare the naive state"
            )
        determinant = torch.linalg.det(
            cell.to(torch.float32) if cell.dtype == torch.float16 else cell
        )
        if bool(
            (~torch.isfinite(determinant) | (determinant.abs() <= 1.0e-12)).any().item()
        ):
            raise ValueError(
                "runtime cell is singular or invalid; re-prepare the naive state"
            )
    _check_prepared_naive_shift_coverage(
        cell, cutoff, pbc, prepared_range, rebuild_flags, status
    )
    if torch.compiler.is_compiling():
        torch._assert_async(
            torch.all(status == 1),
            "prepared naive periodic-image coverage is insufficient; re-prepare the state",
        )
    else:
        failed = status != 1
        if bool(failed.any().item()):
            system = int(failed.nonzero(as_tuple=False)[0, 0].item())
            if int(status[system].item()) == 0:
                raise ValueError(
                    "prepared naive periodic-image coverage is insufficient for "
                    f"system {system}; re-prepare the state"
                )
            raise ValueError(
                f"runtime cell for system {system} is singular or invalid; "
                "re-prepare the naive state"
            )
