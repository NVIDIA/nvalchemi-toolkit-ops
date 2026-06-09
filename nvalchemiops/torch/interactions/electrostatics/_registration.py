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

"""Explicit registration point for Torch electrostatics custom ops."""

from __future__ import annotations

__all__ = ["ensure_electrostatics_ops_registered"]

_ELECTROSTATICS_OPS_REGISTERED = False


def ensure_electrostatics_ops_registered() -> None:
    """Register all Torch electrostatics custom-op chains once."""
    global _ELECTROSTATICS_OPS_REGISTERED
    if _ELECTROSTATICS_OPS_REGISTERED:
        return

    from nvalchemiops.torch.interactions.electrostatics._ewald_corrections_chain import (
        register_ewald_corrections_ops,
    )
    from nvalchemiops.torch.interactions.electrostatics._ewald_real_chain import (
        register_ewald_real_ops,
    )
    from nvalchemiops.torch.interactions.electrostatics._ewald_recip_chain import (
        register_ewald_recip_ops,
    )
    from nvalchemiops.torch.interactions.electrostatics._slab_chain import (
        register_slab_ops,
    )
    from nvalchemiops.torch.interactions.electrostatics.pme import register_pme_ops

    register_ewald_real_ops()
    register_ewald_recip_ops()
    register_ewald_corrections_ops()
    register_slab_ops()
    register_pme_ops()
    _ELECTROSTATICS_OPS_REGISTERED = True
