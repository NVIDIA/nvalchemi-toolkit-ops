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

"""Public API tests for Torch electrostatics exports."""

import inspect
from pathlib import Path

import pytest
import torch

import nvalchemiops.torch.interactions.electrostatics as electrostatics
from nvalchemiops.torch.interactions.electrostatics._util import (
    _compiled_direct_output_deprecation_signal,
)


def test_pme_positional_slots_preserve_031_order() -> None:
    """PME cache metadata must not steal legacy positional argument slots."""
    recip_params = list(
        inspect.signature(electrostatics.pme_reciprocal_space).parameters
    )
    assert recip_params[
        recip_params.index("k_squared") + 1 : recip_params.index("cell_inv_t")
    ] == [
        "compute_forces",
        "compute_charge_gradients",
        "compute_virial",
        "hybrid_forces",
    ]
    assert recip_params[
        recip_params.index("cell_inv_t") : recip_params.index("moduli_x")
    ] == [
        "cell_inv_t",
        "volume",
    ]

    full_params = list(inspect.signature(electrostatics.particle_mesh_ewald).parameters)
    assert full_params[
        full_params.index("k_squared") + 1 : full_params.index("cell_inv_t")
    ] == [
        "neighbor_list",
        "neighbor_ptr",
        "neighbor_shifts",
        "neighbor_matrix",
        "neighbor_matrix_shifts",
        "mask_value",
        "compute_forces",
        "compute_charge_gradients",
        "compute_virial",
        "accuracy",
        "hybrid_forces",
        "pbc",
        "slab_correction",
    ]


def test_ewald_miller_bounds_is_keyword_only_after_legacy_slots() -> None:
    """Ewald Miller bounds must not steal legacy positional argument slots."""
    params = list(inspect.signature(electrostatics.ewald_summation).parameters.values())
    names = [param.name for param in params]
    assert names[names.index("k_cutoff") + 1 : names.index("pbc") + 1] == [
        "batch_idx",
        "neighbor_list",
        "neighbor_ptr",
        "neighbor_shifts",
        "neighbor_matrix",
        "neighbor_matrix_shifts",
        "mask_value",
        "compute_forces",
        "compute_charge_gradients",
        "compute_virial",
        "accuracy",
        "hybrid_forces",
        "pbc",
    ]
    assert params[names.index("miller_bounds")].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize(
    ("public_name", "private_name"),
    [
        ("pme_green_structure_factor", "_pme_green_structure_factor"),
        ("pme_energy_corrections", "_pme_energy_corrections"),
        (
            "pme_energy_corrections_with_charge_grad",
            "_pme_energy_corrections_with_charge_grad",
        ),
    ],
)
def test_top_level_low_level_pme_helpers_warn_at_call_site(
    public_name: str,
    private_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deprecated top-level PME aliases warn with public-call-site stacklevel."""
    monkeypatch.setattr(electrostatics, private_name, lambda *args, **kwargs: "ok")

    with pytest.warns(DeprecationWarning, match=public_name) as record:
        result = getattr(electrostatics, public_name)()

    assert result == "ok"
    assert Path(record[0].filename).name == Path(__file__).name


@pytest.mark.parametrize(
    ("public_name", "private_name"),
    [
        ("pme_green_structure_factor", "_pme_green_structure_factor"),
        ("pme_energy_corrections", "_pme_energy_corrections"),
        (
            "pme_energy_corrections_with_charge_grad",
            "_pme_energy_corrections_with_charge_grad",
        ),
    ],
)
def test_top_level_low_level_pme_helpers_preserve_signature_and_doc(
    public_name: str, private_name: str
) -> None:
    """Deprecated top-level PME aliases keep the wrapped helper API shape."""
    public = getattr(electrostatics, public_name)
    private = getattr(electrostatics, private_name)
    assert inspect.signature(public) == inspect.signature(private)
    assert "Deprecated top-level alias" in inspect.getdoc(public)
    assert inspect.getdoc(private) in inspect.getdoc(public)


def test_compiled_direct_output_deprecation_signal_uses_graph_break(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compiled deprecated direct-output calls get a graph-break migration signal."""
    messages: list[str] = []
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(
        torch._dynamo, "graph_break", lambda msg="": messages.append(msg)
    )

    _compiled_direct_output_deprecation_signal("particle_mesh_ewald")

    assert len(messages) == 1
    assert "direct-output flags" in messages[0]
    assert "particle_mesh_ewald" in messages[0]
