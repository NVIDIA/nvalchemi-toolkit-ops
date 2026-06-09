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

"""Public API tests for JAX electrostatics exports."""

import inspect
from pathlib import Path

import pytest

import nvalchemiops.jax.interactions.electrostatics as electrostatics


def test_pme_metadata_preserves_legacy_positional_flag_order() -> None:
    """New PME metadata kwargs stay behind legacy positional flags."""
    reciprocal_names = list(
        inspect.signature(electrostatics.pme_reciprocal_space).parameters
    )
    full_names = list(inspect.signature(electrostatics.particle_mesh_ewald).parameters)
    legacy_flags = [
        "compute_forces",
        "compute_charge_gradients",
        "compute_virial",
        "hybrid_forces",
    ]
    metadata = ["cell_inv_t", "volume", "moduli_x", "moduli_y", "moduli_z"]

    k_squared_pos = reciprocal_names.index("k_squared")
    assert reciprocal_names[k_squared_pos + 1 : k_squared_pos + 5] == legacy_flags
    assert reciprocal_names[-5:] == metadata
    assert full_names[-5:] == metadata


def test_ewald_miller_bounds_is_keyword_only_after_legacy_slots() -> None:
    """Ewald Miller bounds must not steal legacy positional argument slots."""
    params = list(inspect.signature(electrostatics.ewald_summation).parameters.values())
    names = [param.name for param in params]
    assert names[names.index("k_cutoff") + 1 : names.index("pbc") + 1] == [
        "batch_idx",
        "max_atoms_per_system",
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
