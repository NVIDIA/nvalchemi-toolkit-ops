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

"""Static audit for raw Warp usage in Torch-facing modules.

This is intentionally conservative: raw Warp setup is allowed inside registered
custom-op/runtime bodies and in a fixed backlog of host-only or chain-launcher
helpers. New raw Warp sites must either move behind a boundary or be classified
here with a clear reason.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDITED_ROOTS = (
    ROOT / "nvalchemiops" / "torch" / "neighbors",
    ROOT / "nvalchemiops" / "torch" / "interactions" / "electrostatics",
)

RAW_WARP_NAMES = {"warp_from_torch", "_wp_from_torch"}
RAW_WARP_ATTRS = {
    ("wp", "device_from_torch"),
    ("wp", "empty"),
    ("wp", "from_torch"),
    ("wp", "launch"),
    ("wp", "launch_tiled"),
    ("wp", "ScopedStream"),
    ("wp", "stream_from_torch"),
    ("wp", "zeros"),
}

# Backlog of existing non-decorated raw-Warp helpers. The comprehensive
# compile-ready work should shrink this set; this audit prevents it growing.
APPROVED_RAW_WARP_FUNCTIONS = {
    "nvalchemiops/torch/interactions/electrostatics/_ewald_corrections_chain.py::_batch_energy_corrections_backward_launch",
    "nvalchemiops/torch/interactions/electrostatics/_ewald_corrections_chain.py::_batch_energy_corrections_double_backward_launch",
    "nvalchemiops/torch/interactions/electrostatics/_ewald_corrections_chain.py::_batch_energy_corrections_forward_launch",
    "nvalchemiops/torch/interactions/electrostatics/_ewald_corrections_chain.py::_energy_corrections_backward_launch",
    "nvalchemiops/torch/interactions/electrostatics/_ewald_corrections_chain.py::_energy_corrections_double_backward_launch",
    "nvalchemiops/torch/interactions/electrostatics/_ewald_corrections_chain.py::_energy_corrections_forward_launch",
    "nvalchemiops/torch/interactions/electrostatics/_ewald_corrections_chain.py::_scoped_stream",
    "nvalchemiops/torch/interactions/electrostatics/_ewald_corrections_chain.py::_wp_from_torch",
    "nvalchemiops/torch/interactions/electrostatics/_ewald_direct.py::_fill",
    "nvalchemiops/torch/interactions/electrostatics/_ewald_direct.py::_scoped_stream",
    "nvalchemiops/torch/interactions/electrostatics/_ewald_direct.py::_wp",
    "nvalchemiops/torch/interactions/electrostatics/_ewald_real_chain.py::_backward_impl",
    "nvalchemiops/torch/interactions/electrostatics/_ewald_real_chain.py::_double_backward_impl",
    "nvalchemiops/torch/interactions/electrostatics/_ewald_real_chain.py::_forward_impl",
    "nvalchemiops/torch/interactions/electrostatics/_ewald_real_chain.py::_literal_cell_grad_backward",
    "nvalchemiops/torch/interactions/electrostatics/_ewald_real_chain.py::_literal_cell_grad_forward",
    "nvalchemiops/torch/interactions/electrostatics/_ewald_real_chain.py::_real_cell_grad_via_kernel",
    "nvalchemiops/torch/interactions/electrostatics/_ewald_real_chain.py::_scoped_stream",
    "nvalchemiops/torch/interactions/electrostatics/_ewald_real_chain.py::_wp",
    "nvalchemiops/torch/interactions/electrostatics/_ewald_real_chain.py::f64",
    "nvalchemiops/torch/interactions/electrostatics/_ewald_recip_chain.py::_backward_impl",
    "nvalchemiops/torch/interactions/electrostatics/_ewald_recip_chain.py::_double_backward_impl",
    "nvalchemiops/torch/interactions/electrostatics/_ewald_recip_chain.py::_forward_impl",
    "nvalchemiops/torch/interactions/electrostatics/_ewald_recip_chain.py::_run_fill",
    "nvalchemiops/torch/interactions/electrostatics/_ewald_recip_chain.py::_s_int_empty",
    "nvalchemiops/torch/interactions/electrostatics/_ewald_recip_chain.py::_scoped_stream",
    "nvalchemiops/torch/interactions/electrostatics/_ewald_recip_chain.py::_wp",
    "nvalchemiops/torch/interactions/electrostatics/_slab_chain.py::_run_geometry",
    "nvalchemiops/torch/interactions/electrostatics/_slab_chain.py::_run_moments",
    "nvalchemiops/torch/interactions/electrostatics/_slab_chain.py::_scoped_stream",
    "nvalchemiops/torch/interactions/electrostatics/_slab_chain.py::_slab_backward_values",
    "nvalchemiops/torch/interactions/electrostatics/_slab_chain.py::_slab_double_backward_launch",
    "nvalchemiops/torch/interactions/electrostatics/_slab_chain.py::_slab_forward_launch",
    "nvalchemiops/torch/interactions/electrostatics/_slab_chain.py::_slab_weighted_backward_values",
    "nvalchemiops/torch/interactions/electrostatics/_slab_chain.py::_slab_weighted_double_backward_values",
    "nvalchemiops/torch/interactions/electrostatics/_slab_chain.py::_wp_from_torch",
    "nvalchemiops/torch/interactions/electrostatics/pme.py::_batch_energy_corrections_backward_launch",
    "nvalchemiops/torch/interactions/electrostatics/pme.py::_batch_energy_corrections_charge_grad_forward_launch",
    "nvalchemiops/torch/interactions/electrostatics/pme.py::_batch_energy_corrections_double_backward_launch",
    "nvalchemiops/torch/interactions/electrostatics/pme.py::_batch_energy_corrections_forward_launch",
    "nvalchemiops/torch/interactions/electrostatics/pme.py::_energy_corrections_backward_launch",
    "nvalchemiops/torch/interactions/electrostatics/pme.py::_energy_corrections_charge_grad_forward_launch",
    "nvalchemiops/torch/interactions/electrostatics/pme.py::_energy_corrections_double_backward_launch",
    "nvalchemiops/torch/interactions/electrostatics/pme.py::_energy_corrections_forward_launch",
    "nvalchemiops/torch/interactions/electrostatics/pme.py::_pme_convolve_backward",
    "nvalchemiops/torch/interactions/electrostatics/pme.py::_pme_convolve_double_backward",
    "nvalchemiops/torch/interactions/electrostatics/pme.py::_pme_convolve_forward",
    "nvalchemiops/torch/interactions/electrostatics/pme.py::_pme_scoped_warp_stream",
    "nvalchemiops/torch/interactions/electrostatics/pme.py::_virial_bg_correction_backward_launch",
    "nvalchemiops/torch/interactions/electrostatics/pme.py::_virial_bg_correction_forward_launch",
    "nvalchemiops/torch/interactions/electrostatics/pme.py::_wp_from_torch",
    "nvalchemiops/torch/interactions/electrostatics/slab.py::_prepare_slab_geometry",
    "nvalchemiops/torch/interactions/electrostatics/slab.py::_run_slab_correction_op",
    "nvalchemiops/torch/neighbors/_dispatch.py::estimate_neighbor_list_costs",
    "nvalchemiops/torch/neighbors/batch_cell_list.py::_batch_query_cell_list_optional",
    "nvalchemiops/torch/neighbors/batch_cell_list.py::estimate_batch_cell_list_sizes",
    "nvalchemiops/torch/neighbors/batch_cluster_tile.py::_batch_cluster_tile_pair_outputs_forward",
    "nvalchemiops/torch/neighbors/batch_cluster_tile.py::_batch_query_cluster_tile_coo_optional",
    "nvalchemiops/torch/neighbors/batch_cluster_tile.py::_batch_query_cluster_tile_optional",
    "nvalchemiops/torch/neighbors/batch_cluster_tile.py::batch_cluster_tile_neighbor_list",
    "nvalchemiops/torch/neighbors/batch_cluster_tile.py::batch_query_cluster_tile",
    "nvalchemiops/torch/neighbors/batch_naive.py::_batch_naive_pair_outputs_forward",
    "nvalchemiops/torch/neighbors/cell_list.py::_query_cell_list_direct_eager",
    "nvalchemiops/torch/neighbors/cell_list.py::_query_cell_list_optional",
    "nvalchemiops/torch/neighbors/cell_list.py::estimate_cell_list_sizes",
    "nvalchemiops/torch/neighbors/cluster_tile.py::_cluster_tile_pair_outputs_forward",
    "nvalchemiops/torch/neighbors/cluster_tile.py::_mat33f_from_torch",
    "nvalchemiops/torch/neighbors/cluster_tile.py::_query_cluster_tile_coo_optional",
    "nvalchemiops/torch/neighbors/cluster_tile.py::_query_cluster_tile_optional",
    "nvalchemiops/torch/neighbors/cluster_tile.py::cluster_tile_neighbor_list",
    "nvalchemiops/torch/neighbors/naive.py::_naive_pair_outputs_forward",
    "nvalchemiops/torch/neighbors/neighbor_utils.py::compute_naive_num_shifts",
}

APPROVED_CUSTOM_OPS_WITHOUT_FAKE = {
    "nvalchemiops/torch/interactions/electrostatics/coulomb.py::_batch_coulomb_energy_forces_list",
    "nvalchemiops/torch/interactions/electrostatics/coulomb.py::_batch_coulomb_energy_forces_matrix",
    "nvalchemiops/torch/interactions/electrostatics/coulomb.py::_batch_coulomb_energy_list",
    "nvalchemiops/torch/interactions/electrostatics/coulomb.py::_batch_coulomb_energy_matrix",
    "nvalchemiops/torch/interactions/electrostatics/coulomb.py::_coulomb_energy_forces_list",
    "nvalchemiops/torch/interactions/electrostatics/coulomb.py::_coulomb_energy_forces_matrix",
    "nvalchemiops/torch/interactions/electrostatics/coulomb.py::_coulomb_energy_list",
    "nvalchemiops/torch/interactions/electrostatics/coulomb.py::_coulomb_energy_matrix",
}


def _call_name(node: ast.AST) -> str | tuple[str, str] | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return node.value.id, node.attr
    return None


def _decorator_name(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        node = node.func
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _decorator_target_name(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _has_boundary_decorator(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(
        name.endswith("custom_op") or name == "warp_custom_op"
        for name in (_decorator_name(dec) for dec in node.decorator_list)
    )


def _has_raw_warp_call(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        name = _call_name(child.func)
        if name in RAW_WARP_NAMES or name in RAW_WARP_ATTRS:
            return True
    return False


def _iter_raw_warp_functions() -> list[tuple[str, int, bool]]:
    functions = []
    for root in AUDITED_ROOTS:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            rel = path.relative_to(ROOT).as_posix()
            functions.extend(
                (
                    f"{rel}::{node.name}",
                    node.lineno,
                    _has_boundary_decorator(node),
                )
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and _has_raw_warp_call(node)
            )
    return sorted(functions)


def _iter_custom_ops() -> list[tuple[str, int, bool]]:
    ops = []
    for root in AUDITED_ROOTS:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            rel = path.relative_to(ROOT).as_posix()
            registered_fakes = set()
            custom_ops = []
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef):
                    continue
                for dec in node.decorator_list:
                    if _decorator_name(dec).endswith("custom_op"):
                        custom_ops.append((node.name, node.lineno))
                    if _decorator_target_name(dec) == "register_fake":
                        target = (
                            dec.func.value if isinstance(dec, ast.Call) else dec.value
                        )
                        if isinstance(target, ast.Name):
                            registered_fakes.add(target.id)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(
                    node.func, ast.Name
                ):
                    continue
                if "fake" not in node.func.id or not node.args:
                    continue
                target = node.args[0]
                if isinstance(target, ast.Name):
                    registered_fakes.add(target.id)
            ops.extend(
                (f"{rel}::{name}", lineno, name in registered_fakes)
                for name, lineno in custom_ops
            )
    return sorted(ops)


def test_raw_warp_setup_is_boundary_or_classified() -> None:
    """New raw Warp setup must be boundary-contained or explicitly classified."""
    disallowed = [
        f"{qualified}:{lineno}"
        for qualified, lineno, has_boundary in _iter_raw_warp_functions()
        if qualified not in APPROVED_RAW_WARP_FUNCTIONS and not has_boundary
    ]
    assert not disallowed, "Unclassified raw Warp setup:\n" + "\n".join(disallowed)


def test_custom_ops_have_fake_or_are_classified() -> None:
    """New torch.library custom ops must declare fake/meta behavior."""
    disallowed = [
        f"{qualified}:{lineno}"
        for qualified, lineno, has_fake in _iter_custom_ops()
        if not has_fake and qualified not in APPROVED_CUSTOM_OPS_WITHOUT_FAKE
    ]
    assert not disallowed, "Custom ops without register_fake:\n" + "\n".join(disallowed)
