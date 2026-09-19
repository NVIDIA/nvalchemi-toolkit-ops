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

"""Unit tests for common lazy JAX registration primitives."""

from __future__ import annotations

import functools
import importlib

import jax.numpy as jnp
import pytest
import warp as wp
from warp import JaxCallableGraphMode

from nvalchemiops.jax.neighbors import _cluster_tile_preload, _registration


class TestRegistrationHelpers:
    """Test the direct Warp/JAX registration adapters."""

    def test_register_jax_kernel_preserves_output_order_and_disables_backward(
        self, monkeypatch
    ) -> None:
        """Register kernels with the output tuple's count and order."""
        calls = []

        def fake_jax_kernel(kernel, **kwargs):
            calls.append(
                (kernel, {**kwargs, "in_out_argnames": list(kwargs["in_out_argnames"])})
            )
            kwargs["in_out_argnames"].pop()
            return "registered-kernel"

        monkeypatch.setattr(_registration, "jax_kernel", fake_jax_kernel)
        outputs = ("neighbor_matrix", "num_neighbors", "neighbor_shifts")

        result = _registration._register_jax_kernel("warp-kernel", outputs)

        assert result == "registered-kernel"
        assert calls == [
            (
                "warp-kernel",
                {
                    "num_outputs": 3,
                    "in_out_argnames": [
                        "neighbor_matrix",
                        "num_neighbors",
                        "neighbor_shifts",
                    ],
                    "enable_backward": False,
                },
            )
        ]
        assert outputs == (
            "neighbor_matrix",
            "num_neighbors",
            "neighbor_shifts",
        )

    def test_register_jax_kernel_list_mutation_does_not_affect_source_tuple(
        self, monkeypatch
    ) -> None:
        """Pass a fresh list to jax_kernel without mutating the source tuple."""
        calls = []

        def fake_jax_kernel(kernel, **kwargs):
            calls.append(list(kwargs["in_out_argnames"]))
            kwargs["in_out_argnames"].append("mutated")
            return "registered-kernel"

        monkeypatch.setattr(_registration, "jax_kernel", fake_jax_kernel)
        outputs = ("neighbor_matrix", "num_neighbors")

        _registration._register_jax_kernel("warp-kernel", outputs)

        assert outputs == ("neighbor_matrix", "num_neighbors")
        assert calls == [["neighbor_matrix", "num_neighbors"]]

    def test_register_jax_callable_uses_explicit_graph_mode(self, monkeypatch) -> None:
        """Register callables with the requested graph mode and output order."""
        calls = []

        def fake_jax_callable(callable_obj, **kwargs):
            calls.append(
                (
                    callable_obj,
                    {**kwargs, "in_out_argnames": list(kwargs["in_out_argnames"])},
                )
            )
            kwargs["in_out_argnames"].pop()
            return "registered-callable"

        monkeypatch.setattr(_registration, "jax_callable", fake_jax_callable)
        outputs = ("sorted_positions", "neighbor_matrix")

        result = _registration._register_jax_callable(
            "warp-callable",
            outputs,
            graph_mode=JaxCallableGraphMode.NONE,
        )

        assert result == "registered-callable"
        assert calls == [
            (
                "warp-callable",
                {
                    "num_outputs": 2,
                    "in_out_argnames": ["sorted_positions", "neighbor_matrix"],
                    "graph_mode": JaxCallableGraphMode.NONE,
                },
            )
        ]
        assert outputs == ("sorted_positions", "neighbor_matrix")


class TestLazyJaxKernel:
    """Test lazy dtype-specific direct kernel registration."""

    def test_constructs_and_caches_dtype_registration(self, monkeypatch) -> None:
        """Construct each supported dtype once and cache by normalized dtype."""
        built_dtypes = []
        registration_calls = []

        def build(wp_dtype):
            built_dtypes.append(wp_dtype)
            return "warp-kernel", ("out_a", "out_b")

        def fake_register(kernel, outputs):
            registration_calls.append((kernel, outputs))
            return "jax-kernel"

        monkeypatch.setattr(_registration, "_register_jax_kernel", fake_register)
        registrations = _registration._LazyJaxKernel(
            build,
            {jnp.float32: wp.float32},
        )

        first = registrations[jnp.float32]
        second = registrations[jnp.dtype("float32")]

        assert first == "jax-kernel"
        assert second is first
        assert built_dtypes == [wp.float32]
        assert registration_calls == [("warp-kernel", ("out_a", "out_b"))]

    def test_supports_float64_and_dtype_membership(self, monkeypatch) -> None:
        """Recognize supported dtypes and convert float64 before construction."""
        built_dtypes = []

        def build(wp_dtype):
            built_dtypes.append(wp_dtype)
            return wp_dtype, ("out",)

        monkeypatch.setattr(
            _registration,
            "_register_jax_kernel",
            lambda kernel, outputs: kernel,
        )
        registrations = _registration._LazyJaxKernel(
            build,
            {jnp.float32: wp.float32, jnp.float64: wp.float64},
        )

        assert jnp.float32 in registrations
        assert jnp.dtype("float64") in registrations
        assert jnp.int32 not in registrations
        assert registrations[jnp.float64] == wp.float64
        assert built_dtypes == [wp.float64]

    def test_float32_only_mapping_rejects_unsupported_dtype_before_build(
        self,
    ) -> None:
        """Raise KeyError before build for an undeclared dtype."""
        built_dtypes = []

        def build(wp_dtype):
            built_dtypes.append(wp_dtype)
            return "warp", ("out",)

        registrations = _registration._LazyJaxKernel(
            build,
            {jnp.float32: wp.float32},
        )

        assert jnp.float32 in registrations
        assert jnp.float64 not in registrations

        with pytest.raises(KeyError):
            registrations[jnp.float64]

        assert built_dtypes == []

    def test_copies_caller_supplied_dtype_map(self) -> None:
        """Preserve the declared mappings after the caller mutates its map."""
        dtype_map = {jnp.float32: wp.float32}
        registrations = _registration._LazyJaxKernel(
            lambda wp_dtype: ("warp", ("out",)),
            dtype_map,
        )
        dtype_map[jnp.float64] = wp.float64

        assert jnp.float64 not in registrations

        with pytest.raises(KeyError):
            registrations[jnp.float64]

    def test_forwards_exact_output_tuple_to_register(self, monkeypatch) -> None:
        """Pass the build callback's output tuple unchanged to registration."""
        registration_calls = []
        monkeypatch.setattr(
            _registration,
            "_register_jax_kernel",
            lambda kernel, outputs: registration_calls.append(outputs) or "jax",
        )
        registrations = _registration._LazyJaxKernel(
            lambda wp_dtype: ("warp", ("neighbor_matrix", "num_neighbors")),
            {jnp.float32: wp.float32},
        )

        assert registrations[jnp.float32] == "jax"
        assert registration_calls == [("neighbor_matrix", "num_neighbors")]

    def test_constant_cache_key_reuses_one_wrapper_across_dtypes(
        self, monkeypatch
    ) -> None:
        """Share one registered wrapper when cache_key is dtype-independent."""
        built_dtypes = []
        monkeypatch.setattr(
            _registration,
            "_register_jax_kernel",
            lambda kernel, outputs: object(),
        )
        registrations = _registration._LazyJaxKernel(
            lambda wp_dtype: (built_dtypes.append(wp_dtype) or "warp", ("out",)),
            {jnp.float32: wp.float32, jnp.float64: wp.float64},
            cache_key=lambda wp_dtype: "cells_per_system",
        )

        assert registrations[jnp.float32] is registrations[jnp.float64]
        assert built_dtypes == [wp.float32]


class TestNaiveRegistrationFactory:
    """Test lazy naive direct-kernel factory registrations."""

    def test_single_cutoff_no_pbc_topology(self, monkeypatch) -> None:
        """Register single-cutoff no-PBC topology with exact getter kwargs."""
        getter_calls = []
        registration_calls = []
        monkeypatch.setattr(
            _registration,
            "get_naive_neighbor_matrix_kernel",
            lambda wp_dtype, **kwargs: getter_calls.append((wp_dtype, kwargs))
            or "warp",
        )
        monkeypatch.setattr(
            _registration,
            "_register_jax_kernel",
            lambda kernel, outputs: registration_calls.append((kernel, outputs))
            or "jax",
        )
        registrations = _registration._lazy_naive_kernel(
            operation="single_cutoff",
            batched=False,
            pbc_mode="none",
            selective=True,
            half_fill=True,
        )

        assert registrations[jnp.float32] == "jax"
        assert getter_calls == [
            (
                wp.float32,
                {
                    "pbc_mode": "none",
                    "batched": False,
                    "selective": True,
                    "partial": False,
                    "half_fill": True,
                    "return_vectors": False,
                    "return_distances": False,
                    "pair_fn": None,
                },
            )
        ]
        assert registration_calls == [
            ("warp", ("neighbor_matrix1", "num_neighbors1")),
        ]

    def test_single_cutoff_pbc_geometry_and_pair_fn(self, monkeypatch) -> None:
        """Register PBC geometry and pair outputs with exact ABI order."""
        getter_calls = []
        registration_calls = []
        pair_fn = object()
        monkeypatch.setattr(
            _registration,
            "get_naive_neighbor_matrix_kernel",
            lambda wp_dtype, **kwargs: getter_calls.append((wp_dtype, kwargs))
            or "warp",
        )
        monkeypatch.setattr(
            _registration,
            "_register_jax_kernel",
            lambda kernel, outputs: registration_calls.append((kernel, outputs))
            or "jax",
        )
        registrations = _registration._lazy_naive_kernel(
            operation="single_cutoff",
            batched=True,
            pbc_mode="wrap_on_entry",
            partial=True,
            half_fill=True,
            geometry=True,
            pair_fn=pair_fn,
        )

        assert registrations[jnp.float64] == "jax"
        assert getter_calls == [
            (
                wp.float64,
                {
                    "pbc_mode": "wrap_on_entry",
                    "batched": True,
                    "selective": False,
                    "partial": True,
                    "half_fill": True,
                    "return_vectors": True,
                    "return_distances": True,
                    "pair_fn": pair_fn,
                },
            )
        ]
        assert registration_calls == [
            (
                "warp",
                (
                    "neighbor_matrix1",
                    "neighbor_matrix_shifts1",
                    "num_neighbors1",
                    "neighbor_vectors",
                    "neighbor_distances",
                    "pair_energies",
                    "pair_forces",
                ),
            )
        ]

    def test_dual_cutoff_pbc(self, monkeypatch) -> None:
        """Register dual-cutoff PBC with the dual getter and ABI."""
        getter_calls = []
        registration_calls = []
        monkeypatch.setattr(
            _registration,
            "get_naive_neighbor_matrix_dual_cutoff_kernel",
            lambda wp_dtype, **kwargs: getter_calls.append((wp_dtype, kwargs))
            or "warp",
        )
        monkeypatch.setattr(
            _registration,
            "_register_jax_kernel",
            lambda kernel, outputs: registration_calls.append((kernel, outputs))
            or "jax",
        )
        registrations = _registration._lazy_naive_kernel(
            operation="dual_cutoff",
            batched=False,
            pbc_mode="prewrapped",
            selective=True,
        )

        assert registrations[jnp.float32] == "jax"
        assert getter_calls == [
            (
                wp.float32,
                {
                    "pbc_mode": "prewrapped",
                    "batched": False,
                    "selective": True,
                },
            )
        ]
        assert registration_calls == [
            (
                "warp",
                (
                    "neighbor_matrix1",
                    "neighbor_matrix_shifts1",
                    "num_neighbors1",
                    "neighbor_matrix2",
                    "neighbor_matrix_shifts2",
                    "num_neighbors2",
                ),
            )
        ]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"operation": "invalid"},
            {"pair_fn": object(), "geometry": False},
            {
                "operation": "dual_cutoff",
                "geometry": True,
            },
            {
                "operation": "dual_cutoff",
                "partial": True,
            },
            {
                "operation": "dual_cutoff",
                "half_fill": True,
            },
            {"partial": True, "geometry": False},
            {"partial": True, "selective": True, "geometry": True},
            {"pbc_mode": "invalid"},
        ],
    )
    def test_rejects_invalid_options_before_side_effects(
        self, monkeypatch, kwargs
    ) -> None:
        """Reject unsupported naive options without getter or registration."""
        getter_calls = []
        registration_calls = []
        monkeypatch.setattr(
            _registration,
            "get_naive_neighbor_matrix_kernel",
            lambda *args, **kwargs: getter_calls.append((args, kwargs)),
        )
        monkeypatch.setattr(
            _registration,
            "get_naive_neighbor_matrix_dual_cutoff_kernel",
            lambda *args, **kwargs: getter_calls.append((args, kwargs)),
        )
        monkeypatch.setattr(
            _registration,
            "_register_jax_kernel",
            lambda *args, **kwargs: registration_calls.append((args, kwargs)),
        )
        defaults = {
            "operation": "single_cutoff",
            "batched": False,
            "pbc_mode": "none",
        }
        defaults.update(kwargs)
        registrations = _registration._lazy_naive_kernel(**defaults)

        with pytest.raises(ValueError):
            registrations[jnp.float32]

        assert getter_calls == []
        assert registration_calls == []


class TestCellListRegistrationFactory:
    """Test lazy cell-list direct-kernel factory registrations."""

    def test_registers_each_build_stage_with_exact_abi(self, monkeypatch) -> None:
        """Register every supported build stage with exact output tuples."""
        getter_calls = []
        registration_calls = []
        monkeypatch.setattr(
            _registration,
            "get_build_cell_list_kernel",
            lambda *args, **kwargs: getter_calls.append((args, kwargs)) or "warp",
        )
        monkeypatch.setattr(
            _registration,
            "get_cell_list_cells_per_system_kernel",
            lambda: "cells-warp",
        )
        monkeypatch.setattr(
            _registration,
            "get_gather_positions_and_shifts_kernel",
            lambda wp_dtype: "warp",
        )
        monkeypatch.setattr(
            _registration,
            "get_cell_list_gather_kernel",
            lambda wp_dtype: "warp",
        )
        monkeypatch.setattr(
            _registration,
            "_register_jax_kernel",
            lambda kernel, outputs: registration_calls.append((kernel, outputs))
            or "jax",
        )

        expected = {
            ("construct_bin_size", False): ("cells_per_dimension_single",),
            ("construct_bin_size", True): ("cells_per_dimension_batch",),
            ("count_atoms", False): (
                "atoms_per_cell_count",
                "atom_periodic_shifts",
            ),
            ("count_atoms", True): ("atoms_per_cell_count", "atom_periodic_shifts"),
            ("bin_atoms", False): (
                "atom_to_cell_mapping",
                "atoms_per_cell_count",
                "cell_atom_list",
            ),
            ("bin_atoms", True): (
                "atom_to_cell_mapping",
                "atoms_per_cell_count",
                "cell_atom_list",
            ),
            ("gather", False): ("dst_pos", "dst_shifts"),
            ("gather", True): ("sorted_positions", "sorted_shifts"),
            ("cells_per_system", True): ("cells_per_system",),
        }
        for (stage, batched), outputs in expected.items():
            registrations = _registration._lazy_cell_list_build_kernel(
                stage=stage,
                batched=batched,
            )
            assert registrations[jnp.float32] == "jax"
            assert registration_calls[-1] == (
                "cells-warp" if stage == "cells_per_system" else "warp",
                outputs,
            )

        assert getter_calls == [
            (("construct_bin_size", wp.float32), {"batched": False}),
            (("construct_bin_size", wp.float32), {"batched": True}),
            (("count_atoms", wp.float32), {"batched": False}),
            (("count_atoms", wp.float32), {"batched": True}),
            (("bin_atoms", wp.float32), {"batched": False}),
            (("bin_atoms", wp.float32), {"batched": True}),
        ]

    def test_gather_stage_routes_unbatched_and_batched_getters(
        self, monkeypatch
    ) -> None:
        """Route gather builds to the unbatched and batched gather getters."""
        unbatched_calls = []
        batched_calls = []
        monkeypatch.setattr(
            _registration,
            "get_gather_positions_and_shifts_kernel",
            lambda wp_dtype: unbatched_calls.append(wp_dtype) or "unbatched-warp",
        )
        monkeypatch.setattr(
            _registration,
            "get_cell_list_gather_kernel",
            lambda wp_dtype: batched_calls.append(wp_dtype) or "batched-warp",
        )
        monkeypatch.setattr(
            _registration,
            "_register_jax_kernel",
            lambda kernel, outputs: "jax",
        )

        unbatched = _registration._lazy_cell_list_build_kernel(
            stage="gather",
            batched=False,
        )
        batched = _registration._lazy_cell_list_build_kernel(
            stage="gather",
            batched=True,
        )

        assert unbatched[jnp.float32] == "jax"
        assert batched[jnp.float64] == "jax"
        assert unbatched_calls == [wp.float32]
        assert batched_calls == [wp.float64]

    def test_cells_per_system_reuses_dtype_independent_wrapper(
        self, monkeypatch
    ) -> None:
        """Share one cells-per-system wrapper across floating dtypes."""
        getter_calls = []
        monkeypatch.setattr(
            _registration,
            "get_cell_list_cells_per_system_kernel",
            lambda: getter_calls.append("called") or "cells-warp",
        )
        monkeypatch.setattr(
            _registration,
            "_register_jax_kernel",
            lambda kernel, outputs: object(),
        )
        registrations = _registration._lazy_cell_list_build_kernel(
            stage="cells_per_system",
            batched=True,
        )

        assert registrations[jnp.float32] is registrations[jnp.float64]
        assert getter_calls == ["called"]

    def test_registers_unbatched_sorted_and_direct_queries(self, monkeypatch) -> None:
        """Register unbatched sorted and direct query paths with exact kwargs."""
        getter_calls = []
        registration_calls = []
        monkeypatch.setattr(
            _registration,
            "get_query_cell_list_kernel",
            lambda *args, **kwargs: getter_calls.append((args, kwargs)) or "warp",
        )
        monkeypatch.setattr(
            _registration,
            "_register_jax_kernel",
            lambda kernel, outputs: registration_calls.append((kernel, outputs))
            or "jax",
        )
        pair_fn = object()
        sorted_regs = _registration._lazy_cell_list_query_kernel(
            batched=False,
            selective=True,
            partial=True,
            half_fill=True,
            geometry=True,
            pair_fn=pair_fn,
            atom_centric_path="sorted",
        )
        direct_regs = _registration._lazy_cell_list_query_kernel(
            batched=False,
            selective=False,
            atom_centric_path="direct",
        )

        assert sorted_regs[jnp.float64] == "jax"
        assert direct_regs[jnp.float32] == "jax"
        assert getter_calls == [
            (
                (wp.float64,),
                {
                    "strategy": "atom_centric",
                    "batched": False,
                    "selective": True,
                    "partial": True,
                    "half_fill": True,
                    "return_vectors": True,
                    "return_distances": True,
                    "pair_fn": pair_fn,
                    "atom_centric_path": "sorted",
                },
            ),
            (
                (wp.float32,),
                {
                    "strategy": "atom_centric",
                    "batched": False,
                    "selective": False,
                    "partial": False,
                    "half_fill": False,
                    "return_vectors": False,
                    "return_distances": False,
                    "pair_fn": None,
                    "atom_centric_path": "direct",
                },
            ),
        ]
        assert registration_calls == [
            (
                "warp",
                (
                    "neighbor_matrix",
                    "neighbor_matrix_shifts",
                    "num_neighbors",
                    "neighbor_vectors",
                    "neighbor_distances",
                    "pair_energies",
                    "pair_forces",
                ),
            ),
            (
                "warp",
                (
                    "neighbor_matrix",
                    "neighbor_matrix_shifts",
                    "num_neighbors",
                ),
            ),
        ]

    def test_registers_batched_sorted_query(self, monkeypatch) -> None:
        """Register batched sorted queries with topology-only ABI."""
        getter_calls = []
        registration_calls = []
        monkeypatch.setattr(
            _registration,
            "get_query_cell_list_kernel",
            lambda *args, **kwargs: getter_calls.append((args, kwargs)) or "warp",
        )
        monkeypatch.setattr(
            _registration,
            "_register_jax_kernel",
            lambda kernel, outputs: registration_calls.append((kernel, outputs))
            or "jax",
        )
        registrations = _registration._lazy_cell_list_query_kernel(
            batched=True,
            selective=True,
            half_fill=True,
        )

        assert registrations[jnp.float32] == "jax"
        assert getter_calls == [
            (
                (wp.float32,),
                {
                    "strategy": "atom_centric",
                    "batched": True,
                    "selective": True,
                    "partial": False,
                    "half_fill": True,
                    "return_vectors": False,
                    "return_distances": False,
                    "pair_fn": None,
                    "atom_centric_path": "sorted",
                },
            )
        ]
        assert registration_calls == [
            (
                "warp",
                (
                    "neighbor_matrix",
                    "neighbor_matrix_shifts",
                    "num_neighbors",
                ),
            )
        ]

    @pytest.mark.parametrize(
        "factory_name, kwargs",
        [
            (
                "_lazy_cell_list_build_kernel",
                {"stage": "cells_per_system", "batched": False},
            ),
            (
                "_lazy_cell_list_build_kernel",
                {"stage": "invalid", "batched": False},
            ),
            (
                "_lazy_cell_list_query_kernel",
                {"batched": True, "atom_centric_path": "direct"},
            ),
            (
                "_lazy_cell_list_query_kernel",
                {"batched": False, "geometry": False, "pair_fn": object()},
            ),
        ],
    )
    def test_rejects_invalid_options_before_side_effects(
        self, monkeypatch, factory_name, kwargs
    ) -> None:
        """Reject unsupported cell-list options without getter or registration."""
        factory = getattr(_registration, factory_name)
        getter_calls = []
        registration_calls = []
        monkeypatch.setattr(
            _registration,
            "get_build_cell_list_kernel",
            lambda *args, **kwargs: getter_calls.append((args, kwargs)),
        )
        monkeypatch.setattr(
            _registration,
            "get_query_cell_list_kernel",
            lambda *args, **kwargs: getter_calls.append((args, kwargs)),
        )
        monkeypatch.setattr(
            _registration,
            "_register_jax_kernel",
            lambda *args, **kwargs: registration_calls.append((args, kwargs)),
        )
        registrations = factory(**kwargs)

        with pytest.raises(ValueError):
            registrations[jnp.float32]

        assert getter_calls == []
        assert registration_calls == []


class TestClusterTileSemanticMaps:
    """Test cluster-tile graph registration semantic maps."""

    @pytest.fixture(autouse=True)
    def _isolate_pair_registration_cache(self, monkeypatch):
        """Keep mocked pair registrations out of the production cache."""
        binding = self._binding("cluster_tile")
        original = binding._get_jax_cluster_tile_pair_fn_registration
        monkeypatch.setattr(
            binding,
            "_get_jax_cluster_tile_pair_fn_registration",
            functools.cache(original.__wrapped__),
        )
        yield

    @staticmethod
    def _binding(module_name: str):
        return importlib.import_module(f"nvalchemiops.jax.neighbors.{module_name}")

    @pytest.mark.parametrize(
        ("module_name", "builds_name", "queries_name"),
        [
            ("cluster_tile", "_CLUSTER_TILE_BUILDS", "_CLUSTER_TILE_QUERIES"),
            (
                "batch_cluster_tile",
                "_BATCH_CLUSTER_TILE_BUILDS",
                "_BATCH_CLUSTER_TILE_QUERIES",
            ),
        ],
    )
    def test_binding_maps_contain_only_graph_registrations(
        self, module_name, builds_name, queries_name
    ) -> None:
        """Expose only bundled callback/preload graph registrations."""
        binding = self._binding(module_name)
        builds = getattr(binding, builds_name)
        queries = getattr(binding, queries_name)

        assert set(builds) == {"full", "selective"}
        assert set(queries) == {
            "matrix",
            "matrix_selective",
            "matrix_dual",
            "matrix_dual_selective",
            "matrix_geometry",
            "coo",
            "coo_segmented",
        }
        for registration in (*builds.values(), *queries.values()):
            assert isinstance(registration, _registration._GraphRegistration)
            assert callable(registration.callable)
            assert callable(registration.preload)

    def test_pair_fn_helper_caches_complete_bundles_by_identity(
        self, monkeypatch
    ) -> None:
        """Distinct pair_fn identities cache separate graph bundles."""
        binding = self._binding("cluster_tile")
        preload_calls = []
        monkeypatch.setattr(
            _registration,
            "_register_jax_callable",
            lambda *args, **kwargs: object(),
        )
        monkeypatch.setattr(
            _registration,
            "_preload_cluster_tile_query_kernel",
            lambda **kwargs: preload_calls.append(kwargs),
        )
        pair_one, pair_two = object(), object()

        first = binding._get_jax_cluster_tile_pair_fn_registration(pair_one)
        second = binding._get_jax_cluster_tile_pair_fn_registration(pair_one)
        third = binding._get_jax_cluster_tile_pair_fn_registration(pair_two)

        assert first is second
        assert third is not first
        first.preload()
        assert preload_calls[-1]["pair_fn"] is pair_one

    @staticmethod
    def _mock_registration_map(keys, preload_calls, callable_calls):
        """Build a map of lightweight graph registrations for dispatch tests."""

        def make_registration(key):
            if key == "full":

                def build_callable(*args):
                    callable_calls.append(key)
                    return args[5:14]

            elif key == "selective":

                def build_callable(*args):
                    callable_calls.append(key)
                    return args[5:14]

            elif key in {"coo", "coo_segmented"}:

                def build_callable(*args, key=key):
                    callable_calls.append(key)
                    if key == "coo_segmented":
                        return args[10], args[12], args[13], args[14]
                    return args[9], args[10], args[11]

            else:

                def build_callable(*args, key=key):
                    callable_calls.append(key)
                    if "dual" in key:
                        return tuple(jnp.zeros(1, dtype=jnp.int32) for _ in range(6))
                    if key == "matrix_geometry":
                        return tuple(jnp.zeros(1, dtype=jnp.float32) for _ in range(5))
                    return tuple(jnp.zeros(1, dtype=jnp.int32) for _ in range(3))

            return _registration._GraphRegistration(
                callable=build_callable,
                preload=lambda *args, key=key, **kwargs: preload_calls.append(key),
            )

        return {key: make_registration(key) for key in keys}

    def test_build_dispatch_uses_matching_bundle(self, monkeypatch) -> None:
        """Build dispatch preloads and calls the selected bundle only."""
        binding = self._binding("cluster_tile")
        preload_calls = []
        callable_calls = []
        original_sort = binding._morton_sort_and_gather
        original_builds = binding._CLUSTER_TILE_BUILDS
        with monkeypatch.context() as scoped:
            scoped.setattr(
                binding,
                "_morton_sort_and_gather",
                lambda *args, **kwargs: (
                    jnp.zeros(1, jnp.int32),
                    jnp.zeros(1, jnp.int32),
                    jnp.zeros(1, jnp.float32),
                    jnp.zeros(1, jnp.float32),
                    jnp.zeros(1, jnp.float32),
                ),
            )
            scoped.setattr(
                binding,
                "_CLUSTER_TILE_BUILDS",
                self._mock_registration_map(
                    ("full", "selective"),
                    preload_calls,
                    callable_calls,
                ),
            )

            positions = jnp.zeros((1, 3), dtype=jnp.float32)
            cell = jnp.eye(3, dtype=jnp.float32)
            binding.build_cluster_tile_list(positions, 1.0, cell)
            binding.build_cluster_tile_list(
                positions,
                1.0,
                cell,
                rebuild_flags=jnp.ones(1, dtype=jnp.bool_),
            )

            assert preload_calls == ["full", "selective"]
            assert callable_calls == ["full", "selective"]
        assert binding._morton_sort_and_gather is original_sort
        assert binding._CLUSTER_TILE_BUILDS is original_builds

    def test_matrix_dispatch_uses_matching_bundle(self, monkeypatch) -> None:
        """Matrix dispatch selects one bundled registration by precedence."""
        binding = self._binding("cluster_tile")
        preload_calls = []
        callable_calls = []
        original_queries = binding._CLUSTER_TILE_QUERIES
        with monkeypatch.context() as scoped:
            scoped.setattr(
                binding,
                "_CLUSTER_TILE_QUERIES",
                self._mock_registration_map(
                    (
                        "matrix",
                        "matrix_selective",
                        "matrix_dual",
                        "matrix_dual_selective",
                        "matrix_geometry",
                        "coo",
                        "coo_segmented",
                    ),
                    preload_calls,
                    callable_calls,
                ),
            )

            args = (
                jnp.zeros(1, jnp.int32),
                jnp.zeros(1, jnp.float32),
                jnp.zeros(1, jnp.float32),
                jnp.zeros(1, jnp.float32),
                jnp.zeros(1, jnp.int32),
                jnp.zeros(1, jnp.int32),
                jnp.zeros(1, jnp.int32),
                jnp.eye(3, dtype=jnp.float32),
            )
            binding.query_cluster_tile(*args, 1.0, 1, 4, cutoff2=2.0)
            binding.query_cluster_tile(
                *args,
                1.0,
                1,
                4,
                rebuild_flags=jnp.ones(1, dtype=jnp.bool_),
            )
            binding.query_cluster_tile(
                *args,
                1.0,
                1,
                4,
                return_vectors=True,
                return_distances=True,
            )

            assert preload_calls == [
                "matrix_dual",
                "matrix_selective",
                "matrix_geometry",
            ]
            assert callable_calls == [
                "matrix_dual",
                "matrix_selective",
                "matrix_geometry",
            ]
        assert binding._CLUSTER_TILE_QUERIES is original_queries

    def test_matrix_dispatch_preloads_the_executing_array_device(
        self, monkeypatch
    ) -> None:
        """Matrix dispatch forwards its executing array to the preload bundle."""
        binding = self._binding("cluster_tile")
        device_sources = []
        registration = _registration._GraphRegistration(
            callable=lambda *args: tuple(
                jnp.zeros(1, dtype=jnp.int32) for _ in range(3)
            ),
            preload=lambda **kwargs: device_sources.append(kwargs["device_source"]),
        )
        original_queries = binding._CLUSTER_TILE_QUERIES
        with monkeypatch.context() as scoped:
            scoped.setattr(binding, "_CLUSTER_TILE_QUERIES", {"matrix": registration})
            args = (
                jnp.zeros(1, jnp.int32),
                jnp.zeros(1, jnp.float32),
                jnp.zeros(1, jnp.float32),
                jnp.zeros(1, jnp.float32),
                jnp.zeros(1, jnp.int32),
                jnp.zeros(1, jnp.int32),
                jnp.zeros(1, jnp.int32),
                jnp.eye(3, dtype=jnp.float32),
            )
            binding.query_cluster_tile(*args, 1.0, 1, 4)

            assert device_sources == [args[1]]
        assert binding._CLUSTER_TILE_QUERIES is original_queries

    def test_coo_dispatch_uses_buffer_segmentation(self, monkeypatch) -> None:
        """COO dispatch keys off paired buffers, not rebuild flags alone."""
        binding = self._binding("cluster_tile")
        preload_calls = []
        callable_calls = []
        original_queries = binding._CLUSTER_TILE_QUERIES
        with monkeypatch.context() as scoped:
            scoped.setattr(
                binding,
                "_CLUSTER_TILE_QUERIES",
                self._mock_registration_map(
                    ("coo", "coo_segmented"),
                    preload_calls,
                    callable_calls,
                ),
            )

            common = (
                jnp.zeros(1, jnp.int32),
                jnp.zeros(1, jnp.float32),
                jnp.zeros(1, jnp.float32),
                jnp.zeros(1, jnp.float32),
                jnp.zeros(1, jnp.int32),
                jnp.zeros(1, jnp.int32),
                jnp.zeros(1, jnp.int32),
                jnp.eye(3, dtype=jnp.float32),
            )
            binding.query_cluster_tile_coo(*common, 1.0, 1, 8)
            binding.query_cluster_tile_coo(
                *common,
                1.0,
                1,
                8,
                pair_offsets=jnp.array([0, 0], dtype=jnp.int32),
                pair_counts=jnp.zeros(1, dtype=jnp.int32),
            )

            assert preload_calls == ["coo", "coo_segmented"]
            assert callable_calls == ["coo", "coo_segmented"]
        assert binding._CLUSTER_TILE_QUERIES is original_queries


class TestGraphRegistrationFactories:
    """Test cluster-tile graph registration bundle factories."""

    def test_registration_reuses_preload_option_validators(self) -> None:
        """Registration factories share the preload module's canonical guards."""
        assert (
            _registration._validate_cluster_tile_build_options
            is _cluster_tile_preload._validate_cluster_tile_build_options
        )
        assert (
            _registration._validate_cluster_tile_matrix_options
            is _cluster_tile_preload._validate_cluster_tile_matrix_options
        )
        assert (
            _registration._validate_cluster_tile_coo_options
            is _cluster_tile_preload._validate_cluster_tile_coo_options
        )

    def test_build_registration_uses_exact_abi_and_warp_graph_mode(
        self, monkeypatch
    ) -> None:
        """Build factories register callbacks with the fixed output ABI."""
        calls = []
        monkeypatch.setattr(
            _registration,
            "_register_jax_callable",
            lambda callback, outputs, *, graph_mode: calls.append(
                (callback, outputs, graph_mode)
            )
            or "jax",
        )
        monkeypatch.setattr(
            _registration,
            "_preload_cluster_tile_build_kernel",
            lambda: None,
        )

        registration = _registration._cluster_tile_build_registration(
            "build-callback",
            batched=True,
            segmented=True,
            selective=True,
        )

        assert registration.callable == "jax"
        assert calls == [
            (
                "build-callback",
                (
                    "group_ctr_x",
                    "group_ctr_y",
                    "group_ctr_z",
                    "group_ext_x",
                    "group_ext_y",
                    "group_ext_z",
                    "num_tiles",
                    "tile_counts",
                    "tile_row_group",
                    "tile_col_group",
                    "tile_system",
                ),
                JaxCallableGraphMode.WARP,
            )
        ]

    def test_matrix_registration_uses_exact_abi_and_warp_graph_mode(
        self, monkeypatch
    ) -> None:
        """Matrix factories preserve dual, geometry, and pair output ordering."""
        calls = []
        monkeypatch.setattr(
            _registration,
            "_register_jax_callable",
            lambda callback, outputs, *, graph_mode: calls.append(
                (callback, outputs, graph_mode)
            )
            or "jax",
        )
        monkeypatch.setattr(
            _registration,
            "_preload_cluster_tile_query_kernel",
            lambda **kwargs: None,
        )
        pair_fn = object()

        dual = _registration._cluster_tile_matrix_registration(
            "dual-callback",
            batched=False,
            dual_cutoff=True,
        )
        geometry_pair = _registration._cluster_tile_matrix_registration(
            "pair-callback",
            batched=False,
            geometry=True,
            pair_fn=pair_fn,
        )

        assert dual.callable == "jax"
        assert geometry_pair.callable == "jax"
        assert calls[0] == (
            "dual-callback",
            (
                "neighbor_matrix",
                "num_neighbors",
                "neighbor_matrix_shifts",
                "neighbor_matrix2",
                "num_neighbors2",
                "neighbor_matrix_shifts2",
            ),
            JaxCallableGraphMode.WARP,
        )
        assert calls[1] == (
            "pair-callback",
            (
                "neighbor_matrix",
                "num_neighbors",
                "neighbor_matrix_shifts",
                "neighbor_vectors",
                "neighbor_distances",
                "pair_energies",
                "pair_forces",
            ),
            JaxCallableGraphMode.WARP,
        )

    def test_coo_registration_uses_exact_abi_and_warp_graph_mode(
        self, monkeypatch
    ) -> None:
        """COO factories preserve compact and segmented output ordering."""
        calls = []
        monkeypatch.setattr(
            _registration,
            "_register_jax_callable",
            lambda callback, outputs, *, graph_mode: calls.append(
                (callback, outputs, graph_mode)
            )
            or "jax",
        )
        monkeypatch.setattr(
            _registration,
            "_preload_cluster_tile_coo_kernel",
            lambda **kwargs: None,
        )

        compact = _registration._cluster_tile_coo_registration(
            "compact-callback",
            batched=False,
        )
        segmented = _registration._cluster_tile_coo_registration(
            "segmented-callback",
            batched=True,
            tile_segmented=True,
            coo_segmented=True,
            selective=True,
        )

        assert compact.callable == "jax"
        assert segmented.callable == "jax"
        assert calls[0] == (
            "compact-callback",
            ("pair_counter", "coo_list", "coo_shifts"),
            JaxCallableGraphMode.WARP,
        )
        assert calls[1] == (
            "segmented-callback",
            ("pair_counter", "pair_counts", "coo_list", "coo_shifts"),
            JaxCallableGraphMode.WARP,
        )

    def test_preload_closures_forward_exact_matrix_options(self, monkeypatch) -> None:
        """Matrix bundles preload with the same validated primitive options."""
        preload_calls = []
        monkeypatch.setattr(
            _registration,
            "_register_jax_callable",
            lambda *args, **kwargs: "jax",
        )
        monkeypatch.setattr(
            _registration,
            "_preload_cluster_tile_query_kernel",
            lambda **kwargs: preload_calls.append(kwargs),
        )
        pair_fn = object()

        registration = _registration._cluster_tile_matrix_registration(
            "matrix-callback",
            batched=True,
            tile_segmented=True,
            selective=False,
            dual_cutoff=False,
            geometry=True,
            pair_fn=pair_fn,
        )
        registration.preload()

        assert preload_calls == [
            {
                "batched": True,
                "tile_segmented": True,
                "selective": False,
                "dual_cutoff": False,
                "geometry": True,
                "pair_fn": pair_fn,
            }
        ]

    def test_preload_closures_forward_exact_coo_options(self, monkeypatch) -> None:
        """COO bundles preload with the same validated primitive options."""
        preload_calls = []
        monkeypatch.setattr(
            _registration,
            "_register_jax_callable",
            lambda *args, **kwargs: "jax",
        )
        monkeypatch.setattr(
            _registration,
            "_preload_cluster_tile_coo_kernel",
            lambda **kwargs: preload_calls.append(kwargs),
        )

        registration = _registration._cluster_tile_coo_registration(
            "coo-callback",
            batched=True,
            tile_segmented=True,
            coo_segmented=True,
            selective=True,
        )
        registration.preload()

        assert preload_calls == [
            {
                "batched": True,
                "tile_segmented": True,
                "coo_segmented": True,
                "selective": True,
            }
        ]

    def test_build_preload_closure_is_no_arg(self, monkeypatch) -> None:
        """Build bundles expose a no-argument preload closure."""
        preload_calls = []
        monkeypatch.setattr(
            _registration,
            "_register_jax_callable",
            lambda *args, **kwargs: "jax",
        )
        monkeypatch.setattr(
            _registration,
            "_preload_cluster_tile_build_kernel",
            lambda: preload_calls.append("called"),
        )

        registration = _registration._cluster_tile_build_registration(
            "build-callback",
            batched=False,
            segmented=False,
            selective=False,
        )
        registration.preload()

        assert preload_calls == ["called"]

    @pytest.mark.parametrize(
        "factory_name, kwargs",
        [
            (
                "_cluster_tile_build_registration",
                {
                    "callback": "callback",
                    "batched": False,
                    "segmented": True,
                    "selective": False,
                },
            ),
            (
                "_cluster_tile_build_registration",
                {
                    "callback": "callback",
                    "batched": True,
                    "segmented": False,
                    "selective": True,
                },
            ),
            (
                "_cluster_tile_matrix_registration",
                {
                    "callback": "callback",
                    "batched": False,
                    "tile_segmented": True,
                },
            ),
            (
                "_cluster_tile_matrix_registration",
                {
                    "callback": "callback",
                    "batched": False,
                    "geometry": True,
                    "selective": True,
                },
            ),
            (
                "_cluster_tile_matrix_registration",
                {
                    "callback": "callback",
                    "batched": False,
                    "dual_cutoff": True,
                    "geometry": True,
                },
            ),
            (
                "_cluster_tile_coo_registration",
                {
                    "callback": "callback",
                    "batched": False,
                    "tile_segmented": True,
                },
            ),
            (
                "_cluster_tile_coo_registration",
                {
                    "callback": "callback",
                    "batched": True,
                    "tile_segmented": True,
                    "coo_segmented": True,
                    "selective": False,
                },
            ),
        ],
    )
    def test_rejects_invalid_factory_options_before_side_effects(
        self, monkeypatch, factory_name, kwargs
    ) -> None:
        """Invalid graph bundles do not register callables or preload kernels."""
        callable_calls = []
        preload_calls = []
        monkeypatch.setattr(
            _registration,
            "_register_jax_callable",
            lambda *args, **kwargs: callable_calls.append((args, kwargs)),
        )
        monkeypatch.setattr(
            _registration,
            "_preload_cluster_tile_build_kernel",
            lambda: preload_calls.append("build"),
        )
        monkeypatch.setattr(
            _registration,
            "_preload_cluster_tile_query_kernel",
            lambda **kwargs: preload_calls.append(kwargs),
        )
        monkeypatch.setattr(
            _registration,
            "_preload_cluster_tile_coo_kernel",
            lambda **kwargs: preload_calls.append(kwargs),
        )

        with pytest.raises(ValueError):
            getattr(_registration, factory_name)(**kwargs)

        assert callable_calls == []
        assert preload_calls == []


class TestClusterTilePreload:
    """Test spec-free cluster-tile preload entry points."""

    @pytest.fixture(autouse=True)
    def _isolate_preload_caches(self, monkeypatch):
        """Keep mocked preload entries out of production preload caches."""
        preload = self._preload_module()
        for cache_name in (
            "_preload_cluster_tile_query_kernel_cached",
            "_preload_cluster_tile_coo_kernel_cached",
        ):
            original = getattr(preload, cache_name)
            monkeypatch.setattr(
                preload, cache_name, functools.cache(original.__wrapped__)
            )
        yield

    @staticmethod
    def _preload_module():
        """Import the cluster-tile preload module under test."""
        return importlib.import_module(
            "nvalchemiops.jax.neighbors._cluster_tile_preload"
        )

    @staticmethod
    def _stub_module_loads(preload, monkeypatch) -> None:
        """Avoid sentinel allocation and Warp module loading in routing tests."""
        monkeypatch.setattr(preload, "empty_sentinel", lambda *args: None)
        monkeypatch.setattr(preload, "_load_kernel_modules", lambda *args: None)

    def test_matrix_preload_uses_unbatched_getter(self, monkeypatch) -> None:
        """Unbatched matrix preload constructs the single-system getter."""
        preload = self._preload_module()
        getter_calls = []
        monkeypatch.setattr(preload, "_current_warp_device_alias", lambda: "cuda:0")
        monkeypatch.setattr(
            preload,
            "get_query_cluster_tile_kernel",
            lambda **kwargs: getter_calls.append(kwargs) or object(),
        )
        monkeypatch.setattr(
            preload,
            "get_batch_query_cluster_tile_kernel",
            lambda **kwargs: getter_calls.append(kwargs) or object(),
        )
        self._stub_module_loads(preload, monkeypatch)

        preload._preload_cluster_tile_query_kernel(
            batched=False,
            tile_segmented=False,
            selective=False,
            dual_cutoff=False,
            geometry=False,
            pair_fn=None,
        )

        assert len(getter_calls) == 1
        assert getter_calls[0] == {
            "tile_segmented": False,
            "selective": False,
            "dual_cutoff": False,
            "return_vectors": False,
            "return_distances": False,
            "pair_fn": None,
        }

    def test_matrix_preload_uses_supplied_execution_device(self, monkeypatch) -> None:
        """Matrix preload targets the device selected by the executing array."""
        preload = self._preload_module()
        jax_device = object()
        device_aliases = []

        class DeviceSource:
            """Minimal JAX-array double with one selected execution device."""

            def devices(self):
                return {jax_device}

        monkeypatch.setattr(
            preload.wp,
            "device_from_jax",
            lambda device: "cuda:1" if device is jax_device else "unexpected",
        )
        monkeypatch.setattr(
            preload,
            "get_query_cluster_tile_kernel",
            lambda **kwargs: object(),
        )
        monkeypatch.setattr(
            preload.wp,
            "get_device",
            lambda alias: device_aliases.append(alias) or object(),
        )
        self._stub_module_loads(preload, monkeypatch)

        preload._preload_cluster_tile_query_kernel(
            batched=False,
            device_source=DeviceSource(),
        )

        assert device_aliases == ["cuda:1"]

    def test_matrix_preload_uses_batched_getter(self, monkeypatch) -> None:
        """Batched matrix preload constructs the batch getter."""
        preload = self._preload_module()
        getter_calls = []
        monkeypatch.setattr(preload, "_current_warp_device_alias", lambda: "cuda:0")
        monkeypatch.setattr(
            preload,
            "get_query_cluster_tile_kernel",
            lambda **kwargs: getter_calls.append("unbatched") or object(),
        )
        monkeypatch.setattr(
            preload,
            "get_batch_query_cluster_tile_kernel",
            lambda **kwargs: getter_calls.append(kwargs) or object(),
        )
        self._stub_module_loads(preload, monkeypatch)

        preload._preload_cluster_tile_query_kernel(
            batched=True,
            tile_segmented=True,
            selective=True,
            dual_cutoff=True,
            geometry=False,
            pair_fn=None,
        )

        assert len(getter_calls) == 1
        assert getter_calls[0] == {
            "tile_segmented": True,
            "selective": True,
            "dual_cutoff": True,
            "return_vectors": False,
            "return_distances": False,
            "pair_fn": None,
        }

    def test_coo_preload_uses_matching_getter_and_segmented_build_module(
        self, monkeypatch
    ) -> None:
        """Segmented COO preload loads build modules before the COO getter."""
        preload = self._preload_module()
        side_effects = []
        monkeypatch.setattr(preload, "_current_warp_device_alias", lambda: "cuda:0")
        monkeypatch.setattr(
            preload,
            "_preload_cluster_tile_build_module",
            lambda device_alias: side_effects.append(("build", device_alias)),
        )
        monkeypatch.setattr(
            preload,
            "get_query_cluster_tile_coo_kernel",
            lambda **kwargs: side_effects.append(("getter", kwargs)) or object(),
        )
        monkeypatch.setattr(
            preload,
            "get_batch_query_cluster_tile_coo_kernel",
            lambda **kwargs: side_effects.append(("batch_getter", kwargs)) or object(),
        )
        self._stub_module_loads(preload, monkeypatch)

        preload._preload_cluster_tile_coo_kernel(
            batched=True,
            tile_segmented=True,
            coo_segmented=True,
            selective=True,
        )

        assert side_effects[0] == ("build", "cuda:0")
        assert side_effects[1] == (
            "batch_getter",
            {
                "tile_segmented": True,
                "coo_segmented": True,
                "selective": True,
            },
        )

    def test_compact_coo_preload_uses_unbatched_getter(self, monkeypatch) -> None:
        """Compact COO preload constructs the single-system COO getter."""
        preload = self._preload_module()
        getter_calls = []
        monkeypatch.setattr(preload, "_current_warp_device_alias", lambda: "cuda:0")
        monkeypatch.setattr(
            preload,
            "get_query_cluster_tile_coo_kernel",
            lambda **kwargs: getter_calls.append(kwargs) or object(),
        )
        monkeypatch.setattr(
            preload,
            "get_batch_query_cluster_tile_coo_kernel",
            lambda **kwargs: getter_calls.append(kwargs) or object(),
        )
        self._stub_module_loads(preload, monkeypatch)

        preload._preload_cluster_tile_coo_kernel(
            batched=False,
            tile_segmented=False,
            coo_segmented=False,
            selective=False,
        )

        assert len(getter_calls) == 1
        assert getter_calls[0] == {
            "tile_segmented": False,
            "coo_segmented": False,
            "selective": False,
        }

    def test_build_preload_accepts_no_arguments(self, monkeypatch) -> None:
        """Build preload loads the shared module without option arguments."""
        preload = self._preload_module()
        module_calls = []
        monkeypatch.setattr(
            preload,
            "_preload_cluster_tile_build_module",
            lambda device_alias: module_calls.append(device_alias),
        )
        monkeypatch.setattr(preload, "_current_warp_device_alias", lambda: "cuda:0")

        preload._preload_cluster_tile_build_kernel()

        assert module_calls == ["cuda:0"]

    @pytest.mark.parametrize(
        "preload_name, kwargs",
        [
            (
                "_preload_cluster_tile_query_kernel",
                {
                    "batched": False,
                    "tile_segmented": True,
                },
            ),
            (
                "_preload_cluster_tile_query_kernel",
                {
                    "batched": False,
                    "geometry": True,
                    "selective": True,
                },
            ),
            (
                "_preload_cluster_tile_coo_kernel",
                {
                    "batched": False,
                    "tile_segmented": True,
                },
            ),
            (
                "_preload_cluster_tile_coo_kernel",
                {
                    "batched": True,
                    "tile_segmented": True,
                    "coo_segmented": True,
                    "selective": False,
                },
            ),
        ],
    )
    def test_rejects_invalid_keyword_options_before_side_effects(
        self, monkeypatch, preload_name, kwargs
    ) -> None:
        """Invalid preload keyword options do not touch device or getters."""
        preload = self._preload_module()
        side_effects = []
        monkeypatch.setattr(
            preload,
            "_current_warp_device_alias",
            lambda: side_effects.append("device_alias") or "cpu",
        )
        monkeypatch.setattr(
            preload,
            "get_query_cluster_tile_kernel",
            lambda **kwargs: side_effects.append("matrix_getter"),
        )
        monkeypatch.setattr(
            preload,
            "get_batch_query_cluster_tile_kernel",
            lambda **kwargs: side_effects.append("batch_matrix_getter"),
        )
        monkeypatch.setattr(
            preload,
            "get_query_cluster_tile_coo_kernel",
            lambda **kwargs: side_effects.append("coo_getter"),
        )
        monkeypatch.setattr(
            preload,
            "get_batch_query_cluster_tile_coo_kernel",
            lambda **kwargs: side_effects.append("batch_coo_getter"),
        )
        monkeypatch.setattr(
            preload,
            "_preload_cluster_tile_build_module",
            lambda *args: side_effects.append("build_module"),
        )
        monkeypatch.setattr(
            preload.wp,
            "get_device",
            lambda *args: side_effects.append("get_device"),
        )
        monkeypatch.setattr(
            preload,
            "empty_sentinel",
            lambda *args: side_effects.append("sentinel"),
        )
        monkeypatch.setattr(
            preload,
            "_load_kernel_modules",
            lambda *args: side_effects.append("module_load"),
        )

        with pytest.raises(ValueError):
            getattr(preload, preload_name)(**kwargs)

        assert side_effects == []
