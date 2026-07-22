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

import importlib
import importlib.util

import jax.numpy as jnp
import pytest
import warp as wp
from warp.jax_experimental import GraphMode

from nvalchemiops.jax.neighbors import _registration

_batch_cell_list = importlib.import_module("nvalchemiops.jax.neighbors.batch_cell_list")
_cell_list = importlib.import_module("nvalchemiops.jax.neighbors.cell_list")


def _clear_jax_registration_caches() -> None:
    """Clear registration caches that mocked tests can populate."""
    _registration._get_cell_list_build_jax_kernel.cache_clear()
    _registration._get_cell_list_query_jax_kernel.cache_clear()
    _registration._get_naive_jax_kernel.cache_clear()
    _registration._get_cluster_tile_build_jax_callable.cache_clear()
    _registration._get_cluster_tile_query_jax_callable.cache_clear()
    _cell_list._get_jax_cell_list_pair_outputs_kernel.cache_clear()
    _batch_cell_list._get_jax_batch_cell_list_pair_outputs_kernel.cache_clear()

    for registrations in (
        _cell_list._CELL_LIST_BUILD_REGISTRATIONS,
        _cell_list._CELL_LIST_QUERY_REGISTRATIONS,
        _batch_cell_list._BATCH_CELL_LIST_BUILD_REGISTRATIONS,
        _batch_cell_list._BATCH_CELL_LIST_QUERY_REGISTRATIONS,
    ):
        for registration in registrations.values():
            registration._cache.clear()


@pytest.fixture(autouse=True)
def _clear_jax_registration_caches_after_each_test():
    """Isolate cached registrations from temporary mocks, even on failures."""
    _clear_jax_registration_caches()
    try:
        yield
    finally:
        _clear_jax_registration_caches()


class TestRegistrationHelpers:
    """Test the direct Warp/JAX registration adapters."""

    def test_register_jax_kernel_preserves_output_order_and_disables_backward(
        self, monkeypatch
    ) -> None:
        """Register kernels with the schema's output count and order."""
        calls = []

        def fake_jax_kernel(kernel, **kwargs):
            calls.append(
                (kernel, {**kwargs, "in_out_argnames": list(kwargs["in_out_argnames"])})
            )
            kwargs["in_out_argnames"].pop()
            return "registered-kernel"

        monkeypatch.setattr(_registration, "jax_kernel", fake_jax_kernel)
        schema = _registration._JaxOutputSchema(
            ("neighbor_matrix", "num_neighbors", "neighbor_shifts")
        )

        result = _registration._register_jax_kernel("warp-kernel", schema)

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
        assert schema.in_out_argnames == (
            "neighbor_matrix",
            "num_neighbors",
            "neighbor_shifts",
        )
        assert schema.num_outputs == 3

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
        schema = _registration._JaxOutputSchema(("sorted_positions", "neighbor_matrix"))

        result = _registration._register_jax_callable(
            "warp-callable",
            schema,
            graph_mode=GraphMode.NONE,
        )

        assert result == "registered-callable"
        assert calls == [
            (
                "warp-callable",
                {
                    "num_outputs": 2,
                    "in_out_argnames": ["sorted_positions", "neighbor_matrix"],
                    "graph_mode": GraphMode.NONE,
                },
            )
        ]
        assert schema.in_out_argnames == ("sorted_positions", "neighbor_matrix")


class TestLazyDtypeRegistrations:
    """Test lazy dtype-specific wrapper registration."""

    def test_constructs_and_caches_dtype_registration(self) -> None:
        """Construct each supported dtype once and cache by normalized dtype."""
        built_dtypes = []

        def factory(wp_dtype):
            built_dtypes.append(wp_dtype)
            return "jax:kernel-float32"

        registrations = _registration._LazyDtypeRegistrations(
            factory,
            {jnp.float32: wp.float32},
        )

        first = registrations[jnp.float32]
        second = registrations[jnp.dtype("float32")]

        assert first == "jax:kernel-float32"
        assert second is first
        assert built_dtypes == [wp.float32]

    def test_supports_float64_and_dtype_membership(self) -> None:
        """Recognize supported dtypes and convert float64 before construction."""
        built_dtypes = []

        def factory(wp_dtype):
            built_dtypes.append(wp_dtype)
            return wp_dtype

        registrations = _registration._LazyDtypeRegistrations(
            factory,
            {jnp.float32: wp.float32, jnp.float64: wp.float64},
        )

        assert jnp.float32 in registrations
        assert jnp.dtype("float64") in registrations
        assert jnp.int32 not in registrations
        assert registrations[jnp.float64] == wp.float64
        assert built_dtypes == [wp.float64]

    def test_float32_only_mapping_rejects_unsupported_dtype_before_factory(
        self,
    ) -> None:
        """Raise KeyError before the factory for an undeclared dtype."""
        built_dtypes = []
        registrations = _registration._LazyDtypeRegistrations(
            lambda wp_dtype: built_dtypes.append(wp_dtype),
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
        registrations = _registration._LazyDtypeRegistrations(
            lambda wp_dtype: wp_dtype,
            dtype_map,
        )
        dtype_map[jnp.float64] = wp.float64

        assert jnp.float64 not in registrations

        with pytest.raises(KeyError):
            registrations[jnp.float64]

    def test_cells_per_system_reuses_a_dtype_independent_registration(
        self, monkeypatch
    ) -> None:
        """Share the batch cells-per-system wrapper across floating dtypes."""
        binding = importlib.import_module("nvalchemiops.jax.neighbors.batch_cell_list")
        registrations = []
        try:
            with monkeypatch.context() as scoped_monkeypatch:
                scoped_monkeypatch.setattr(
                    _registration,
                    "_get_cell_list_build_jax_kernel",
                    lambda spec: registrations.append(spec) or object(),
                )
                binding = importlib.reload(binding)
                cells_per_system = binding._BATCH_CELL_LIST_BUILD_REGISTRATIONS[
                    "cells_per_system"
                ]

                assert cells_per_system[jnp.float32] is cells_per_system[jnp.float64]
                assert registrations == [
                    _registration._CellListBuildJaxSpec(
                        "cells_per_system",
                        wp.float32,
                        True,
                    ),
                ]
        finally:
            importlib.reload(binding)


class TestJaxOutputSchema:
    """Test immutable registration output schemas."""

    def test_schema_is_frozen(self) -> None:
        """Reject mutation of a frozen output schema."""
        schema = _registration._JaxOutputSchema(("neighbor_matrix", "num_neighbors"))

        with pytest.raises(AttributeError):
            schema.in_out_argnames = ("replacement",)


class TestNaiveJaxKernelRegistration:
    """Test lazy registration of direct naive JAX kernels."""

    @staticmethod
    def _module():
        """Import the direct naive registration module under test."""
        assert (
            importlib.util.find_spec("nvalchemiops.jax.neighbors._registration")
            is not None
        )
        return _registration

    @staticmethod
    def _spec(module, **kwargs):
        """Build a valid single-cutoff geometry registration spec."""
        defaults = {
            "operation": "single_cutoff",
            "wp_dtype": wp.float32,
            "batched": False,
            "pbc_mode": "none",
            "selective": False,
            "partial": False,
            "half_fill": False,
            "return_vectors": True,
            "return_distances": True,
            "pair_fn": None,
        }
        defaults.update(kwargs)
        return module._NaiveJaxKernelSpec(**defaults)

    def test_registers_exact_single_pbc_pair_schema_and_getter_kwargs(
        self, monkeypatch
    ) -> None:
        """Construct single-cutoff pair kernels with the complete ABI schema."""
        module = self._module()
        module._get_naive_jax_kernel.cache_clear()
        getter_calls = []
        registration_calls = []

        def fake_getter(wp_dtype, **kwargs):
            getter_calls.append((wp_dtype, kwargs))
            return "warp-kernel"

        def fake_register(kernel, schema):
            registration_calls.append((kernel, schema))
            return "jax-kernel"

        monkeypatch.setattr(module, "get_naive_neighbor_matrix_kernel", fake_getter)
        monkeypatch.setattr(module, "_register_jax_kernel", fake_register)
        spec = self._spec(
            module,
            batched=True,
            pbc_mode="wrap_on_entry",
            partial=True,
            half_fill=True,
            pair_fn="pair-fn",
        )

        assert module._get_naive_jax_kernel(spec) == "jax-kernel"
        assert getter_calls == [
            (
                wp.float32,
                {
                    "pbc_mode": "wrap_on_entry",
                    "batched": True,
                    "selective": False,
                    "partial": True,
                    "half_fill": True,
                    "return_vectors": True,
                    "return_distances": True,
                    "pair_fn": "pair-fn",
                },
            )
        ]
        assert registration_calls == [
            (
                "warp-kernel",
                _registration._JaxOutputSchema(
                    (
                        "neighbor_matrix1",
                        "neighbor_matrix_shifts1",
                        "num_neighbors1",
                        "neighbor_vectors",
                        "neighbor_distances",
                        "pair_energies",
                        "pair_forces",
                    )
                ),
            )
        ]

    def test_registers_exact_dual_schema_and_ignores_half_fill(
        self, monkeypatch
    ) -> None:
        """Preserve the dual getter's public half-fill compatibility behavior."""
        module = self._module()
        module._get_naive_jax_kernel.cache_clear()
        getter_calls = []
        registration_calls = []

        def fake_getter(wp_dtype, **kwargs):
            getter_calls.append((wp_dtype, kwargs))
            return "dual-warp-kernel"

        def fake_register(kernel, schema):
            registration_calls.append((kernel, schema))
            return "dual-jax-kernel"

        monkeypatch.setattr(
            module,
            "get_naive_neighbor_matrix_dual_cutoff_kernel",
            fake_getter,
        )
        monkeypatch.setattr(module, "_register_jax_kernel", fake_register)
        spec = self._spec(
            module,
            operation="dual_cutoff",
            wp_dtype=wp.float64,
            pbc_mode="prewrapped",
            return_vectors=False,
            return_distances=False,
        )

        assert module._get_naive_jax_kernel(spec) == "dual-jax-kernel"
        assert getter_calls == [
            (
                wp.float64,
                {
                    "pbc_mode": "prewrapped",
                    "batched": False,
                    "selective": False,
                },
            )
        ]
        assert registration_calls == [
            (
                "dual-warp-kernel",
                _registration._JaxOutputSchema(
                    (
                        "neighbor_matrix1",
                        "neighbor_matrix_shifts1",
                        "num_neighbors1",
                        "neighbor_matrix2",
                        "neighbor_matrix_shifts2",
                        "num_neighbors2",
                    )
                ),
            )
        ]

    def test_caches_equal_specs_and_separates_pair_function_identities(
        self, monkeypatch
    ) -> None:
        """Cache immutable equivalent specs while preserving pair function identity."""
        module = self._module()
        module._get_naive_jax_kernel.cache_clear()
        getter_calls = []

        def fake_getter(wp_dtype, **kwargs):
            getter_calls.append((wp_dtype, kwargs))
            return object()

        monkeypatch.setattr(module, "get_naive_neighbor_matrix_kernel", fake_getter)
        monkeypatch.setattr(
            module, "_register_jax_kernel", lambda kernel, schema: object()
        )
        pair_fn_one = object()
        pair_fn_two = object()
        first = self._spec(module, pair_fn=pair_fn_one)
        equal_first = self._spec(module, pair_fn=pair_fn_one)
        second = self._spec(module, pair_fn=pair_fn_two)

        assert module._get_naive_jax_kernel(first) is module._get_naive_jax_kernel(
            equal_first
        )
        assert module._get_naive_jax_kernel(first) is not module._get_naive_jax_kernel(
            second
        )
        assert len(getter_calls) == 2

    @pytest.mark.parametrize(
        "changes",
        [
            {"return_vectors": True, "return_distances": False},
            {"pair_fn": object(), "return_vectors": False, "return_distances": False},
            {
                "operation": "dual_cutoff",
                "return_vectors": True,
                "return_distances": True,
            },
            {
                "operation": "dual_cutoff",
                "partial": True,
                "return_vectors": False,
                "return_distances": False,
            },
            {
                "operation": "dual_cutoff",
                "half_fill": True,
                "return_vectors": False,
                "return_distances": False,
            },
            {"pbc_mode": "invalid"},
        ],
    )
    def test_rejects_invalid_specs_before_getter_or_registration(
        self, monkeypatch, changes
    ) -> None:
        """Reject unsupported direct registrations without constructing side effects."""
        module = self._module()
        module._get_naive_jax_kernel.cache_clear()
        getter_calls = []
        registration_calls = []
        monkeypatch.setattr(
            module,
            "get_naive_neighbor_matrix_kernel",
            lambda *args, **kwargs: getter_calls.append((args, kwargs)),
        )
        monkeypatch.setattr(
            module,
            "_register_jax_kernel",
            lambda *args, **kwargs: registration_calls.append((args, kwargs)),
        )

        with pytest.raises(ValueError):
            module._get_naive_jax_kernel(self._spec(module, **changes))

        assert getter_calls == []
        assert registration_calls == []


class TestNaiveBindingRegistrations:
    """Test lazy registry wiring in the naive JAX bindings."""

    @staticmethod
    def _import_binding(module_name: str):
        """Import one naive JAX binding module."""
        return importlib.import_module(f"nvalchemiops.jax.neighbors.{module_name}")

    @pytest.mark.parametrize(
        ("module_name", "batched"),
        [
            ("batch_naive", True),
            ("naive_dual_cutoff", False),
            ("batch_naive_dual_cutoff", True),
        ],
    )
    def test_import_does_not_construct_direct_wrappers(
        self, monkeypatch, module_name, batched
    ) -> None:
        """Delay direct registrations until a dtype lookup requests one."""
        calls = []

        def fake_getter(spec):
            calls.append(spec)
            return "lazy-jax-kernel"

        module = self._import_binding(module_name)
        try:
            with monkeypatch.context() as scoped_monkeypatch:
                scoped_monkeypatch.setattr(
                    _registration,
                    "_get_naive_jax_kernel",
                    fake_getter,
                )
                module = importlib.reload(module)

                assert calls == []
                registrations = (
                    module._DIRECT_BATCH_NAIVE_KERNELS
                    if module_name == "batch_naive"
                    else module._DIRECT_NAIVE_DUAL_KERNELS
                    if module_name == "naive_dual_cutoff"
                    else module._DIRECT_BATCH_NAIVE_DUAL_KERNELS
                )
                key = (
                    ("prewrapped", True, False)
                    if module_name == "batch_naive"
                    else ("prewrapped", True)
                )
                assert registrations[key][jnp.float64] == "lazy-jax-kernel"
                assert calls == [
                    _registration._NaiveJaxKernelSpec(
                        operation=(
                            "dual_cutoff"
                            if "dual_cutoff" in module_name
                            else "single_cutoff"
                        ),
                        wp_dtype=wp.float64,
                        batched=batched,
                        pbc_mode="prewrapped",
                        selective=True,
                        partial=False,
                        half_fill=False,
                        return_vectors=False,
                        return_distances=False,
                        pair_fn=None,
                    )
                ]
        finally:
            importlib.reload(module)

    @pytest.mark.parametrize(
        ("module_name", "batched"),
        [
            ("naive_dual_cutoff", False),
            ("batch_naive_dual_cutoff", True),
        ],
    )
    def test_dual_binding_lookup_uses_exact_hard_coded_spec(
        self, monkeypatch, module_name, batched
    ) -> None:
        """Use direct dual-cutoff specs without forwarding public half_fill."""
        calls = []

        def fake_getter(spec):
            calls.append(spec)
            return "dual-jax-kernel"

        with monkeypatch.context() as scoped_monkeypatch:
            scoped_monkeypatch.setattr(
                _registration,
                "_get_naive_jax_kernel",
                fake_getter,
            )
            module = importlib.reload(self._import_binding(module_name))
            registrations = (
                module._DIRECT_NAIVE_DUAL_KERNELS
                if module_name == "naive_dual_cutoff"
                else module._DIRECT_BATCH_NAIVE_DUAL_KERNELS
            )

            assert (
                registrations[("wrap_on_entry", False)][jnp.float32]
                == "dual-jax-kernel"
            )
            assert calls == [
                _registration._NaiveJaxKernelSpec(
                    operation="dual_cutoff",
                    wp_dtype=wp.float32,
                    batched=batched,
                    pbc_mode="wrap_on_entry",
                    selective=False,
                    partial=False,
                    half_fill=False,
                    return_vectors=False,
                    return_distances=False,
                    pair_fn=None,
                )
            ]
        importlib.reload(module)


class TestCellListJaxKernelRegistration:
    """Test lazy registration of direct cell-list JAX kernels."""

    @staticmethod
    def _module():
        """Import the direct cell-list registration module under test."""
        assert (
            importlib.util.find_spec("nvalchemiops.jax.neighbors._registration")
            is not None
        )
        return _registration

    def test_mocked_build_registration_does_not_leak_to_real_binding_lookup(
        self, monkeypatch
    ) -> None:
        """Do not reuse a mocked build wrapper after its patch is restored."""
        module = self._module()
        spec = module._CellListBuildJaxSpec("construct_bin_size", wp.float32, False)

        try:
            with monkeypatch.context() as scoped_monkeypatch:
                scoped_monkeypatch.setattr(
                    module,
                    "get_build_cell_list_kernel",
                    lambda *args, **kwargs: "fake-warp-kernel",
                )
                scoped_monkeypatch.setattr(
                    module,
                    "_register_jax_kernel",
                    lambda *args, **kwargs: "fake-jax-kernel",
                )
                assert module._get_cell_list_build_jax_kernel(spec) == "fake-jax-kernel"
        finally:
            _clear_jax_registration_caches()

        assert (
            _cell_list._CELL_LIST_BUILD_REGISTRATIONS["construct_bin_size"][jnp.float32]
            != "fake-jax-kernel"
        )

    @pytest.mark.parametrize(
        ("module_name", "registry_name", "batched"),
        [
            ("cell_list", "_CELL_LIST_BUILD_REGISTRATIONS", False),
            ("batch_cell_list", "_BATCH_CELL_LIST_BUILD_REGISTRATIONS", True),
        ],
    )
    def test_cell_list_binding_import_defers_direct_registration(
        self, monkeypatch, module_name, registry_name, batched
    ) -> None:
        """Construct direct cell-list wrappers only when their dtype is used."""
        calls = []
        module = importlib.import_module(f"nvalchemiops.jax.neighbors.{module_name}")
        try:
            with monkeypatch.context() as scoped_monkeypatch:
                scoped_monkeypatch.setattr(
                    _registration,
                    "_get_cell_list_build_jax_kernel",
                    lambda spec: calls.append(spec) or "lazy-jax-kernel",
                )
                module = importlib.reload(module)

                assert calls == []
                assert (
                    getattr(module, registry_name)["construct_bin_size"][jnp.float32]
                    == "lazy-jax-kernel"
                )
                assert calls == [
                    _registration._CellListBuildJaxSpec(
                        "construct_bin_size",
                        wp.float32,
                        batched,
                    )
                ]
        finally:
            importlib.reload(module)

    def test_registers_each_build_stage_with_exact_abi(self, monkeypatch) -> None:
        """Register every supported cell-list build ABI independently."""
        module = self._module()
        module._get_cell_list_build_jax_kernel.cache_clear()
        getter_calls = []
        registrations = []
        monkeypatch.setattr(
            module,
            "get_build_cell_list_kernel",
            lambda *args, **kwargs: getter_calls.append((args, kwargs)) or "warp",
        )
        monkeypatch.setattr(
            module,
            "get_cell_list_cells_per_system_kernel",
            lambda: "cells-warp",
        )
        monkeypatch.setattr(
            module,
            "get_gather_positions_and_shifts_kernel",
            lambda wp_dtype: "warp",
        )
        monkeypatch.setattr(
            module,
            "get_cell_list_gather_kernel",
            lambda wp_dtype: "warp",
        )
        monkeypatch.setattr(
            module,
            "_register_jax_kernel",
            lambda kernel, schema: registrations.append((kernel, schema)) or "jax",
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
            spec = module._CellListBuildJaxSpec(stage, wp.float32, batched)
            assert module._get_cell_list_build_jax_kernel(spec) == "jax"
            assert registrations[-1] == (
                "cells-warp" if stage == "cells_per_system" else "warp",
                _registration._JaxOutputSchema(outputs),
            )

        assert getter_calls == [
            (("construct_bin_size", wp.float32), {"batched": False}),
            (("construct_bin_size", wp.float32), {"batched": True}),
            (("count_atoms", wp.float32), {"batched": False}),
            (("count_atoms", wp.float32), {"batched": True}),
            (("bin_atoms", wp.float32), {"batched": False}),
            (("bin_atoms", wp.float32), {"batched": True}),
        ]

    def test_registers_query_getter_flags_and_schema(self, monkeypatch) -> None:
        """Preserve sorted/direct, partial, geometry, and pair output ABI."""
        module = self._module()
        module._get_cell_list_query_jax_kernel.cache_clear()
        calls = []
        registered = []
        monkeypatch.setattr(
            module,
            "get_query_cell_list_kernel",
            lambda *args, **kwargs: calls.append((args, kwargs)) or "warp",
        )
        monkeypatch.setattr(
            module,
            "_register_jax_kernel",
            lambda kernel, schema: registered.append((kernel, schema)) or "jax",
        )
        pair_fn = object()
        spec = module._CellListQueryJaxSpec(
            wp.float64, False, True, True, True, True, True, pair_fn, "sorted"
        )

        assert module._get_cell_list_query_jax_kernel(spec) == "jax"
        assert calls == [
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
            )
        ]
        assert registered == [
            (
                "warp",
                _registration._JaxOutputSchema(
                    (
                        "neighbor_matrix",
                        "neighbor_matrix_shifts",
                        "num_neighbors",
                        "neighbor_vectors",
                        "neighbor_distances",
                        "pair_energies",
                        "pair_forces",
                    )
                ),
            )
        ]

    def test_cache_axes_and_invalid_specs_have_no_side_effects(
        self, monkeypatch
    ) -> None:
        """Cache equivalent specs and reject unsupported combinations early."""
        module = self._module()
        module._get_cell_list_query_jax_kernel.cache_clear()
        getter_calls = []
        monkeypatch.setattr(
            module,
            "get_query_cell_list_kernel",
            lambda *args, **kwargs: getter_calls.append((args, kwargs)) or object(),
        )
        monkeypatch.setattr(module, "_register_jax_kernel", lambda *args: object())
        valid = module._CellListQueryJaxSpec(
            wp.float32, False, True, False, False, False, False, None, "direct"
        )
        assert module._get_cell_list_query_jax_kernel(
            valid
        ) is module._get_cell_list_query_jax_kernel(valid)
        assert len(getter_calls) == 1
        for spec in (
            module._CellListBuildJaxSpec("estimate_sizes", wp.float32, False),
            module._CellListQueryJaxSpec(
                wp.float32, True, True, False, False, False, False, None, "direct"
            ),
            module._CellListQueryJaxSpec(
                wp.float32, False, True, False, False, True, False, None, "sorted"
            ),
            module._CellListQueryJaxSpec(
                wp.float32, False, True, False, False, False, False, object(), "sorted"
            ),
        ):
            with pytest.raises(ValueError):
                (
                    module._get_cell_list_build_jax_kernel(spec)
                    if isinstance(spec, module._CellListBuildJaxSpec)
                    else module._get_cell_list_query_jax_kernel(spec)
                )
        assert len(getter_calls) == 1


class TestClusterTileJaxRegistration:
    """Test shared cluster-tile JAX callback registrations."""

    @staticmethod
    def _query_spec(**kwargs):
        """Build a valid single-system matrix query registration spec."""
        defaults = {
            "batched": False,
            "output_format": "matrix",
            "tile_segmented": False,
            "coo_segmented": False,
            "selective": False,
            "dual_cutoff": False,
            "return_vectors": False,
            "return_distances": False,
            "pair_fn": None,
        }
        defaults.update(kwargs)
        return _registration._ClusterTileQueryJaxSpec(**defaults)

    def test_registers_exact_build_schemas(self, monkeypatch) -> None:
        """Build registrations retain each callback's ordered in-place ABI."""
        _registration._get_cluster_tile_build_jax_callable.cache_clear()
        calls = []
        monkeypatch.setattr(
            _registration,
            "_register_jax_callable",
            lambda callback, schema, *, graph_mode: calls.append(
                (callback, schema, graph_mode)
            )
            or "jax",
        )

        cases = [
            (
                _registration._ClusterTileBuildJaxSpec(False, False, False),
                (
                    "group_ctr_x",
                    "group_ctr_y",
                    "group_ctr_z",
                    "group_ext_x",
                    "group_ext_y",
                    "group_ext_z",
                    "num_tiles",
                    "tile_row_group",
                    "tile_col_group",
                ),
            ),
            (
                _registration._ClusterTileBuildJaxSpec(True, False, False),
                (
                    "group_ctr_x",
                    "group_ctr_y",
                    "group_ctr_z",
                    "group_ext_x",
                    "group_ext_y",
                    "group_ext_z",
                    "num_tiles",
                    "tile_row_group",
                    "tile_col_group",
                    "tile_system",
                ),
            ),
            (
                _registration._ClusterTileBuildJaxSpec(True, True, True),
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
            ),
        ]
        for index, (spec, expected) in enumerate(cases):
            assert (
                _registration._get_cluster_tile_build_jax_callable(spec, index) == "jax"
            )
            assert calls[-1] == (
                index,
                _registration._JaxOutputSchema(expected),
                GraphMode.WARP,
            )

    def test_registers_exact_query_schemas_and_preserves_pair_fn_cache_axis(
        self, monkeypatch
    ) -> None:
        """Query schemas follow matrix/COO ABI and distinct pair functions cache apart."""
        _registration._get_cluster_tile_query_jax_callable.cache_clear()
        calls = []
        monkeypatch.setattr(
            _registration,
            "_register_jax_callable",
            lambda callback, schema, *, graph_mode: calls.append(
                (callback, schema, graph_mode)
            )
            or object(),
        )
        pair_one, pair_two = object(), object()
        matrix = self._query_spec(
            dual_cutoff=True,
        )
        geometry_pair = self._query_spec(
            return_vectors=True,
            return_distances=True,
            pair_fn=pair_one,
        )
        segmented_coo = self._query_spec(
            batched=True,
            output_format="coo",
            tile_segmented=True,
            coo_segmented=True,
            selective=True,
        )

        assert _registration._get_cluster_tile_query_jax_callable(matrix, "matrix")
        assert calls[-1][1] == _registration._JaxOutputSchema(
            (
                "neighbor_matrix",
                "num_neighbors",
                "neighbor_matrix_shifts",
                "neighbor_matrix2",
                "num_neighbors2",
                "neighbor_matrix_shifts2",
            )
        )
        first = _registration._get_cluster_tile_query_jax_callable(
            geometry_pair, "pair-one"
        )
        assert first is _registration._get_cluster_tile_query_jax_callable(
            geometry_pair, "pair-one"
        )
        assert (
            _registration._get_cluster_tile_query_jax_callable(
                self._query_spec(
                    return_vectors=True,
                    return_distances=True,
                    pair_fn=pair_two,
                ),
                "pair-two",
            )
            is not first
        )
        assert _registration._get_cluster_tile_query_jax_callable(segmented_coo, "coo")
        assert calls[-1][1] == _registration._JaxOutputSchema(
            ("pair_counter", "pair_counts", "coo_list", "coo_shifts")
        )
        assert all(call[2] is GraphMode.WARP for call in calls)

    @pytest.mark.parametrize(
        "spec",
        [
            _registration._ClusterTileBuildJaxSpec(False, True, False),
            _registration._ClusterTileBuildJaxSpec(True, False, True),
            _registration._ClusterTileQueryJaxSpec(
                False, "matrix", True, False, False, False, False, False, None
            ),
            _registration._ClusterTileQueryJaxSpec(
                False, "coo", False, False, False, True, False, False, None
            ),
            _registration._ClusterTileQueryJaxSpec(
                False, "coo", False, False, False, False, True, True, None
            ),
            _registration._ClusterTileQueryJaxSpec(
                False, "matrix", False, False, False, True, True, True, None
            ),
        ],
    )
    def test_rejects_invalid_cluster_tile_specs_before_registration(
        self, monkeypatch, spec
    ) -> None:
        """Unsupported combinations do not construct callback registrations."""
        _registration._get_cluster_tile_build_jax_callable.cache_clear()
        _registration._get_cluster_tile_query_jax_callable.cache_clear()
        calls = []
        monkeypatch.setattr(
            _registration,
            "_register_jax_callable",
            lambda *args, **kwargs: calls.append((args, kwargs)),
        )

        with pytest.raises(ValueError):
            if isinstance(spec, _registration._ClusterTileBuildJaxSpec):
                _registration._get_cluster_tile_build_jax_callable(spec, "callback")
            else:
                _registration._get_cluster_tile_query_jax_callable(spec, "callback")
        assert calls == []

    def test_preload_forwards_the_same_query_spec_object(self, monkeypatch) -> None:
        """Query preload keeps the immutable registration spec object intact."""
        preload = importlib.import_module(
            "nvalchemiops.jax.neighbors._cluster_tile_preload"
        )
        matrix_spec = self._query_spec(pair_fn=object())
        coo_spec = self._query_spec(
            batched=True,
            output_format="coo",
            tile_segmented=True,
            coo_segmented=True,
            selective=True,
        )
        matrix_calls = []
        coo_calls = []
        monkeypatch.setattr(preload, "_current_warp_device_alias", lambda: "cuda:0")
        monkeypatch.setattr(
            preload,
            "_preload_cluster_tile_query_kernel_cached",
            lambda device_alias, spec: matrix_calls.append((device_alias, spec)),
        )
        monkeypatch.setattr(
            preload,
            "_preload_cluster_tile_coo_kernel_cached",
            lambda device_alias, spec: coo_calls.append((device_alias, spec)),
        )

        preload._preload_cluster_tile_query_kernel(matrix_spec)
        preload._preload_cluster_tile_coo_kernel(coo_spec)

        assert matrix_calls == [("cuda:0", matrix_spec)]
        assert matrix_calls[0][1] is matrix_spec
        assert coo_calls == [("cuda:0", coo_spec)]
        assert coo_calls[0][1] is coo_spec

    def test_single_build_registration_and_preload_share_specs(
        self, monkeypatch
    ) -> None:
        """Use each single-system build spec object for registration and preload."""
        binding = importlib.import_module("nvalchemiops.jax.neighbors.cluster_tile")
        registrations = []
        preloads = []
        try:
            with monkeypatch.context() as scoped_monkeypatch:
                scoped_monkeypatch.setattr(
                    _registration,
                    "_get_cluster_tile_build_jax_callable",
                    lambda spec, callback: registrations.append(spec) or callback,
                )
                binding = importlib.reload(binding)
                scoped_monkeypatch.setattr(
                    binding,
                    "_preload_cluster_tile_build_kernel",
                    lambda spec: preloads.append(spec),
                )
                scoped_monkeypatch.setattr(
                    binding,
                    "_jax_build_cluster_tile_list",
                    lambda *args: args[5:14],
                )
                scoped_monkeypatch.setattr(
                    binding,
                    "_jax_build_cluster_tile_list_selective",
                    lambda *args: args[5:14],
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

                assert registrations == [
                    binding._CLUSTER_TILE_BUILD_SPEC,
                    binding._CLUSTER_TILE_BUILD_SELECTIVE_SPEC,
                ]
                assert preloads == registrations
                assert preloads[0] is registrations[0]
                assert preloads[1] is registrations[1]
        finally:
            importlib.reload(binding)

    def test_batch_build_registration_and_preload_share_specs(
        self, monkeypatch
    ) -> None:
        """Use each batched build spec object for registration and preload."""
        binding = importlib.import_module(
            "nvalchemiops.jax.neighbors.batch_cluster_tile"
        )
        registrations = []
        preloads = []
        try:
            with monkeypatch.context() as scoped_monkeypatch:
                scoped_monkeypatch.setattr(
                    _registration,
                    "_get_cluster_tile_build_jax_callable",
                    lambda spec, callback: registrations.append(spec) or callback,
                )
                binding = importlib.reload(binding)
                scoped_monkeypatch.setattr(
                    binding,
                    "_preload_cluster_tile_build_kernel",
                    lambda spec: preloads.append(spec),
                )
                scoped_monkeypatch.setattr(
                    binding,
                    "_jax_batch_build_cluster_tile_list",
                    lambda *args: args[7:17],
                )
                scoped_monkeypatch.setattr(
                    binding,
                    "_jax_batch_build_cluster_tile_list_selective",
                    lambda *args: (*args[7:14], args[15], *args[17:20]),
                )

                positions = jnp.zeros((1, 3), dtype=jnp.float32)
                cell_batch = jnp.eye(3, dtype=jnp.float32)[jnp.newaxis, :, :]
                batch_ptr = jnp.array([0, 1], dtype=jnp.int32)
                binding.batch_build_cluster_tile_list(
                    positions,
                    1.0,
                    cell_batch,
                    batch_ptr,
                )
                binding.batch_build_cluster_tile_list(
                    positions,
                    1.0,
                    cell_batch,
                    batch_ptr,
                    rebuild_flags=jnp.ones(1, dtype=jnp.bool_),
                    tile_offsets=jnp.array([0, 1], dtype=jnp.int32),
                    tile_counts=jnp.zeros(1, dtype=jnp.int32),
                )

                assert registrations == [
                    binding._BATCH_CLUSTER_TILE_BUILD_SPEC,
                    binding._BATCH_CLUSTER_TILE_BUILD_SELECTIVE_SPEC,
                ]
                assert preloads == registrations
                assert preloads[0] is registrations[0]
                assert preloads[1] is registrations[1]
        finally:
            importlib.reload(binding)

    @pytest.mark.parametrize(
        ("preload_name", "spec"),
        [
            (
                "_preload_cluster_tile_query_kernel",
                _registration._ClusterTileQueryJaxSpec(
                    False, "coo", False, False, False, False, False, False, None
                ),
            ),
            (
                "_preload_cluster_tile_query_kernel",
                _registration._ClusterTileQueryJaxSpec(
                    False, "matrix", False, True, False, False, False, False, None
                ),
            ),
            (
                "_preload_cluster_tile_query_kernel",
                _registration._ClusterTileQueryJaxSpec(
                    False, "matrix", True, False, False, False, False, False, None
                ),
            ),
            (
                "_preload_cluster_tile_query_kernel",
                _registration._ClusterTileQueryJaxSpec(
                    False, "matrix", False, False, False, False, True, False, None
                ),
            ),
            (
                "_preload_cluster_tile_query_kernel",
                _registration._ClusterTileQueryJaxSpec(
                    False, "matrix", False, False, True, True, True, True, None
                ),
            ),
            (
                "_preload_cluster_tile_coo_kernel",
                _registration._ClusterTileQueryJaxSpec(
                    False, "matrix", False, False, False, False, False, False, None
                ),
            ),
            (
                "_preload_cluster_tile_coo_kernel",
                _registration._ClusterTileQueryJaxSpec(
                    False, "coo", False, False, False, True, False, False, None
                ),
            ),
            (
                "_preload_cluster_tile_coo_kernel",
                _registration._ClusterTileQueryJaxSpec(
                    False, "coo", False, False, False, False, True, True, None
                ),
            ),
            (
                "_preload_cluster_tile_coo_kernel",
                _registration._ClusterTileQueryJaxSpec(
                    True, "coo", True, True, False, False, False, False, None
                ),
            ),
            (
                "_preload_cluster_tile_coo_kernel",
                _registration._ClusterTileQueryJaxSpec(
                    True, "coo", True, False, True, False, False, False, None
                ),
            ),
        ],
    )
    def test_preload_rejects_invalid_query_specs_before_side_effects(
        self, monkeypatch, preload_name, spec
    ) -> None:
        """Invalid preload specs cannot select a device or construct kernels."""
        preload = importlib.import_module(
            "nvalchemiops.jax.neighbors._cluster_tile_preload"
        )
        side_effects = []
        preload._preload_cluster_tile_query_kernel_cached.cache_clear()
        preload._preload_cluster_tile_coo_kernel_cached.cache_clear()
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
            getattr(preload, preload_name)(spec)

        assert side_effects == []
