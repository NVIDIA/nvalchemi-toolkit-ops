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

# ==============================================================================
# NValchemi Toolkit Ops - Makefile
# ==============================================================================

.DEFAULT_GOAL := help

UV_DEFAULT_EXTRAS ?= --extra torch --extra jax

# ==============================================================================
# INSTALLATION
# ==============================================================================

.PHONY: install
install:  ## Install the package with default CUDA extras
	uv sync $(UV_DEFAULT_EXTRAS)

.PHONY: setup-ci
setup-ci:  ## Setup CI environment
	uv venv --python 3.12
	uv sync $(UV_DEFAULT_EXTRAS)
	uv run pre-commit install --install-hooks
	uv run pip install -r test/test-requires.txt

# ==============================================================================
# LINTING
# ==============================================================================

.PHONY: lint
lint:  ## Run all linting checks
	uv run pre-commit run check-added-large-files -a
	uv run pre-commit run trailing-whitespace -a
	uv run pre-commit run end-of-file-fixer -a
	uv run pre-commit run debug-statements -a
	uv run pre-commit run pyupgrade -a --show-diff-on-failure
	uv run pre-commit run ruff-check -a --show-diff-on-failure
	uv run pre-commit run ruff-format -a --show-diff-on-failure

.PHONY: lint-fix
lint-fix:  ## Run linting and auto-fix issues
	uv run pre-commit run ruff-check -a --hook-stage manual
	uv run pre-commit run ruff-format -a

.PHONY: format
format:  ## Format code with ruff
	uv run ruff format .
	uv run ruff check --fix .

.PHONY: interrogate
interrogate:  ## Check docstring coverage
	uv run pre-commit run interrogate -a

.PHONY: license
license:  ## Check license headers
	uv run python test/_license/header_check.py

# ==============================================================================
# TESTING
# ==============================================================================

# Every pytest invocation below runs one group in its own process. This is not
# just tidiness: JAX and Torch each reserve GPU memory and do not release it to
# one another cleanly, so a single process importing both exhausts a smaller card.
#
# Groups are declared as a name plus the pytest arguments for that name, rather
# than a single path, so a group can exclude a subtree (see electrostatics below).
FULL_GROUPS := types math neighbors dynamics batch_utils warp_dispatch \
	torch_boundary segment_ops segment_ops_backward segment_ops_torch \
	segment_ops_jax interactions dispersion electrostatics electrostatics_jax \
	electrostatics_torch

ARGS_types                := test/test_types.py
ARGS_math                 := test/math
ARGS_neighbors            := test/neighbors
ARGS_dynamics             := test/dynamics
ARGS_batch_utils          := test/test_batch_utils.py
ARGS_warp_dispatch        := test/test_warp_dispatch.py
ARGS_torch_boundary       := test/torch
ARGS_segment_ops          := test/test_segment_ops.py
ARGS_segment_ops_backward := test/test_segment_ops_backward.py
ARGS_segment_ops_torch    := test/test_segment_ops_torch.py
ARGS_segment_ops_jax      := test/test_segment_ops_jax.py
ARGS_interactions         := test/interactions --ignore=test/interactions/electrostatics --ignore=test/interactions/dispersion
ARGS_dispersion           := test/interactions/dispersion
ARGS_electrostatics       := test/interactions/electrostatics --ignore=test/interactions/electrostatics/bindings
ARGS_electrostatics_jax   := test/interactions/electrostatics/bindings/jax
ARGS_electrostatics_torch := test/interactions/electrostatics/bindings/torch

# ------------------------------------------------------------------------------
# The pull-request suite
# ------------------------------------------------------------------------------
# 56 test files out of 113, chosen by tools/group_coverage.py: at each step it
# takes the file with the best ratio of newly-covered statements to seconds,
# until the selection clears the coverage gate. That reaches 73.4% against a 70%
# threshold, for 36 minutes of the full suite's four and a half hours.
#
# The selection drops the expensive multipole PME and API tests on its own, not
# by hand: they cost minutes each, because Warp compiles a kernel specialised per
# (spline order, l_max, dtype), and they walk lines their cheaper siblings
# already cover. The nightly tier still runs them.
#
# Regenerate with `make minimal-suite-report` when a module gains behaviour the
# selected files do not exercise. Do not hand-edit these lists without re-running
# it — the point of them is that they are measured, not guessed.
MINIMAL_GROUPS ?= types_min math_min neighbors_min dynamics_min warp_dispatch_min \
	segment_ops_min segment_ops_backward_min segment_ops_torch_min \
	segment_ops_jax_min interactions_min dispersion_min electrostatics_min \
	electrostatics_jax_min electrostatics_torch_min

ARGS_types_min                := test/test_types.py
ARGS_warp_dispatch_min        := test/test_warp_dispatch.py
ARGS_segment_ops_min          := test/test_segment_ops.py
ARGS_segment_ops_backward_min := test/test_segment_ops_backward.py
ARGS_segment_ops_torch_min    := test/test_segment_ops_torch.py
ARGS_segment_ops_jax_min      := test/test_segment_ops_jax.py
ARGS_interactions_min         := test/interactions/test_lj.py
ARGS_electrostatics_jax_min   := test/interactions/electrostatics/bindings/jax/test_slab.py

ARGS_math_min := test/math/test_spline.py \
	test/math/test_spherical_harmonics.py \
	test/math/bindings/jax/test_spline.py \
	test/math/bindings/torch/test_autograd.py \
	test/math/bindings/torch/test_gto.py \
	test/math/bindings/torch/test_gto_self_overlap.py \
	test/math/bindings/torch/test_solid_harmonics.py \
	test/math/bindings/torch/test_spline.py

ARGS_dynamics_min := test/dynamics/test_constraints.py \
	test/dynamics/test_fire.py \
	test/dynamics/test_fire2.py \
	test/dynamics/test_utils.py \
	test/dynamics/test_velocity_rescaling.py

ARGS_dispersion_min := test/interactions/dispersion/bindings/jax/test_dftd3.py \
	test/interactions/dispersion/bindings/torch/test_dftd3.py

ARGS_electrostatics_min := test/interactions/electrostatics/test_deriv_check_selftest.py \
	test/interactions/electrostatics/test_jax_autograd.py \
	test/interactions/electrostatics/test_util.py

ARGS_electrostatics_torch_min := test/interactions/electrostatics/bindings/torch/test_gradient_contract.py \
	test/interactions/electrostatics/bindings/torch/test_multipole_coverage_gaps.py \
	test/interactions/electrostatics/bindings/torch/test_multipole_features.py \
	test/interactions/electrostatics/bindings/torch/test_multipole_real_space.py \
	test/interactions/electrostatics/bindings/torch/test_multipole_reciprocal.py \
	test/interactions/electrostatics/bindings/torch/test_slab.py

ARGS_neighbors_min := test/neighbors/test_cell_list_kernel_getters.py \
	test/neighbors/test_cluster_tile_kernel_getters.py \
	test/neighbors/test_cluster_tile_kernels.py \
	test/neighbors/test_compat_imports.py \
	test/neighbors/test_pair_outputs.py \
	test/neighbors/bindings/jax/test_batch_cluster_tile.py \
	test/neighbors/bindings/jax/test_batch_naive.py \
	test/neighbors/bindings/jax/test_cluster_tile.py \
	test/neighbors/bindings/jax/test_naive.py \
	test/neighbors/bindings/jax/test_naive_dual_cutoff.py \
	test/neighbors/bindings/jax/test_neighborlist.py \
	test/neighbors/bindings/jax/test_rebuild_detection.py \
	test/neighbors/bindings/torch/test_base_dispatch.py \
	test/neighbors/bindings/torch/test_batch_cell_list.py \
	test/neighbors/bindings/torch/test_batch_cluster_tile.py \
	test/neighbors/bindings/torch/test_batch_naive.py \
	test/neighbors/bindings/torch/test_batch_naive_dual_cutoff.py \
	test/neighbors/bindings/torch/test_cell_list.py \
	test/neighbors/bindings/torch/test_cell_list_partial.py \
	test/neighbors/bindings/torch/test_cluster_tile.py \
	test/neighbors/bindings/torch/test_naive.py \
	test/neighbors/bindings/torch/test_naive_dual_cutoff.py \
	test/neighbors/bindings/torch/test_neighborlist.py \
	test/neighbors/bindings/torch/test_rebuild_detection.py

# 36 minutes in one job would still be too slow to wait on, so CI splits the
# suite across parallel runners. Groups are packed by measured cost rather than
# spread round-robin, because the costs differ by two orders of magnitude:
# electrostatics_torch alone is 14.5 minutes while types is a rounding error.
# The packing below comes to roughly 14.5, 12.4 and 9.1 minutes on a cold kernel
# cache, so shard 1 sets the pace. Every group still runs in its own process, so
# the JAX/Torch split is intact.
SHARD_COUNT := 3
SHARD_1_GROUPS := electrostatics_torch_min
SHARD_2_GROUPS := neighbors_min math_min electrostatics_min
SHARD_3_GROUPS := electrostatics_jax_min dispersion_min dynamics_min \
	interactions_min types_min warp_dispatch_min segment_ops_min \
	segment_ops_backward_min segment_ops_torch_min segment_ops_jax_min

# pytest-skip-slow skips anything marked `slow` unless --slow is passed, so the
# marker is what separates the two tiers: the pull-request suite leaves this
# empty and the nightly suite sets --slow to run everything. Before this was
# wired up nothing passed --slow, which meant slow-marked tests ran nowhere.
PYTEST_SLOW_FLAG ?=

GROUPS ?= $(FULL_GROUPS)
# Union of both tiers: the minimal tier may define narrower groups of its own,
# and their data files still need cleaning before a run and collecting after it.
COVERAGE_DATA_FILES := $(foreach grp,$(sort $(FULL_GROUPS) $(MINIMAL_GROUPS)),.coverage.$(grp))
COVERAGE_BASELINE_FILE ?=
COVERAGE_FAIL_UNDER ?= 70

# The pull-request suite covers less than the full suite by construction, so it
# gets its own floor. Set just below what the selected groups actually measure,
# so the gate still fails when a change adds code nothing exercises. Verified by
# `make minimal-suite-report`.
MINIMAL_COVERAGE_FAIL_UNDER ?= 70

# Test selection for the impacted tier. `--testmon-nocollect` is load-bearing:
# testmon and coverage.py both install a sys.settrace hook and Python allows one
# per thread, so a run that lets testmon collect reports near-zero coverage.
# With nocollect, testmon still deselects unaffected tests but leaves tracing to
# coverage. The database is built separately by `testmon-collect`.
PYTEST_TESTMON_FLAGS ?=

# pytest exits 5 when a group collects no tests, which is a legitimate outcome
# both for a filtered run and for a testmon run where nothing was affected.
# Only the impacted tier excludes the minimal files; every other tier must run
# them. Applying the ignores unconditionally would make `test-full` skip all 56
# of them, and for the groups whose whole path is already minimal it would
# collect nothing at all — an empty run that exit code 5 then hides.
SKIP_MINIMAL_FILES ?=

define run_group
	COVERAGE_FILE=.coverage.$(1) \
	uv run coverage run -m pytest $(ARGS_$(1)) $(PYTEST_SLOW_FLAG) \
		$(PYTEST_TESTMON_FLAGS) $(if $(SKIP_MINIMAL_FILES),$(IGNORE_$(1))); \
	RET=$$?; if [ $$RET -ne 0 ] && [ $$RET -ne 5 ]; then exit $$RET; fi;
endef

.PHONY: testmon-collect
testmon-collect:  ## Build the testmon database (never combine this with coverage)
	@# testmon and coverage.py both install a sys.settrace hook and Python allows
	@# only one, so a run that collects for testmon reports near-zero coverage.
	@# This target therefore runs on its own, without `coverage run`.
	$(foreach grp,$(FULL_GROUPS),\
		uv run pytest --testmon $(ARGS_$(grp)); \
		RET=$$?; if [ $$RET -ne 0 ] && [ $$RET -ne 5 ]; then exit $$RET; fi;) true

.PHONY: coverage-run
coverage-run:  ## Run $(GROUPS) under coverage, one process per group
	rm -f .coverage $(foreach grp,$(GROUPS),.coverage.$(grp) .coverage.$(grp).*)
	$(foreach grp,$(GROUPS),$(call run_group,$(grp))) true

# Combining shard data needs only coverage and the source tree, so CI runs that
# step on a cheap runner with `UV_RUN="uv run --no-project --with coverage"`
# rather than installing Torch and JAX to read some SQLite files.
UV_RUN ?= uv run

.PHONY: coverage-combine
coverage-combine:  ## Merge per-group coverage data and enforce the gate
	@# `concurrency = ["multiprocessing", "thread"]` puts coverage in parallel
	@# mode, so each group writes .coverage.<group>.<host>.<pid>.<random> rather
	@# than the bare name given in COVERAGE_FILE. Glob for both.
	@coverage_files=""; \
	if [ -n "$(COVERAGE_BASELINE_FILE)" ] && [ -f "$(COVERAGE_BASELINE_FILE)" ]; then \
		coverage_files="$$coverage_files $(COVERAGE_BASELINE_FILE)"; \
	fi; \
	for coverage_prefix in $(COVERAGE_DATA_FILES); do \
		for coverage_file in "$$coverage_prefix" "$$coverage_prefix".*; do \
			if [ -f "$$coverage_file" ]; then \
				coverage_files="$$coverage_files $$coverage_file"; \
			fi; \
		done; \
	done; \
	if [ -z "$$coverage_files" ]; then \
		echo "no coverage data files found; refusing to report a bogus number"; \
		exit 1; \
	fi; \
	$(UV_RUN) coverage combine --data-file=.coverage $$coverage_files
	$(UV_RUN) coverage report --show-missing --fail-under=$(COVERAGE_FAIL_UNDER)
	$(UV_RUN) coverage xml --fail-under=$(COVERAGE_FAIL_UNDER) -o nvalchemiops.coverage.xml

.PHONY: test-full
test-full:  ## Full suite including slow tests, with coverage (nightly and main)
	$(MAKE) coverage-run GROUPS="$(FULL_GROUPS)" PYTEST_SLOW_FLAG=--slow
	$(MAKE) coverage-combine

.PHONY: pytest
pytest: test-full  ## Alias for test-full, kept for the docs and PR template

.PHONY: test-minimal
test-minimal:  ## Pull-request suite: fewest tests that still clear the gate
	$(MAKE) coverage-run GROUPS="$(MINIMAL_GROUPS)"
	$(MAKE) coverage-combine COVERAGE_FAIL_UNDER=$(MINIMAL_COVERAGE_FAIL_UNDER)

# The impacted tier is a UNION with test-minimal, not a filter of it: the minimal
# suite always runs in full so there is a floor of confidence and a coverage
# guarantee, and this adds whatever else the change actually touched, from
# anywhere in the suite. Files already covered by the minimal tier are ignored
# here so the two do not run the same test twice.
#
# Needs a testmon database restored from cache; with none, testmon selects
# everything and this degrades to a full run. CI therefore skips this tier when
# the database is missing rather than accidentally running the whole suite.
$(foreach grp,$(FULL_GROUPS),\
  $(eval IGNORE_$(grp) := $(foreach path,$(ARGS_$(grp)_min),--ignore=$(path))))

.PHONY: test-impacted
test-impacted:  ## Tests affected by the change, beyond the minimal suite
	$(MAKE) coverage-run GROUPS="$(FULL_GROUPS)" SKIP_MINIMAL_FILES=1 \
		PYTEST_TESTMON_FLAGS="--testmon --testmon-nocollect"
	@echo "Impacted-tier coverage data left for the gate job to combine."

.PHONY: test-minimal-shard
test-minimal-shard:  ## Run one shard of the pull-request suite (SHARD=1..3)
	@# No gate here: a shard covers a fraction of the code by definition, so the
	@# threshold is enforced once, after the shards' data is combined.
	@test -n "$(SHARD)" || { echo "set SHARD=1..$(SHARD_COUNT)"; exit 1; }
	@test -n "$(SHARD_$(SHARD)_GROUPS)" || { echo "no groups for SHARD=$(SHARD)"; exit 1; }
	$(MAKE) coverage-run GROUPS="$(SHARD_$(SHARD)_GROUPS)"

# Re-deriving the suite needs a per-test duration table. Get one from any full
# CI job log:
#   gh api repos/<owner>/<repo>/actions/jobs/<job-id>/logs > job.log
#   uv run python tools/extract_ci_durations.py job.log -o ci_durations.json
.PHONY: minimal-suite-report
minimal-suite-report:  ## Re-derive the minimal suite from per-test coverage
	bash tools/measure_test_coverage.sh
	uv run python tools/group_coverage.py \
		--cov-dir .covctx --durations ci_durations.json \
		--select --granularity file --target $(MINIMAL_COVERAGE_FAIL_UNDER)

# ==============================================================================
# COVERAGE
# ==============================================================================

.PHONY: coverage
coverage: test-full  ## Full suite plus the combined report and XML

.PHONY: coverage-html
coverage-html:  ## Generate HTML coverage report
	mkdir -p htmlcov
	uv run pytest --cov --cov-report=html:htmlcov test/;
	@echo "Coverage report generated at htmlcov/index.html"

# ==============================================================================
# DOCUMENTATION
# ==============================================================================

# The PyData theme requires serial output; benchmark PNGs remain parallel.
DOCS_JOBS ?= 1
BENCHMARK_PLOT_JOBS ?= auto

.PHONY: docs-install-examples
docs-install-examples:  ## Install example dependencies
	@echo "Installing example dependencies..."
	@for req in examples/*-requires.txt examples/*/*-requires.txt; do \
		if [ -f "$$req" ]; then \
			echo "Installing dependencies from $$req"; \
			uv pip install -r "$$req"; \
		fi; \
	done

.PHONY: docs-install-benchmarks
docs-install-benchmarks:  ## Install benchmark dependencies
	@echo "Installing benchmark dependencies..."
	@if [ -f "benchmarks/benchmark-requires.txt" ]; then \
		echo "Installing dependencies from benchmarks/benchmark-requires.txt"; \
		uv pip install -r "benchmarks/benchmark-requires.txt"; \
	fi

.PHONY: docs
docs: docs-install-examples docs-install-benchmarks  ## Build documentation
	cd docs && $(MAKE) html DOCS_JOBS="$(DOCS_JOBS)" BENCHMARK_PLOT_JOBS="$(BENCHMARK_PLOT_JOBS)"

.PHONY: docs-clean
docs-clean:  ## Clean documentation build
	cd docs && make clean
	rm -rf docs/examples/
	rm -rf docs/benchmarks/_static/*.png
	rm -rf benchmarks/*/results/
	rm -rf benchmarks/*/*/results/

.PHONY: docs-rebuild
docs-rebuild: docs-clean docs  ## Clean and rebuild documentation

# ==============================================================================
# BUILD & PACKAGING
# ==============================================================================

.PHONY: build
build:  ## Build wheel package
	uv build

.PHONY: clean
clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage*
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf nvalchemiops.coverage.xml
	rm -rf pytest-junit-results.xml
	rm -rf .testmondata*
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# ==============================================================================
# HELP
# ==============================================================================

.PHONY: help
help:  ## Show this help message
	@echo "NValchemi Toolkit Ops - Available Commands"
	@echo "==========================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
