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

.PHONY: pytest
pytest:  ## Run pytest with coverage
	rm -f .coverage
	uv run pytest --cov-fail-under=0 --cov=nvalchemiops test/test_types.py && \
	uv run pytest --cov-fail-under=0 --cov=nvalchemiops --cov-append test/math && \
	uv run pytest --cov-fail-under=0 --cov=nvalchemiops --cov-append test/neighbors && \
	uv run pytest --cov-fail-under=0 --cov=nvalchemiops --cov-append test/interactions

PYTEST_TESTMON_FLAGS ?= --testmon --testmon-nocollect
TEST_MODULES := types:test/test_types.py math:test/math neighbors:test/neighbors interactions:test/interactions
COVERAGE_DATA_FILES := $(foreach mod,$(TEST_MODULES),.coverage.$(firstword $(subst :, ,$(mod))))
COVERAGE_BASELINE_FILE ?=

.PHONY: testmon-collect
testmon-collect:  ## Build testmon dependency database (no coverage)
	$(foreach mod,$(TEST_MODULES),\
		uv run pytest --testmon $(lastword $(subst :, ,$(mod))) || true;)

.PHONY: testmon-coverage
testmon-coverage:  ## Run tests with coverage (testmon selects by default)
	rm -f .coverage $(COVERAGE_DATA_FILES) $(addsuffix .*, $(COVERAGE_DATA_FILES))
	$(foreach mod,$(TEST_MODULES),\
		COVERAGE_FILE=.coverage.$(firstword $(subst :, ,$(mod))) \
		uv run coverage run -m pytest $(PYTEST_TESTMON_FLAGS) $(lastword $(subst :, ,$(mod))); \
		RET=$$?; if [ $$RET -ne 0 ] && [ $$RET -ne 5 ]; then exit $$RET; fi;) true
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
	if [ -n "$$coverage_files" ]; then \
		uv run coverage combine --data-file=.coverage $$coverage_files || true; \
	else \
		coverage_files=$$(find . -maxdepth 1 -name ".coverage.*" ! -name ".coverage.baseline" -print); \
		if [ -n "$$coverage_files" ]; then \
			uv run coverage combine --data-file=.coverage $$coverage_files || true; \
		fi; \
	fi
	uv run coverage report --show-missing --fail-under=70
	uv run coverage xml -o nvalchemiops.coverage.xml

# ==============================================================================
# COVERAGE
# ==============================================================================

.PHONY: coverage
coverage: pytest
	@echo "Ran coverage"
	rm -f nvalchemiops.coverage.xml; \
	uv run coverage xml --fail-under=70

.PHONY: coverage-html
coverage-html:  ## Generate HTML coverage report
	mkdir htmlcov
	uv run pytest --cov --cov-report=html:htmlcov/index.html test/;
	@echo "Coverage report generated at htmlcov/index.html"

# ==============================================================================
# DOCUMENTATION
# ==============================================================================

.PHONY: docs-install-examples
docs-install-examples:  ## Install example dependencies
	@echo "Installing example dependencies..."
	@for req in examples/*/*-requires.txt; do \
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
	cd docs && make html

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
