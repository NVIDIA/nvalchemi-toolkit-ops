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

# ==============================================================================
# INSTALLATION
# ==============================================================================

.PHONY: install
install:  ## Install the package with all extras
	uv sync --all-extras

.PHONY: setup-ci
setup-ci:  ## Setup CI environment
	uv venv --python 3.12
	uv sync --all-extras
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
	@if [ -n "$$CI" ]; then \
		uv run pytest --junitxml=pytest.xml --cov --cov-report=xml:nvalchemiops.coverage.xml test; \
	else \
		uv run pytest --cov --cov-report=term-missing:skip-covered test/; \
	fi

# ==============================================================================
# COVERAGE
# ==============================================================================

.PHONY: coverage
coverage: pytest
	@echo "Ran coverage"

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
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# ==============================================================================
# HELP
# ==============================================================================

.PHONY: help
help:  ## Show this help message
	@echo "NValchemi Toolkit Ops - Available Commands"
	@echo "==========================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
