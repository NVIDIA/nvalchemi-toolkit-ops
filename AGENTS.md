# AGENTS.md - Coding Agent Guidelines for nvalchemi-toolkit-ops

## Project Overview

GPU-accelerated computational chemistry primitives built on **NVIDIA Warp** (`warp-lang`).
Python package (`nvalchemiops`) providing neighbor lists, dispersion corrections, and
electrostatics kernels with optional PyTorch integration. Requires Python >= 3.11, < 3.14.

## Build & Development Commands

Package manager: **uv** (lockfile: `uv.lock`).

```bash
# Install all dependencies (including dev + optional torch)
uv sync --all-extras

# Full CI setup (venv + deps + pre-commit hooks + test extras)
make setup-ci

# Build wheel
uv build
```

## Lint & Format

Ruff handles both linting and formatting. Pre-commit is required for all contributions.

```bash
# Run all lint checks (what CI runs)
make lint

# Auto-fix lint + format issues
make format
# Or equivalently:
uv run ruff format .
uv run ruff check --fix .

# Check docstring coverage (95% threshold)
make interrogate

# Check license headers
make license
```

**Ruff rules** (configured in `pyproject.toml`):

- Selected: `E` (pycodestyle), `F` (pyflakes), `S` (bandit), `I` (isort), `PERF`
- Ignored: `E501` (line length), `S311`
- Auto-fixable: `I` (import sorting)
- Per-file overrides: `F401` ignored in `__init__.py`; `S101` ignored in `test/`;
`E402` ignored in `examples/`

## Testing

Tests require a **GPU** to run. The test suite uses pytest with coverage.

```bash
# Run full test suite with coverage
make pytest
# Or directly:
uv run pytest --cov --cov-report=term-missing:skip-covered test/

# Run a single test file
uv run pytest test/test_types.py -vv

# Run a single test class
uv run pytest test/test_types.py::TestGetWpDtype -vv

# Run a single test method
uv run pytest test/test_types.py::TestGetWpDtype::test_float32 -vv

# Run tests by keyword expression
uv run pytest -k "dispersion" -vv

# Skip slow tests
uv run pytest -m "not slow" test/

# Run only GPU-marked tests
uv run pytest -m "gpu" test/

# Run only last failed tests
uv run pytest --lf test/
```

**Pytest configuration** (`pyproject.toml`):

- Default addopts: `-vv -r xfXs --cov --cov-fail-under=70`
- Custom markers: `slow`, `gpu`
- Coverage fail-under: 75% (coverage.py) / 70% (pytest-cov addopts)

Test directory structure mirrors source: `test/interactions/`, `test/math/`, `test/neighbors/`.
Test files follow the `test_*.py` naming pattern.

## Code Style & Conventions

### File Headers

Every file **must** include the SPDX license header with the correct
license type (the repository as a whole is Apache-2.0) and year span.

Full license can be found in `test/_license/header.txt`.

### Imports

Order (enforced by ruff `I` rule):

1. `from __future__ import annotations` (when needed for forward refs)
2. Standard library (`import math`, `import functools`)
3. Third-party (`import torch`, `import warp as wp`, `import numpy as np`)
4. Local (`from nvalchemiops.types import ...`)

Warp is always aliased: `import warp as wp`.

### Naming

- Functions/methods: `snake_case` (`build_cell_list`, `estimate_max_neighbors`)
- Private/internal helpers: leading underscore (`_coulomb_energy_kernel`, `_resolve_warp_dtype`)
- Classes: `PascalCase` (`OutputSpec`, `WarpAutogradContextManager`)
- Constants: `UPPER_SNAKE_CASE` (`PI`, `NON_GRAD_INPUTS`)
- Variables: `snake_case` (`total_atoms`, `wp_device`)

### Type Annotations

- Use modern Python syntax: `str | None` not `Optional[str]`, `list[str]` not `List[str]`
- Use `from __future__ import annotations` for forward references
- Annotate all public function signatures including return types
- Warp-specific types: `wp.array`, `wp.vec3d`, `wp.float32`, etc.

### Formatting

- 4-space indentation
- Double quotes for strings and docstrings
- Trailing commas in multi-line collections and function args
- Ruff handles all formatting; do not manually override

### Docstrings

**NumPy/SciPy style** docstrings are required on all public functions (95% coverage enforced
by `interrogate`). Include `Parameters`, `Returns`, `Notes`, and `See Also` sections as
appropriate.

```python
def build_cell_list(positions, cutoff, box_vectors):
    """Build a cell list for neighbor searching.

    Parameters
    ----------
    positions : wp.array
        Atomic positions of shape (N, 3).
    cutoff : float
        Cutoff distance for neighbor interactions.
    box_vectors : wp.array
        Periodic box vectors of shape (3, 3).

    Returns
    -------
    wp.array
        Cell list indices.
    """
```

Warp kernel docstrings should include "Thread launch" and "Modifies" notes explaining
parallel execution behavior.

### Error Handling

- Use `raise ValueError(...)` for invalid input (wrong dtype, shape, value)
- Use `raise RuntimeError(...)` for unsupported states or failed operations
- Validate tensor shapes and dtypes at public API entry points
- Handle empty/edge cases early with guard returns (`if total_atoms == 0: return`)

### Module Organization

- `nvalchemiops/interactions/` - Core Warp kernels for interatomic interactions
- `nvalchemiops/torch/` - PyTorch-specific bindings and autograd integration
- `nvalchemiops/jax/` - Jax-specific bindings
- `nvalchemiops/math/` - Math primitives
- `nvalchemiops/neighbors/` - Warp-based neighbor computation

Keep mathematical kernels separate from framework-specific wrappers.

### Exports

Every module defines `__all__` explicitly to control the public API surface.

### Test Conventions

- Group related tests in classes (`class TestGetWpDtype:`)
- Each test method gets a docstring
- Use `pytest.raises` with `match=` for exception testing
- Use `@pytest.mark.parametrize` for data-driven tests
- Mark slow tests with `@pytest.mark.slow`, GPU tests with `@pytest.mark.gpu`

### Commits

**Agents are not allowed to make commits.** Humans should make manual commits
after reviewing changes. Humans must sign off (`git commit -s`) per the
DCO requirement.
