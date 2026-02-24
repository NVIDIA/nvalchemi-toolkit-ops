import os

import pytest


@pytest.fixture(scope="module", autouse=True)
def set_env_vars():
    """Set JAX specific environment variables"""
    pytest.importorskip("jax")
    if "XLA_PYTHON_CLIENT_PREALLOCATE" in os.environ:
        old_preallocate = os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]
    else:
        old_preallocate = ""
    if "XLA_PYTHON_CLIENT_PREALLOCATE" in os.environ:
        old_allocator = os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]
    else:
        old_allocator = ""
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    yield
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = old_allocator
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = old_preallocate
