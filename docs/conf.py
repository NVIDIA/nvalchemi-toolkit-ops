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

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import logging
import multiprocessing
import os
import pathlib
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from importlib.metadata import version
from inspect import signature

import dotenv
from docutils import nodes
from sphinx_gallery.sorting import FileNameSortKey

# -- Load environment vars -----------------------------------------------------
# Defaults build API docs, execute examples, and generate benchmark plots.
dotenv.load_dotenv()
os.environ.setdefault("JAX_ENABLE_X64", "1")

# Examples run one per process (see ``isolate_gallery_examples``), but a single
# JAX example still preallocates a fraction of total device memory on its first
# use and holds it for that process's life: measured at 93.8 GiB of a GB10's
# 121.7 GiB, which is unified with host RAM. The platform allocator drops the
# pool entirely -- JAX allocates and frees through the driver per buffer, so
# Torch's caching allocator is the only one holding memory and the whole gallery
# peaks near 350 MiB. ``test/conftest.py`` does the same for pytest.
#
# Assigned rather than ``setdefault``: a preallocating value inherited from the
# shell is exactly the failure this prevents, so the environment does not get a
# vote. sphinx-multiversion passes this conf directory to every ref it builds,
# so setting it here covers the historical tags too.
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
doc_version = os.getenv(
    "SPHINX_MULTIVERSION_NAME",
    os.getenv("DOC_VERSION", "main"),
)
legacy_plot_gallery = os.getenv("PLOT_GALLERY")
run_examples_value = os.getenv(
    "RUN_EXAMPLES",
    legacy_plot_gallery if legacy_plot_gallery is not None else "True",
)
run_examples = run_examples_value.lower() in ("true", "1", "yes")
run_stale_examples = os.getenv("RUN_STALE_EXAMPLES", "False").lower() in (
    "true",
    "1",
    "yes",
)
filename_pattern = os.getenv("FILENAME_PATTERN", r"/[0-9]+.*\.py")
benchmark_plot_jobs = os.getenv("BENCHMARK_PLOT_JOBS", "auto")
logging.info(
    f"Doc config - version: {doc_version}, run_examples: {run_examples}, "
    f"run_stale: {run_stale_examples}, benchmark_plot_jobs: {benchmark_plot_jobs}"
)
if legacy_plot_gallery is not None and "RUN_EXAMPLES" not in os.environ:
    logging.info("PLOT_GALLERY is kept for compatibility; prefer RUN_EXAMPLES")

root = pathlib.Path(__file__).parent
release = version("nvalchemi-toolkit-ops")

# A historical release can opt into compatible runtime dependencies without
# changing the shared Sphinx environment. For example, v0.2.0 reads
# ``SMV_SITE_PACKAGES_V0_2_0`` when sphinx-multiversion builds that tag.
normalized_doc_version = "".join(
    character if character.isalnum() else "_" for character in doc_version
).upper()
version_site_packages = os.getenv(
    f"SMV_SITE_PACKAGES_{normalized_doc_version}",
)
if version_site_packages:
    sys.path.insert(0, version_site_packages)

# sphinx-multiversion builds tagged source trees with this configuration file.
# Point autodoc at the source tree being rendered rather than the checkout that
# launched the build.
source_root = pathlib.Path(os.getenv("SPHINX_MULTIVERSION_SOURCEDIR", root))
sys.path.insert(0, source_root.parent.as_posix())

# Sphinx does not make its own conf directory importable, and sphinxext.py next
# to this file holds both the gallery reset hook and the entry point for the
# processes that run examples, each reached by import path. Appended rather than
# inserted so that ``docs/benchmarks`` cannot shadow the repository's top-level
# ``benchmarks`` package.
sys.path.append(root.as_posix())

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
version = ".".join(release.split(".")[:2])
project = "ALCHEMI Toolkit-Ops"
copyright = "2025, NVIDIA"
author = "NVIDIA"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx_favicon",
    "myst_parser",
    "sphinx_design",
    "sphinx_togglebutton",
    "sphinx_gallery.gen_gallery",
    "sphinx_multiversion",
]

# Publish the development branch, release-candidate preview branches, and
# stable semantic-version release tags. RC branches follow the repository's
# ``0.4.0-rc`` convention, with optional SemVer-style increments such as
# ``0.4.0-rc.1``.
smv_branch_whitelist = os.getenv(
    "SMV_BRANCH_WHITELIST",
    r"^(main|\d+\.\d+\.\d+-rc(?:\.\d+)?)$",
)
smv_tag_whitelist = os.getenv(
    "SMV_TAG_WHITELIST",
    r"^v\d+\.\d+\.\d+$",
)
# CI fetches non-checked-out branches as ``origin/*`` remote-tracking refs.
smv_remote_whitelist = r"^origin$"
smv_released_pattern = r"^refs/tags/v\d+\.\d+\.\d+$"
smv_outputdir_format = "{ref.name}"
smv_latest_version = "main"

# Sphinx-Gallery intentionally stores a sort-key class in its config. Sphinx
# cannot pickle that value, but the gallery regenerates it deterministically.
suppress_warnings = ["config.cache"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://jax.readthedocs.io/en/latest", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "warp": ("https://nvidia.github.io/warp/latest", None),
}

source_suffix = [".rst", ".md"]
myst_enable_extensions = ["colon_fence", "dollarmath"]
myst_heading_anchors = 3
templates_path = ["_templates"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["templates"]
exclude_patterns = [
    "_build",
    "sphinxext.py",
    "Thumbs.db",
    ".DS_Store",
]
autodoc_typehints = "description"
autodoc_preserve_defaults = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_context = {"default_mode": "light"}
html_static_path = ["_static", "benchmarks/_static"]
html_css_files = [
    "css/nvidia-sphinx-theme.css",
    "css/benchmark-docs.css",
]
html_theme_options = {
    "logo": {
        "text": "ALCHEMI Toolkit-Ops",
        "image_light": "_static/NVIDIA-Logo-V-ForScreen-ForLightBG.png",
        "image_dark": "_static/NVIDIA-Logo-V-ForScreen-ForDarkBG.png",
    },
    "navbar_align": "content",
    "navigation_with_keys": True,
    "navbar_start": [
        "navbar-logo",
        "version-switcher",
    ],
    "external_links": [],
    "icon_links": [
        {
            # Label for this link
            "name": "Github",
            # URL where the link will redirect
            "url": "https://www.github.com/NVIDIA/nvalchemi-toolkit-ops",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fa-brands fa-square-github",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        }
    ],
    "show_toc_level": 2,
}
favicons = ["favicon.ico"]


# -- Gallery example isolation -------------------------------------------------
def isolate_gallery_examples() -> None:
    """Run each gallery example in a process of its own.

    Sphinx-Gallery executes examples in the Sphinx process, so every example
    shares one CUDA context. Mixing the Torch and JAX examples in that context
    corrupts it: a clean build dies part way through with
    ``CUDA_ERROR_ILLEGAL_ADDRESS``. The fault reproduces under every XLA
    allocator setting, survives unloading all Warp modules, and no pair of
    examples triggers it on its own -- it takes the accumulated state of a
    dozen. Reordering the gallery therefore only changes which example dies.
    ``tools/repro_d3_torch_then_jax.py`` has a short version.

    A process per example is already how the rest of the repository runs this
    code: ``weekly-examples.yml`` forks per script and the Makefile forks per
    test group, both citing the same Torch/JAX interaction. It is also the only
    arrangement that does not depend on which examples happen to coexist.

    A pool of one, rebuilt per example, is what makes each call a new
    interpreter rather than a reused worker. ``spawn`` keeps that interpreter
    free of whatever the Sphinx process has already loaded, at the price of
    everything crossing the boundary having to pickle -- which is why
    ``reset_modules`` names its hook by import path instead of passing the
    function.

    A spawned interpreter starts from a bare ``sys.path``, so the entries this
    file adds are handed over as ``PYTHONPATH``. Without the conf directory
    there the hook would not import; without the source root, examples from a
    tag built by sphinx-multiversion would import the installed package rather
    than the tree being rendered.
    """
    from sphinx_gallery import gen_rst
    from sphinxext import run_example

    spawn = multiprocessing.get_context("spawn")
    worker_path = os.pathsep.join(
        entry
        for entry in (
            version_site_packages,
            source_root.parent.as_posix(),
            root.as_posix(),
            os.environ.get("PYTHONPATH"),
        )
        if entry
    )

    def generate_file_rst_isolated(fname, *args, **kwargs):
        original_path = os.environ.get("PYTHONPATH")
        os.environ["PYTHONPATH"] = worker_path
        try:
            with ProcessPoolExecutor(max_workers=1, mp_context=spawn) as executor:
                return executor.submit(run_example, fname, *args, **kwargs).result()
        except BrokenProcessPool as exc:
            # A CUDA abort kills the interpreter outright, so there is no
            # traceback to fold into the example page. Name the example instead
            # of letting a bare pool error reach the user.
            raise RuntimeError(
                f"example {fname} crashed the worker process running it; "
                f"run it directly to see the fault ({exc})"
            ) from exc
        finally:
            if original_path is None:
                del os.environ["PYTHONPATH"]
            else:
                os.environ["PYTHONPATH"] = original_path

    gen_rst.generate_file_rst = generate_file_rst_isolated


# https://sphinx-gallery.github.io/stable/configuration.html
# Multiple galleries: examples and benchmarks
sphinx_gallery_conf = {
    "examples_dirs": ["../examples/"],
    "gallery_dirs": ["examples"],
    # Sphinx-Gallery calls this setting ``plot_gallery``, but it controls
    # whether example scripts execute, including examples that produce no plot.
    "plot_gallery": run_examples,
    "filename_pattern": filename_pattern,
    "ignore_pattern": r"(^_|utils\.py$)",  # Exclude files starting with _ or ending with utils.py
    "image_srcset": ["1x"],
    "run_stale_examples": run_stale_examples,
    "backreferences_dir": "modules/backreferences",
    "doc_module": ("nvalchemiops",),
    # Named by import path, not passed as a function: ``gallery_conf`` is
    # pickled to the process each example runs in. See docs/sphinxext.py.
    "reset_modules": (
        "matplotlib",
        "sphinxext.reset_seeds",
    ),
    "reset_modules_order": "both",
    "show_memory": False,
    "exclude_implicit_doc": {r"load_model", r"load_default_package"},
    "log_level": {"backreference_missing": "warning", "gallery_examples": "debug"},
    # Suppress thumbnail generation warnings for examples without plots
    "thumbnail_size": (250, 250),
    "min_reported_time": 0,
    "capture_repr": ("_repr_html_", "__repr__"),
    # Class ref causes a benign [config.cache] warning; default is NumberOfCodeLinesSortKey.
    "within_subsection_order": FileNameSortKey,
}


# -- Benchmark plot generation ------------------------------------------------
def generate_benchmark_plots(app):
    """Generate benchmark plots at the start of the Sphinx build."""
    from docs.benchmarks.generate_plots import main as generate_plots_main

    if "jobs" in signature(generate_plots_main).parameters:
        generate_plots_main(jobs=benchmark_plot_jobs)
    else:
        generate_plots_main()


def set_multiversion_release(app, config):  # noqa: ARG001
    """Use a tag or RC branch as the displayed documentation version."""
    ref_name = os.getenv("SPHINX_MULTIVERSION_NAME", "")
    if ref_name.startswith("v"):
        release_name = ref_name.removeprefix("v")
    elif re.fullmatch(r"\d+\.\d+\.\d+-rc(?:\.\d+)?", ref_name):
        release_name = ref_name
    else:
        return
    config.release = release_name
    config.version = ".".join(config.release.split(".")[:2])


def set_figure_alt_text(app, doctree, docname):  # noqa: ARG001
    """Use figure captions when Sphinx would expose an image filename as alt text."""
    for figure in doctree.findall(nodes.figure):
        caption = next(
            (child for child in figure.children if isinstance(child, nodes.caption)),
            None,
        )
        if caption is None:
            continue
        description = caption.astext().strip()
        if not description:
            continue
        for image in figure.findall(nodes.image):
            alt = str(image.get("alt", "")).strip()
            image_name = pathlib.Path(str(image.get("uri", ""))).name
            if not alt or (image_name and alt.endswith(image_name)):
                image["alt"] = description


def setup(app):
    """Sphinx setup hook to register event handlers."""
    if run_examples:
        # Before ``builder-inited``, which is when the gallery starts executing.
        isolate_gallery_examples()
    app.connect("config-inited", set_multiversion_release, priority=999)
    app.connect("builder-inited", generate_benchmark_plots)
    app.connect("doctree-resolved", set_figure_alt_text)
    if (source_root / "benchmarks" / "sphinxext.py").is_file():
        from docs.benchmarks.sphinxext import inline_neighborlist_svgs

        app.connect("doctree-resolved", inline_neighborlist_svgs)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
