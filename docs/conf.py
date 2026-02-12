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
import os
import pathlib
import sys
from importlib.metadata import version

import dotenv
from sphinx_gallery.sorting import FileNameSortKey

# -- Load environment vars -----------------------------------------------------
# Note: To override, use environment variables (e.g. PLOT_GALLERY=True make html)
# Defaults will build API docs and execute examples
dotenv.load_dotenv()
doc_version = os.getenv("DOC_VERSION", "main")
plot_gallery = os.getenv("PLOT_GALLERY", "True").lower() in ("true", "1", "yes")
run_stale_examples = os.getenv("RUN_STALE_EXAMPLES", "False").lower() in (
    "true",
    "1",
    "yes",
)
filename_pattern = os.getenv("FILENAME_PATTERN", r"/[0-9]+.*\.py")
logging.info(
    f"Doc config - version: {doc_version}, plot_gallery: {plot_gallery}, run_stale: {run_stale_examples}"
)

root = pathlib.Path(__file__).parent
release = version("nvalchemi-toolkit-ops")

sys.path.insert(0, root.parent.as_posix())
# Add current folder to use sphinxext.py
sys.path.insert(0, os.path.dirname(__file__))

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
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx_favicon",
    "myst_parser",
    "sphinx_design",
    "sphinx_togglebutton",
    "sphinx_gallery.gen_gallery",
]

source_suffix = [".rst", ".md"]
myst_enable_extensions = ["colon_fence", "dollarmath"]
templates_path = ["_templates"]
exclude_patterns = ["_build", "sphinxext.py", "Thumbs.db", ".DS_Store"]
autodoc_typehints = "description"
autodoc_preserve_defaults = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static", "benchmarks/_static"]
html_css_files = [
    "css/nvidia-sphinx-theme.css",
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
    # Uncomment below when you have multiple doc versions deployed
    # "switcher": {
    #     "json_url": "https://your-gitlab-pages-url/versions.json",
    #     "version_match": version,
    # },
}
favicons = ["favicon.ico"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["templates"]

# https://sphinx-gallery.github.io/stable/configuration.html
# Multiple galleries: examples and benchmarks
sphinx_gallery_conf = {
    "examples_dirs": ["../examples/"],
    "gallery_dirs": ["examples"],
    "plot_gallery": plot_gallery,
    "filename_pattern": filename_pattern,
    "ignore_pattern": r"(utils\.py)",  # Exclude utility files and old benchmarks
    "image_srcset": ["1x"],
    "within_subsection_order": FileNameSortKey,
    "run_stale_examples": run_stale_examples,
    "backreferences_dir": "modules/backreferences",
    "doc_module": ("nvalchemiops",),
    "reset_modules": (
        "matplotlib",
        "sphinxext.reset_torch",
    ),
    "reset_modules_order": "both",
    "show_memory": False,
    "exclude_implicit_doc": {r"load_model", r"load_default_package"},
    "log_level": {"backreference_missing": "warning", "gallery_examples": "debug"},
    # Suppress thumbnail generation warnings for examples without plots
    "thumbnail_size": (250, 250),
    "min_reported_time": 0,
    "capture_repr": ("_repr_html_", "__repr__"),
}


# -- Benchmark plot generation ------------------------------------------------
def generate_benchmark_plots(app):
    """Generate benchmark plots at the start of the Sphinx build."""
    from benchmarks.generate_plots import main as generate_plots_main

    generate_plots_main()


def setup(app):
    """Sphinx setup hook to register event handlers."""
    app.connect("builder-inited", generate_benchmark_plots)
