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

"""Sphinx transforms for interactive benchmark figures."""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import TYPE_CHECKING

from docutils import nodes

if TYPE_CHECKING:
    from sphinx.application import Sphinx

__all__ = ["inline_neighborlist_svgs"]

_NL_PLOT_RE = re.compile(
    r"^nl-(?:backend-)?(?:cscl|nh3)-(?:system-size|constant-workload|batch)-scaling"
    r"(?:-jax)?-(?:time|throughput|memory)\.png$"
)
_SVG_ID_RE = re.compile(r'\bid="([^"]+)"')
_SVG_HREF_RE = re.compile(r'(?P<attribute>(?:xlink:)?href)="#(?P<id>[^"]+)"')
_SVG_URL_RE = re.compile(r"url\(#([^)]+)\)")
_CUTOFF_ID_RE = re.compile(r"^nl-cutoff-(6A|15A|25A)-")


def _namespace_svg(svg: str, namespace: str, alt_text: str) -> str:
    """Namespace one SVG's IDs and expose its cutoff groups to page CSS."""
    id_map = {
        identifier: f"{namespace}-{identifier}"
        for identifier in _SVG_ID_RE.findall(svg)
    }

    def replace_id(match: re.Match[str]) -> str:
        identifier = match.group(1)
        cutoff_match = _CUTOFF_ID_RE.match(identifier)
        cutoff_attr = (
            f' data-nl-cutoff="{cutoff_match.group(1)}"' if cutoff_match else ""
        )
        return f'id="{id_map[identifier]}"{cutoff_attr}'

    def replace_href(match: re.Match[str]) -> str:
        identifier = match.group("id")
        replacement = id_map.get(identifier, identifier)
        return f'{match.group("attribute")}="#{replacement}"'

    def replace_url(match: re.Match[str]) -> str:
        identifier = match.group(1)
        return f"url(#{id_map.get(identifier, identifier)})"

    svg = _SVG_ID_RE.sub(replace_id, svg)
    svg = _SVG_HREF_RE.sub(replace_href, svg)
    svg = _SVG_URL_RE.sub(replace_url, svg)
    root_attributes = (
        'class="nl-cutoff-plot" role="img" focusable="false" '
        f'aria-label="{html.escape(alt_text, quote=True)}" '
    )
    return svg.replace("<svg ", f"<svg {root_attributes}", 1)


def _inline_svg_markup(svg_path: Path, alt_text: str) -> str:
    """Return file-safe inline HTML for one generated neighbor-list SVG."""
    source = svg_path.read_text(encoding="utf-8")
    svg_start = source.find("<svg ")
    if svg_start < 0 or "</svg>" not in source:
        raise ValueError(f"Generated plot is not a complete SVG: {svg_path}")

    namespace = re.sub(r"[^A-Za-z0-9_.-]+", "-", svg_path.stem)
    svg = _namespace_svg(source[svg_start:].strip(), namespace, alt_text)
    href = f"../_static/{html.escape(svg_path.name, quote=True)}"
    return f'<a class="nl-cutoff-plot-link" href="{href}">{svg}</a>'


def inline_neighborlist_svgs(
    app: Sphinx,
    doctree: nodes.document,
    docname: str,
) -> None:
    """Inline layered NL plots so cutoff switches also work under ``file://``."""
    if app.builder.format != "html" or docname != "benchmarks/neighborlist":
        return

    static_dir = Path(__file__).parent / "_static"
    for image in list(doctree.findall(nodes.image)):
        image_name = Path(str(image.get("uri", ""))).name
        if not _NL_PLOT_RE.fullmatch(image_name):
            continue
        svg_path = static_dir / f"{Path(image_name).stem}.svg"
        if not svg_path.is_file():
            raise FileNotFoundError(f"Layered neighbor-list plot not found: {svg_path}")
        alt_text = str(image.get("alt", "Neighbor-list benchmark plot"))
        image.replace_self(
            nodes.raw("", _inline_svg_markup(svg_path, alt_text), format="html")
        )
