# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Create compatibility redirects for the former unversioned documentation."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path

REDIRECT_TEMPLATE = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="robots" content="noindex">
    <meta http-equiv="refresh" content="0; url={escaped_target}">
    <link rel="canonical" href="{escaped_target}">
    <title>Documentation moved</title>
    <script>
      window.location.replace(
        {javascript_target} + window.location.search + window.location.hash
      );
    </script>
  </head>
  <body>
    <p><a href="{escaped_target}">Continue to the versioned documentation.</a></p>
  </body>
</html>
"""


def write_redirect(redirect_page: Path, latest_page: Path) -> None:
    """Write one redirect while preserving query strings and fragments."""
    redirect_page.parent.mkdir(parents=True, exist_ok=True)
    target = os.path.relpath(latest_page, redirect_page.parent).replace(os.sep, "/")
    redirect_page.write_text(
        REDIRECT_TEMPLATE.format(
            escaped_target=html.escape(target, quote=True),
            javascript_target=json.dumps(target),
        ),
        encoding="utf-8",
    )


def load_aliases(aliases_path: Path) -> dict[Path, Path]:
    """Load legacy-to-current page mappings from a whitespace-delimited file."""
    aliases: dict[Path, Path] = {}
    for line_number, raw_line in enumerate(
        aliases_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            legacy_path, current_path = line.split()
        except ValueError as error:
            raise ValueError(
                f"Invalid redirect mapping at {aliases_path}:{line_number}"
            ) from error
        aliases[Path(legacy_path)] = Path(current_path)
    return aliases


def create_redirects(
    site_root: Path,
    latest_version: str = "main",
    aliases_path: Path | None = None,
) -> int:
    """Mirror latest-version HTML paths at the site root as redirects.

    Parameters
    ----------
    site_root : pathlib.Path
        Root of the generated GitHub Pages artifact.
    latest_version : str, default="main"
        Directory containing the canonical development documentation.
    aliases_path : pathlib.Path or None, default=None
        Optional file mapping legacy HTML paths to current HTML paths.

    Returns
    -------
    int
        Number of compatibility redirect files created.
    """
    site_root = site_root.resolve()
    latest_root = site_root / latest_version
    if not latest_root.is_dir():
        raise FileNotFoundError(f"Versioned documentation not found: {latest_root}")

    count = 0
    for latest_page in sorted(latest_root.rglob("*.html")):
        relative_page = latest_page.relative_to(latest_root)
        if relative_page == Path("index.html"):
            continue

        redirect_page = site_root / relative_page
        if redirect_page.exists():
            raise FileExistsError(
                f"Redirect would replace generated site content: {redirect_page}"
            )

        write_redirect(redirect_page, latest_page)
        count += 1

    if aliases_path is not None:
        for legacy_path, current_path in load_aliases(aliases_path).items():
            redirect_page = site_root / legacy_path
            latest_page = latest_root / current_path
            if redirect_page.exists():
                continue
            if not latest_page.is_file():
                raise FileNotFoundError(
                    f"Legacy redirect target does not exist: {latest_page}"
                )
            write_redirect(redirect_page, latest_page)
            count += 1

    return count


def main() -> None:
    """Generate unversioned compatibility redirects for a Pages artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site_root", type=Path)
    parser.add_argument("--latest-version", default="main")
    parser.add_argument(
        "--aliases",
        type=Path,
        default=Path(__file__).with_name("legacy_pages_redirects.txt"),
    )
    args = parser.parse_args()

    count = create_redirects(args.site_root, args.latest_version, args.aliases)
    print(f"Created {count} unversioned documentation redirects")


if __name__ == "__main__":
    main()
