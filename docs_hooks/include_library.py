"""MkDocs build hook.

Injects the repository's root ``LIBRARY.md`` into the "Polymer Library" docs page so
that the page is always in sync with the file shipped in the repo (single source of
truth). While injecting, relative links that point at parameter files in the repo
(e.g. ``polyply/data/martini3/PS.martini3.ff``) are rewritten to absolute GitHub URLs
so they resolve on the published documentation site.
"""

import re
from pathlib import Path

# Page that hosts the library listing and the placeholder it contains.
TARGET_PAGE = "reference/polymer-library.md"
PLACEHOLDER = "{{ LIBRARY }}"

# Base URL for linking to source files in the repository.
GH_BLOB_BASE = "https://github.com/marrink-lab/polyply_1.0/blob/master/"

# Matches markdown links whose target is a path inside the polyply package data.
_REL_LINK = re.compile(r"\]\((polyply/data/[^)]+)\)")


def on_page_markdown(markdown, page, config, files):
    if page.file.src_uri != TARGET_PAGE:
        return markdown

    library_md = Path(config["docs_dir"]).parent / "LIBRARY.md"
    text = library_md.read_text(encoding="utf-8")
    text = _REL_LINK.sub(lambda m: f"]({GH_BLOB_BASE}{m.group(1)})", text)
    return markdown.replace(PLACEHOLDER, text)
