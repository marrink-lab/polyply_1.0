"""MkDocs build hook.

Two things happen here:

* The repository's root ``LIBRARY.md`` is injected into the "Polymer Library" docs page
  so that the page is always in sync with the file shipped in the repo (single source
  of truth). This applies only to that one page.
* On *every* page, relative links that point at parameter files in the repo
  (e.g. ``polyply/data/martini3/PS.martini3.ff``) are rewritten to absolute GitHub URLs
  so they resolve on the published documentation site -- whether they come from the
  injected library listing or from a tutorial that references a data file directly.

The links are pinned to the git revision the docs are built from, so a parameter
file always resolves to the version that the page was generated against (rather than
whatever currently happens to be on ``master``). The revision is taken from, in order:

1. ``$POLYPLY_DOCS_REF`` / ``$GITHUB_SHA`` (set this in CI to control the target),
2. the exact tag at ``HEAD`` (nice URLs for release builds), else the commit SHA,
3. ``master`` as a last resort (e.g. building from a tarball with no git history).
"""

import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path

# Page that hosts the library listing and the placeholder it contains.
TARGET_PAGE = "reference/polymer-library.md"
PLACEHOLDER = "{{ LIBRARY }}"

# Comment character in the .ff / .itp files (everything after it is ignored).
COMMENT_CHAR = ";"

# Matches markdown links whose target is a path inside the polyply package data,
# with an optional "#mol=NAME" fragment used to deep-link to one moleculetype
# inside a file that bundles several (e.g. vinyl_polymers.ff).
_REL_LINK = re.compile(r"\]\((polyply/data/[^)#]+)(?:#mol=([^)]+))?\)")


@lru_cache(maxsize=None)
def _source_ref():
    """Return the git revision the docs are being built from (computed once)."""
    for var in ("POLYPLY_DOCS_REF", "GITHUB_SHA"):
        ref = os.environ.get(var)
        if ref:
            return ref
    try:
        # Exact tag at HEAD, if any -> readable URLs for release builds.
        return subprocess.check_output(
            ["git", "describe", "--tags", "--exact-match"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        pass
    try:
        # Otherwise pin to the exact commit the docs were generated from.
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "master"


def _moleculetype_line(ff_path, name):
    """1-based line of the ``[ moleculetype ]`` header for *name*, or ``None``.

    Returns the header line so the anchor lands on the ``[ moleculetype ]``
    directive that starts the block; the molecule name is on the next
    (non-comment) line. Used to turn ``vinyl_polymers.ff#mol=STYR`` into a
    ``#L<line>`` deep link. Because the surrounding links are pinned to the build
    revision, the resolved line number stays valid for that published page.

    Inline comments are stripped first, so a ``[ moleculetype ] ; note`` header
    or a ``STYR 3 ; note`` name line still match.
    """
    try:
        lines = ff_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    header_lineno = None
    for lineno, raw in enumerate(lines, start=1):
        code = raw.split(COMMENT_CHAR, 1)[0].strip()  # drop any inline comment
        if code.lower().replace(" ", "") == "[moleculetype]":
            header_lineno = lineno
            continue
        if header_lineno is not None:
            if not code:
                continue  # skip blanks/comment-only lines between header and name
            if code.split()[0] == name:
                return header_lineno
            header_lineno = None  # a different molecule; keep looking
    return None


def on_page_markdown(markdown, page, config, files):
    # Special case for the library page only: it carries a {{ LIBRARY }} placeholder,
    # which we replace with the contents of the repo-root LIBRARY.md (the single source
    # of truth). No other page has this placeholder. The link rewriting below is the
    # general case and still runs for this page too, right after the injection.
    if page.file.src_uri == TARGET_PAGE:
        library_md = Path(config["docs_dir"]).parent / "LIBRARY.md"
        markdown = markdown.replace(PLACEHOLDER, library_md.read_text(encoding="utf-8"))

    # Rewrite repo-relative data-file links to absolute GitHub URLs on *every* page,
    # so a link to e.g. polyply/data/martini3/PS.martini3.ff resolves on the published
    # site no matter which page it appears on (the library listing, a tutorial, ...).
    # Derive the blob base from repo_url so it tracks the source repo even when the
    # docs site itself is published elsewhere (e.g. a different org's GitHub Pages).
    repo_root = Path(config["docs_dir"]).parent
    blob_base = f"{config['repo_url'].rstrip('/')}/blob/{_source_ref()}/"

    def _rewrite(match):
        path, mol = match.group(1), match.group(2)
        anchor = ""
        if mol:
            lineno = _moleculetype_line(repo_root / path, mol)
            if lineno:
                anchor = f"#L{lineno}"
        return f"]({blob_base}{path}{anchor})"

    return _REL_LINK.sub(_rewrite, markdown)
