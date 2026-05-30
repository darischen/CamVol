#!/usr/bin/env python3
"""Compute the next X.Y.Z release version for HandVol.

The next patch is one above the highest existing patch for the MAJOR.MINOR base
read from the repo-root VERSION file. A bare `v<base>` tag (e.g. `v1.0`) counts
as patch 0; if no tag matches the base, the patch starts at 0 (-> `<base>.0`).

Used by the release workflow: prints the next version (no leading `v`) to stdout.
"""

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def next_version(base, tags):
    """Return the next `X.Y.Z` string for `base` (`MAJOR.MINOR`) given `tags`.

    Only tags for this exact base are considered: the bare `v<base>` form
    (patch 0) and `v<base>.<patch>`. All other tags are ignored.
    """
    base = base.strip()
    bare = re.compile(r"^v" + re.escape(base) + r"$")
    patched = re.compile(r"^v" + re.escape(base) + r"\.(\d+)$")
    patches = []
    for tag in tags:
        tag = tag.strip()
        if not tag:
            continue
        if bare.match(tag):
            patches.append(0)
            continue
        m = patched.match(tag)
        if m:
            patches.append(int(m.group(1)))
    next_patch = max(patches) + 1 if patches else 0
    return f"{base}.{next_patch}"


def _read_base():
    return (REPO_ROOT / "VERSION").read_text(encoding="utf-8").strip()


def _git_tags():
    result = subprocess.run(
        ["git", "tag", "--list"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    return result.stdout.splitlines()


def main():
    print(next_version(_read_base(), _git_tags()))


if __name__ == "__main__":
    sys.exit(main())
