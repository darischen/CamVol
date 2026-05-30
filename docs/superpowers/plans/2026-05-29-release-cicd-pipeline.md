# Release CI/CD Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Auto-build the HandVol Windows installer and publish it to GitHub Releases on every push/merge to `main`, auto-incrementing the patch version.

**Architecture:** A `VERSION` file holds the `MAJOR.MINOR` base. A small pure, unit-tested Python module derives the next `X.Y.Z` from existing git tags (bare `v1.0` counts as patch 0). `build_installer.py` takes a `--version` flag and bakes it into the NSIS installer (DisplayVersion + output filename). A GitHub Actions workflow on `windows-latest` ties it together: compute version → ensure NSIS → build → `gh release create` with auto-generated notes.

**Tech Stack:** Python 3.11, pytest, NSIS (makensis), GitHub Actions, GitHub CLI (`gh`), Git LFS.

**Run tests with:** `python -m pytest tests/ -q` from the repo root (matches existing convention; repo root must be `sys.path[0]` so `handvol` / `installer` packages import).

---

### Task 1: Version computation logic + VERSION file

**Files:**
- Create: `VERSION`
- Create: `installer/__init__.py` (empty — makes `installer` importable as a package so tests can `from installer.compute_version import next_version` regardless of pytest invocation)
- Create: `installer/compute_version.py`
- Test: `tests/test_compute_version.py`

- [ ] **Step 1: Create the VERSION file**

Create `VERSION` with exactly this content (no trailing blank line needed, a trailing newline is fine):

```
1.0
```

- [ ] **Step 2: Create the empty package marker**

Create `installer/__init__.py` as an empty file (0 bytes).

- [ ] **Step 3: Write the failing tests**

Create `tests/test_compute_version.py`:

```python
from installer.compute_version import next_version


def test_bare_base_tag_counts_as_patch_zero():
    # The repo's existing `v1.0` tag is treated as patch 0, so the first
    # auto-release is 1.0.1 (matches the product owner's expectation).
    assert next_version("1.0", ["v1.0"]) == "1.0.1"


def test_increments_above_highest_patch():
    assert next_version("1.0", ["v1.0", "v1.0.1", "v1.0.2"]) == "1.0.3"


def test_patch_compared_numerically_not_lexically():
    assert next_version("1.0", ["v1.0.9", "v1.0.10"]) == "1.0.11"


def test_no_matching_tags_starts_at_zero():
    assert next_version("1.0", []) == "1.0.0"


def test_fresh_minor_base_starts_at_zero():
    # After bumping VERSION to 1.1, with no v1.1.* tags yet, start at .0.
    assert next_version("1.1", ["v1.0", "v1.0.5"]) == "1.1.0"


def test_other_minor_and_major_tags_are_ignored():
    assert next_version("1.0", ["v1.0.3", "v2.0.0", "v1.1.7"]) == "1.0.4"


def test_handles_whitespace_and_blank_entries():
    assert next_version("1.0", [" v1.0.1 ", "", "v1.0"]) == "1.0.2"
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `python -m pytest tests/test_compute_version.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'installer.compute_version'`

- [ ] **Step 5: Implement `compute_version.py`**

Create `installer/compute_version.py`:

```python
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
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_compute_version.py -q`
Expected: PASS (7 passed)

- [ ] **Step 7: Smoke-test the CLI against the real repo**

Run: `python installer/compute_version.py`
Expected output: `1.0.1` (because the repo has the `v1.0` tag and `VERSION` is `1.0`)

- [ ] **Step 8: Run the full suite (no regressions)**

Run: `python -m pytest tests/ -q`
Expected: all green (the new 7 + the existing 22 audio/scrubber/voice tests)

- [ ] **Step 9: Commit**

```bash
git add VERSION installer/__init__.py installer/compute_version.py tests/test_compute_version.py
git commit -m "feat(release): VERSION file + tested next-version computation"
```

---

### Task 2: Version injection into the installer build

**Files:**
- Modify: `build_installer.py` (add `argparse`, a module constant, two pure helpers, refactor `compile_nsis`, thread `--version` through `main`)
- Modify: `installer/handvol-installer.nsi` (default `VERSION` define, versioned `OutFile`, `${VERSION}` DisplayVersion)
- Test: `tests/test_build_installer.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_build_installer.py`:

```python
from build_installer import _makensis_command, _installer_exe_path


def test_makensis_command_includes_version_define():
    cmd = _makensis_command("makensis.exe", "1.0.4")
    assert "/DVERSION=1.0.4" in cmd


def test_makensis_command_includes_proj_root_and_script():
    cmd = _makensis_command("makensis.exe", "1.0.4")
    assert cmd[0] == "makensis.exe"
    assert any(part.startswith("/DPROJ_ROOT=") for part in cmd)
    assert str(cmd[-1]).endswith("handvol-installer.nsi")


def test_installer_exe_path_is_versioned():
    assert _installer_exe_path("1.0.4").name == "HandVol-1.0.4-Installer.exe"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_build_installer.py -q`
Expected: FAIL — `ImportError: cannot import name '_makensis_command' from 'build_installer'`

- [ ] **Step 3: Add the argparse import**

In `build_installer.py`, change the import block at the top (currently starts at line 22):

```python
import argparse
import re
import sys
import shutil
import subprocess
import zipfile
from pathlib import Path
from urllib.request import urlopen
```

- [ ] **Step 4: Add the NSIS_SCRIPT module constant**

In `build_installer.py`, add this next to the other path constants (after the `PYTHON_DIR = BUILD_DIR / "python"` line, ~line 40):

```python
NSIS_SCRIPT = REPO_ROOT / "installer" / "handvol-installer.nsi"
```

- [ ] **Step 5: Add the two pure helpers**

In `build_installer.py`, add these two functions immediately above `def compile_nsis(`:

```python
def _makensis_command(makensis, version):
    """Build the makensis argv. Passing PROJ_ROOT and VERSION as /D defines lets
    the .nsi resolve all paths from the repo root and bake the version in."""
    return [
        makensis,
        f"/DPROJ_ROOT={REPO_ROOT}",
        f"/DVERSION={version}",
        str(NSIS_SCRIPT),
    ]


def _installer_exe_path(version):
    """Path of the versioned installer the .nsi emits (OutFile mirrors this)."""
    return DIST_DIR / f"HandVol-{version}-Installer.exe"
```

- [ ] **Step 6: Refactor `compile_nsis` to take a version and use the helpers**

In `build_installer.py`, replace the entire `compile_nsis` function with:

```python
def compile_nsis(version):
    """Compile the NSIS script into dist/HandVol-<version>-Installer.exe."""
    if not NSIS_SCRIPT.exists():
        log(f"ERR NSIS script not found at {NSIS_SCRIPT}")
        sys.exit(1)

    makensis = shutil.which("makensis")
    if makensis is None:
        log("ERR makensis not found on PATH.")
        log("    Install NSIS 3.x from https://nsis.sourceforge.io/Download")
        log("    and ensure makensis.exe is on PATH, then re-run.")
        sys.exit(1)

    DIST_DIR.mkdir(exist_ok=True)
    log(f"Compiling NSIS script (version {version}) ...")
    result = subprocess.run(
        _makensis_command(makensis, version),
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        log(f"ERR NSIS compilation failed:\n{result.stdout}\n{result.stderr}")
        sys.exit(1)

    exe_path = _installer_exe_path(version)
    if not exe_path.exists():
        log(f"ERR installer not found at {exe_path} after compile")
        sys.exit(1)
    log(f"OK  built {exe_path} ({exe_path.stat().st_size / 1024**2:.1f} MB)")
```

- [ ] **Step 7: Add `parse_args` and thread version through `main`**

In `build_installer.py`, add this function just above `def main():`:

```python
def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the HandVol Windows installer.")
    parser.add_argument(
        "--version",
        default="0.0.0-dev",
        help="Version baked into the installer (DisplayVersion + output "
             "filename). Defaults to 0.0.0-dev for local builds.",
    )
    return parser.parse_args()
```

Then in `main()`, change the first line and the final two lines. The current `main()` begins:

```python
def main():
    log("HandVol Windows Installer Builder")
```

Change it to:

```python
def main():
    args = parse_args()
    log("HandVol Windows Installer Builder")
```

And change the call `compile_nsis()` to:

```python
    compile_nsis(args.version)
```

And change the final log line from:

```python
    log(f"Build complete! Installer: {DIST_DIR / 'HandVol-Installer.exe'}")
```

to:

```python
    log(f"Build complete! Installer: {_installer_exe_path(args.version)}")
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `python -m pytest tests/test_build_installer.py -q`
Expected: PASS (3 passed)

- [ ] **Step 9: Update the NSIS script to consume `${VERSION}`**

In `installer/handvol-installer.nsi`:

(a) After the existing PROJ_ROOT guard block (the `!endif` on line 12), add:

```nsis
!ifndef VERSION
  !define VERSION "0.0.0-dev"
!endif
```

(b) Replace the `OutFile` line (currently line 21):

```nsis
OutFile "${PROJ_ROOT}\dist\HandVol-Installer.exe"
```

with:

```nsis
OutFile "${PROJ_ROOT}\dist\HandVol-${VERSION}-Installer.exe"
```

(c) Replace the DisplayVersion line (currently line 72):

```nsis
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\HandVol" "DisplayVersion" "1.0"
```

with:

```nsis
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\HandVol" "DisplayVersion" "${VERSION}"
```

- [ ] **Step 10: Run the full suite (no regressions)**

Run: `python -m pytest tests/ -q`
Expected: all green.

- [ ] **Step 11: Commit**

```bash
git add build_installer.py installer/handvol-installer.nsi tests/test_build_installer.py
git commit -m "feat(release): bake --version into installer (DisplayVersion + filename)"
```

> **Note on heavyweight end-to-end build:** A real `python build_installer.py --version 9.9.9` run (which downloads embeddable Python, pip-installs mediapipe/opencv, needs `makensis` + an LFS-resolved model) is exercised on the CI runner in Task 3 and during rollout (Task 4). The unit tests above verify the version-wiring without a multi-minute build.

---

### Task 3: GitHub Actions release workflow

**Files:**
- Create: `.github/workflows/release.yml`

- [ ] **Step 1: Create the workflow file**

Create `.github/workflows/release.yml`:

```yaml
name: Release

on:
  push:
    branches: [main]
  workflow_dispatch:
    inputs:
      publish:
        description: "Publish a GitHub Release (false = build artifact only, dry run)"
        type: boolean
        default: false

permissions:
  contents: write

# Serialize releases so two quick merges can't race on the same tag: the second
# run starts only after the first has pushed its tag, so it computes the next
# patch correctly.
concurrency:
  group: release
  cancel-in-progress: false

jobs:
  release:
    # Auto-release on push unless the tip commit opts out with [skip release].
    # Manual (workflow_dispatch) runs always proceed.
    if: ${{ github.event_name == 'workflow_dispatch' || !contains(github.event.head_commit.message, '[skip release]') }}
    runs-on: windows-latest
    steps:
      - name: Checkout (with LFS model + full tag history)
        uses: actions/checkout@v4
        with:
          lfs: true
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Ensure NSIS is available
        shell: pwsh
        run: |
          if (-not (Get-Command makensis -ErrorAction SilentlyContinue)) {
            choco install nsis -y --no-progress
          }
          $nsisDir = "C:\Program Files (x86)\NSIS"
          if (Test-Path $nsisDir) {
            Add-Content -Path $env:GITHUB_PATH -Value $nsisDir
          }

      - name: Compute next version
        id: ver
        shell: bash
        run: |
          VERSION="$(python installer/compute_version.py)"
          echo "version=$VERSION" >> "$GITHUB_OUTPUT"
          echo "Next version: $VERSION"

      - name: Build installer
        shell: bash
        run: python build_installer.py --version "${{ steps.ver.outputs.version }}"

      - name: Publish GitHub Release
        if: ${{ github.event_name == 'push' || inputs.publish }}
        shell: bash
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          VERSION="${{ steps.ver.outputs.version }}"
          gh release create "v${VERSION}" \
            "dist/HandVol-${VERSION}-Installer.exe" \
            --title "HandVol v${VERSION}" \
            --generate-notes

      - name: Upload installer artifact (dry run)
        if: ${{ github.event_name == 'workflow_dispatch' && !inputs.publish }}
        uses: actions/upload-artifact@v4
        with:
          name: HandVol-Installer
          path: dist/HandVol-*-Installer.exe
          if-no-files-found: error
```

- [ ] **Step 2: Validate the YAML parses**

Run (best effort; if PyYAML is missing, `python -m pip install pyyaml` first, or rely on GitHub's parser on push):

```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/release.yml')); print('YAML OK')"
```

Expected: `YAML OK`

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/release.yml
git commit -m "feat(release): GitHub Actions pipeline — patch bump, build, publish"
```

---

### Task 4: Rollout & first-release verification (operational — run by a human)

These steps are not code; they validate the pipeline on the real runner and ship the first release. Do them in order, after Tasks 1–3 are merged-ready.

- [ ] **Step 1: (Recommended) One local end-to-end build sanity check**

With local NSIS installed and the LFS model resolved (`git lfs pull` if needed), run:

```bash
python build_installer.py --version 1.0.1
```

Expected: `dist/HandVol-1.0.1-Installer.exe` is produced. (Optional: confirm its Add/Remove Programs DisplayVersion shows `1.0.1` after a test install.) This is the heavyweight path; skip if you'd rather let CI be the first full build.

- [ ] **Step 2: Land the pipeline on `main` WITHOUT shipping a release yet**

Merge `feat/release-cicd` into `main` with `[skip release]` in the merge commit message, e.g.:

```bash
git checkout main
git merge --no-ff feat/release-cicd -m "Merge release CI/CD pipeline [skip release]"
git push origin main
```

This puts the workflow on the default branch (so `workflow_dispatch` appears in the Actions UI) without auto-publishing.

- [ ] **Step 3: Dry-run the pipeline on the runner**

In GitHub → Actions → "Release" → "Run workflow", leave `publish` = false, run it. Confirm: NSIS installs, LFS model resolves, version computes to `1.0.1`, the build succeeds, and a `HandVol-Installer` artifact (containing `HandVol-1.0.1-Installer.exe`) is uploaded. Download and spot-check it.

- [ ] **Step 4: Ship the first real release**

Either re-run "Run workflow" with `publish` = true, OR push any normal (non-`[skip release]`) commit/merge to `main`. Confirm a `v1.0.1` release appears under Releases with the installer attached and auto-generated notes.

- [ ] **Step 5: From here on**

Every normal merge to `main` ships the next patch automatically. To release a feature minor, edit `VERSION` (e.g. `1.0` → `1.1`) in a PR; the next release becomes `v1.1.0`. Use `[skip release]` on merges that shouldn't ship.

---

## Self-review notes (addressed)

- **Spec coverage:** trigger (Task 3 `on: push`), VERSION-file source of truth (Task 1), bare-`v1.0`→`1.0.1` seeding (Task 1 test + CLI smoke), `[skip release]` guard (Task 3 `if:`), NSIS install-if-missing (Task 3), LFS pull (Task 3 checkout), version injection + versioned asset (Task 2), `gh release --generate-notes` (Task 3), `workflow_dispatch` dry-run (Task 3 + Task 4), concurrency serialization (Task 3). All present.
- **Type/name consistency:** `_makensis_command`, `_installer_exe_path`, `next_version`, output filename `HandVol-${VERSION}-Installer.exe`, tag `v${VERSION}` used consistently across build script, tests, .nsi, and workflow.
- **No placeholders:** every code/step block is concrete.
```
