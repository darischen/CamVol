# HandVol Release CI/CD Pipeline — Design

Date: 2026-05-29
Status: Approved (design)

## Goal

Automatically build the HandVol Windows installer and publish it to the GitHub
Releases section whenever changes land on `main`, auto-incrementing the **patch**
version (the third number: `1.0.0 → 1.0.1`). Major/minor stay under manual control
because current work is bug fixes only.

## Decisions (locked)

- **Trigger:** push/merge to `main`. Any merge to `main` is a push to `main`, so
  finishing a fix branch and merging it ships a release. One release per merge.
- **Versioning source of truth:** a `VERSION` file at the repo root holding the
  `MAJOR.MINOR` base (e.g. `1.0`). The patch is derived automatically from existing
  release tags. Bumping a feature release = edit `VERSION` to `1.1` in a PR.
- **Runner:** `windows-latest` (build is Windows-only: embeddable Python + NSIS).

## Flow

```
push/merge to main
  -> compute next version (VERSION file + existing git tags)
  -> [skip if commit message contains "[skip release]"]
  -> ensure NSIS present (choco install if makensis not on PATH)
  -> git lfs pull (MediaPipe model)
  -> python build_installer.py --version X.Y.Z
  -> gh release create vX.Y.Z dist/HandVol-X.Y.Z-Installer.exe --generate-notes
```

## Version derivation

Implemented in `installer/compute_version.py` as a **pure, unit-tested function**:
`next_version(base: str, tags: list[str]) -> str`.

Rules:
- `base` comes from the `VERSION` file (e.g. `1.0`).
- Consider tags matching the base: both the bare `v<base>` form and `v<base>.<patch>`.
- Treat a bare `v<base>` tag as **patch 0**.
- `next patch = max(matching patches) + 1`; if no matching tag exists, patch = `0`.
- Output `X.Y.Z` (the workflow prefixes `v` for the tag).

Worked examples (base `1.0`):

| Existing tags            | Next version |
|--------------------------|--------------|
| `v1.0`                   | `1.0.1`      |
| `v1.0`, `v1.0.1`         | `1.0.2`      |
| `v1.0.3`                 | `1.0.4`      |
| (none)                   | `1.0.0`      |
| base `1.1`, no `v1.1.*`  | `1.1.0`      |

The repo currently has exactly one tag, `v1.0`, so the **first auto-release will be
`v1.0.1`** — matching the user's stated expectation without retagging.

## The workflow — `.github/workflows/release.yml`

- `on: push: branches: [main]`
- `permissions: contents: write` (create tags/releases)
- `concurrency:` group `release` with no cancel, so two fast merges serialize and
  cannot race on the same tag.
- Job (`windows-latest`):
  1. `actions/checkout` with `lfs: true` and `fetch-depth: 0` (need full tags).
  2. **Skip guard:** read head commit message; if it contains `[skip release]`, exit
     the job early (success, no release).
  3. `actions/setup-python` (host interpreter to run the orchestrator scripts).
  4. **Ensure NSIS:** if `makensis` is not on PATH, `choco install nsis -y`; then add
     the NSIS dir to PATH for subsequent steps.
  5. Compute version: run `compute_version.py` (reads `VERSION`, lists tags via
     `git tag`), write `X.Y.Z` to `$GITHUB_OUTPUT`.
  6. `python build_installer.py --version X.Y.Z`.
  7. `gh release create vX.Y.Z dist/HandVol-X.Y.Z-Installer.exe --generate-notes`
     (creates + pushes the tag, publishes the release with auto-generated notes from
     merged commits since the last tag, uploads the installer). Uses `GITHUB_TOKEN`.

### Optional manual dry-run (recommended for first run)
Add `workflow_dispatch` with a boolean `publish` input (default `false`). When run
manually with `publish=false`, the job builds and uploads the installer as a workflow
**artifact** instead of publishing a release — lets us validate the build/version path
once before trusting auto-publish.

## Version injection into the build

- `build_installer.py` gains an optional `--version X.Y.Z` argument. It passes
  `/DVERSION=X.Y.Z` to `makensis`.
- `installer/handvol-installer.nsi`:
  - Define a default `VERSION` (e.g. `!ifndef VERSION` → `!define VERSION "0.0.0-dev"`)
    so local manual builds without the flag still compile.
  - Use `${VERSION}` for the Add/Remove Programs `DisplayVersion` (replaces hard-coded
    `"1.0"` at line 72).
  - Produce a versioned output name `HandVol-${VERSION}-Installer.exe` for the release
    asset. (Exact `OutFile` handling — versioned name vs. a copy step — to be finalized
    in the implementation plan; the release must upload a versioned filename.)

## Files

| File | Change |
|------|--------|
| `.github/workflows/release.yml` | NEW — the pipeline |
| `VERSION` | NEW — contains `1.0` |
| `installer/compute_version.py` | NEW — version logic (pure fn + CLI) |
| `tests/test_compute_version.py` | NEW — unit tests for the logic |
| `build_installer.py` | MODIFY — add `--version`, pass `/DVERSION`; update the post-compile expected-output-path check to match the versioned `.exe` name |
| `installer/handvol-installer.nsi` | MODIFY — `${VERSION}` for DisplayVersion + output name, default define |

## Testing strategy

- **Unit tests** (`tests/test_compute_version.py`) for `next_version`, covering every
  row of the table above plus malformed/ignored tags (e.g. `v2.0.0` when base is `1.0`
  must not affect the `1.0` patch sequence).
- **Workflow validation:** the optional `workflow_dispatch` dry-run builds and uploads
  an artifact without publishing, exercising NSIS install, LFS pull, version compute,
  and build end-to-end before the first real release.

## Out of scope (YAGNI / follow-ups)

- Build caching (embeddable Python download + pip wheels). Noted as a future speedup;
  v1 re-downloads each run for simplicity.
- Code signing the installer `.exe`.
- Embedding the version into the running app (tray tooltip, etc.).
- Changelog file generation beyond GitHub's auto-generated release notes.

## Call-outs

- Merging this pipeline to `main` will itself trigger the first release (`v1.0.1`),
  shipping the already-merged audio fix. Tag the merge `[skip release]` to suppress.
- `windows-latest` is ephemeral and has no preinstalled NSIS; the workflow installs it
  (the user's local makensis does not carry over to CI).
