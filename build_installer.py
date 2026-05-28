#!/usr/bin/env python3
"""
Build script for the HandVol Windows installer.

Orchestrates, in order:
  1. Download/cache embeddable Python 3.11
  2. Extract it to build/python
  3. Enable site-packages in the embeddable runtime (edit pythonXY._pth)
  4. Bootstrap pip into the embeddable runtime (get-pip.py)
  5. pip install -r requirements.txt into the bundled runtime
  6. Copy the MediaPipe model (from Git LFS) and HandVol source
  7. Compile the NSIS script -> dist/HandVol-Installer.exe

Why steps 3-4 exist (and aren't in a "plain" embeddable build):
The python-x.y.z-embed-amd64.zip distribution intentionally ships WITHOUT pip
and with a `pythonXY._pth` file that disables the normal `site` import and the
`Lib\\site-packages` search path. Without fixing both, `python -m pip` fails and
even hand-copied packages won't import at runtime. So we patch the ._pth first,
then bootstrap pip, then install normally into Lib\\site-packages.
"""

import re
import sys
import shutil
import subprocess
import zipfile
from pathlib import Path
from urllib.request import urlopen

PYTHON_VERSION = "3.11.9"
PYTHON_URL = f"https://www.python.org/ftp/python/{PYTHON_VERSION}/python-{PYTHON_VERSION}-embed-amd64.zip"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"

REPO_ROOT = Path(__file__).resolve().parent
CACHE_DIR = REPO_ROOT / "cache"
PYTHON_CACHE = CACHE_DIR / "python-embed.zip"
GET_PIP_CACHE = CACHE_DIR / "get-pip.py"
BUILD_DIR = REPO_ROOT / "build"
DIST_DIR = REPO_ROOT / "dist"
PYTHON_DIR = BUILD_DIR / "python"


def log(msg):
    print(f"[BUILD] {msg}", file=sys.stderr)


def download(url, dest, label):
    """Download `url` to `dest` unless it is already cached."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        log(f"OK  {label} cached: {dest}")
        return
    log(f"Downloading {label} ...")
    try:
        with urlopen(url) as response, open(dest, "wb") as f:
            shutil.copyfileobj(response, f)
    except Exception as e:  # noqa: BLE001 - surface any download failure clearly
        log(f"ERR failed to download {label}: {e}")
        if dest.exists():
            dest.unlink()  # don't leave a truncated cache file behind
        sys.exit(1)
    log(f"OK  downloaded {label} -> {dest}")


def extract_python():
    """Extract the embeddable Python zip to build/python."""
    log(f"Extracting Python to {PYTHON_DIR} ...")
    with zipfile.ZipFile(PYTHON_CACHE, "r") as z:
        z.extractall(PYTHON_DIR)
    log("OK  extracted embeddable Python")


def enable_site_packages():
    """Patch pythonXY._pth so site + Lib\\site-packages are active.

    The embeddable ._pth ships with `import site` commented out and no
    site-packages entry. We uncomment `import site` and add the path so both
    pip (after bootstrap) and the installed third-party packages are importable.
    """
    pth_files = list(PYTHON_DIR.glob("python*._pth"))
    if not pth_files:
        log(f"ERR no python*._pth found in {PYTHON_DIR}")
        sys.exit(1)
    pth = pth_files[0]

    lines = pth.read_text(encoding="utf-8").splitlines()
    out = ["import site" if re.match(r"#\s*import site$", ln.strip()) else ln
           for ln in lines]

    # Path entries the bundled runtime needs (relative to the python dir):
    #   ..                  the install root next to handvol.pyw, so `import
    #                        handvol` (the package) resolves. A ._pth file
    #                        SUPPRESSES the usual "script dir on sys.path[0]"
    #                        behavior, so without this the app dies at
    #                        `from handvol import ...` with ModuleNotFoundError.
    #   Lib\site-packages   the pip-installed dependencies.
    anchor = out.index(".") + 1 if "." in out else len(out)
    for entry in ("..", "Lib\\site-packages"):
        if entry not in (l.strip() for l in out):
            out.insert(anchor, entry)
            anchor += 1
    if "import site" not in out:
        out.append("import site")

    pth.write_text("\n".join(out) + "\n", encoding="utf-8")
    log(f"OK  patched {pth.name} (site enabled; .. and Lib\\site-packages on path)")


def bootstrap_pip():
    """Install pip into the embeddable runtime via get-pip.py."""
    python_exe = PYTHON_DIR / "python.exe"
    log("Bootstrapping pip ...")
    result = subprocess.run(
        [str(python_exe), str(GET_PIP_CACHE), "--no-warn-script-location"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        log(f"ERR get-pip failed:\n{result.stdout}\n{result.stderr}")
        sys.exit(1)
    log("OK  pip bootstrapped")


def install_pip_packages():
    """Install requirements.txt into the bundled runtime's site-packages."""
    python_exe = PYTHON_DIR / "python.exe"
    requirements = REPO_ROOT / "requirements.txt"
    log("Installing pip packages (mediapipe, opencv, pycaw, ...) ...")
    result = subprocess.run(
        [str(python_exe), "-m", "pip", "install",
         "--no-warn-script-location", "-r", str(requirements)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        log(f"ERR pip install failed:\n{result.stdout}\n{result.stderr}")
        sys.exit(1)
    log("OK  installed all packages")


def copy_mediapipe_model():
    """Copy the MediaPipe model (tracked via Git LFS) into the build."""
    src = REPO_ROOT / "models" / "gesture_recognizer.task"
    dst = BUILD_DIR / "models" / "gesture_recognizer.task"
    if not src.exists():
        log(f"ERR MediaPipe model not found at {src}")
        log("    Run: git lfs pull")
        sys.exit(1)
    # An LFS pointer file is a few hundred bytes; the real model is several MB.
    if src.stat().st_size < 100 * 1024:
        log(f"ERR {src} looks like an unresolved Git LFS pointer "
            f"({src.stat().st_size} bytes). Run: git lfs pull")
        sys.exit(1)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    log(f"OK  copied MediaPipe model ({src.stat().st_size / 1024**2:.1f} MB)")


def copy_handvol_source():
    """Copy the HandVol package + entry point + requirements into the build."""
    shutil.copytree(REPO_ROOT / "handvol", BUILD_DIR / "handvol",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(REPO_ROOT / "handvol.pyw", BUILD_DIR / "handvol.pyw")
    shutil.copy2(REPO_ROOT / "requirements.txt", BUILD_DIR / "requirements.txt")
    log("OK  copied HandVol source")


def compile_nsis():
    """Compile the NSIS script into dist/HandVol-Installer.exe."""
    nsis_script = REPO_ROOT / "installer" / "handvol-installer.nsi"
    if not nsis_script.exists():
        log(f"ERR NSIS script not found at {nsis_script}")
        sys.exit(1)

    makensis = shutil.which("makensis")
    if makensis is None:
        log("ERR makensis not found on PATH.")
        log("    Install NSIS 3.x from https://nsis.sourceforge.io/Download")
        log("    and ensure makensis.exe is on PATH, then re-run.")
        sys.exit(1)

    DIST_DIR.mkdir(exist_ok=True)
    log("Compiling NSIS script ...")
    # Pass the absolute repo root as a define. makensis resolves relative
    # File/OutFile paths against the .nsi's own directory, so the script builds
    # every path from ${PROJ_ROOT} instead of relying on cwd.
    result = subprocess.run(
        [makensis, f"/DPROJ_ROOT={REPO_ROOT}", str(nsis_script)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        log(f"ERR NSIS compilation failed:\n{result.stdout}\n{result.stderr}")
        sys.exit(1)

    exe_path = DIST_DIR / "HandVol-Installer.exe"
    if not exe_path.exists():
        log(f"ERR installer not found at {exe_path} after compile")
        sys.exit(1)
    log(f"OK  built {exe_path} ({exe_path.stat().st_size / 1024**2:.1f} MB)")


def main():
    log("HandVol Windows Installer Builder")
    log("=" * 50)

    if not (REPO_ROOT / "handvol.pyw").exists():
        log("ERR not in HandVol repo root (handvol.pyw missing)")
        sys.exit(1)

    # Always start from a clean build tree; keep the download cache.
    if BUILD_DIR.exists():
        log(f"Cleaning {BUILD_DIR} ...")
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True)

    download(PYTHON_URL, PYTHON_CACHE, "embeddable Python")
    download(GET_PIP_URL, GET_PIP_CACHE, "get-pip.py")
    extract_python()
    enable_site_packages()
    bootstrap_pip()
    install_pip_packages()
    copy_mediapipe_model()
    copy_handvol_source()
    compile_nsis()

    log("=" * 50)
    log(f"Build complete! Installer: {DIST_DIR / 'HandVol-Installer.exe'}")


if __name__ == "__main__":
    main()
