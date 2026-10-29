#!/usr/bin/env bash
# buddy installer for macOS and Linux (run in a Terminal, from the buddy repo root)
#
#   bash install.sh [path-to-napari-buddy] [env-name] [auto|cpu]
#
# Defaults: napari-buddy path = ../napari-buddy, env name = buddy, GPU = auto (CUDA PyTorch when an NVIDIA GPU is found).
# An existing environment of that name is removed first, so the script can be run again after a failed install.
# Every package version comes from constraints/<platform>.txt, so every install gets the same packages.
# Windows: install.ps1 (same steps). What the script does, and what to do when it fails: INSTALL.md.

set -euo pipefail

NAPARI_PATH="${1:-../napari-buddy}"
ENV_NAME="${2:-buddy}"
USE_GPU="${3:-auto}"     # auto | cpu

[ -f pyproject.toml ] || { echo "Run this from the buddy repo root (the folder with pyproject.toml)." >&2; exit 1; }
command -v conda >/dev/null 2>&1 || { echo "conda not found: install Miniconda or Anaconda first (INSTALL.md)." >&2; exit 1; }

case "$(uname -s)" in
    Linux)  PLATFORM=linux ;;
    Darwin) PLATFORM=macos ;;
    *)      echo "Unknown platform $(uname -s): on Windows run install.ps1." >&2; exit 1 ;;
esac
CONSTRAINTS="constraints/${PLATFORM}.txt"
OMNIPOSE=$(grep -E '^omnipose==' "${CONSTRAINTS}")

# pip unpacks PyTorch (up to 2.5 GB) in TMPDIR; on some Linux machines /tmp is too small or not executable
export TMPDIR="${HOME}/.cache/buddy-install-tmp"
mkdir -p "${TMPDIR}"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "==> Removing the existing conda environment '${ENV_NAME}'"
    conda env remove --name "${ENV_NAME}" -y
fi

echo "==> Creating conda environment '${ENV_NAME}' (Python 3.12, Java 11 for reading .vsi files)"
# Java comes from conda-forge (activating the env sets JAVA_HOME); Bio-Formats itself is a pip package (bioformats_jar).
conda create --name "${ENV_NAME}" python=3.12 conda-forge::openjdk=11 -y

run() { conda run --no-capture-output -n "${ENV_NAME}" "$@"; }

# PyTorch. Linux: the PyPI wheel is the CUDA 12.4 build (2.5 GB, with its CUDA libraries as dependencies) and works
# with any NVIDIA driver >= 525; without a GPU (or with "cpu") the CPU wheel is installed first, so pip keeps it
# (--extra-index-url, not --index-url: torch's own dependencies such as setuptools come from PyPI; the +cpu wheel wins).
# macOS: no CUDA, the default wheel is the right one.
if [ "${PLATFORM}" = "linux" ] && { [ "${USE_GPU}" = "cpu" ] || ! command -v nvidia-smi >/dev/null 2>&1; }; then
    echo "==> No NVIDIA GPU found (or 'cpu' given): installing the CPU version of PyTorch"
    run pip install -c "${CONSTRAINTS}" torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
fi

echo "==> Installing buddy (editable, with test + napari + notebooks extras; versions from ${CONSTRAINTS})"
# The 'napari' extra pulls the napari viewer + Qt backend INTO this env (pyqt6; pyqt5 on Linux), so
# 'python -m napari' launches from here and never falls back to a base-conda napari.
run pip install -c "${CONSTRAINTS}" -e ".[test,napari,notebooks]"

echo "==> Installing Omnipose (${OMNIPOSE}, without its own dependency list: what it needs is in pyproject.toml)"
run pip install --no-deps "${OMNIPOSE}"

# --- Apple Silicon note -------------------------------------------------
# If a step fails on an M1/M2/M3 Mac, see INSTALL.md -> Troubleshooting.
# -----------------------------------------------------------------------

if [ -d "${NAPARI_PATH}" ]; then
    echo "==> Installing napari-buddy plugin from ${NAPARI_PATH}"
    run pip install -c "${CONSTRAINTS}" -e "${NAPARI_PATH}"
else
    echo "==> napari-buddy not found at '${NAPARI_PATH}' - skipping."
    echo "    Install it later with: conda activate ${ENV_NAME} && pip install -e <path-to-napari-buddy>"
fi

rm -rf "${TMPDIR}"

echo "==> Smoke test (imports only)"
run python -c "import buddy, bigfish, bioformats_jar, torch, torchvision; from buddy.segmentation import masks; masks.check_omnipose(); print('buddy imports OK | GPU available:', torch.cuda.is_available())"
# load the bud model once now: the first load after an install takes minutes (file scan, CUDA start-up), later ~15 s
echo "==> Loading the bud model once (the first load is slow; it is fast from now on)"
run python -c "from buddy.buds.bud_model import BudModel; BudModel.load('DIC_DAPI_only', use_gpu=True); print('bud model OK')"
run java -version

echo ""
echo "Done. Next steps:"
echo "  conda activate ${ENV_NAME}"
echo "  pytest -m 'not slow'                   # fast tests (about a minute)"
echo "  python -m napari                       # the viewer; Plugins -> Napari Buddy"
echo "  python -m notebook notebooks           # the analysis notebooks"
echo ""
echo "Tip: always launch with 'python -m napari' (not bare 'napari') so it runs THIS env's napari."
