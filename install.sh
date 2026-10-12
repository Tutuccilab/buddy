#!/usr/bin/env bash
# buddy installer for macOS / Linux (run from your Terminal, in the buddy repo root)
#
#   bash install.sh [path-to-napari-buddy] [env-name] [auto|cpu]
#
# Defaults: napari-buddy path = ../napari-buddy, env name = buddy, GPU = auto (CUDA PyTorch if an NVIDIA GPU is found)

set -euo pipefail

NAPARI_PATH="${1:-../napari-buddy}"
ENV_NAME="${2:-buddy}"
USE_GPU="${3:-auto}"     # auto | cpu

echo "==> Creating conda environment '${ENV_NAME}' (Python 3.10, Java 11 + Maven for reading .vsi files)"
# Java + Maven come from conda-forge (all platforms), so Bio-Formats does not have to download a Java on first use
# (which fails without internet or behind a proxy). Activating the env sets JAVA_HOME.
conda create --name "${ENV_NAME}" python=3.10 conda-forge::openjdk=11 conda-forge::maven -y

echo "==> Installing buddy (editable, with test + napari + notebooks extras)"
# The 'napari' extra pulls in napari[pyqt6] - the viewer + Qt backend -
# INTO this env, so 'python -m napari' launches from here and never falls
# back to a base-conda napari.
conda run --no-capture-output -n "${ENV_NAME}" pip install -e ".[test,napari,notebooks]"

echo "==> Ensuring bioio OME-TIFF writer plugin"
# bioio 3.x split its writers into separate plugin packages, so
# buddy/image.py imports OmeTiffWriter from bioio_ome_tiff. Install it
# explicitly so this works even if the on-disk pyproject.toml predates
# that dependency.
conda run --no-capture-output -n "${ENV_NAME}" pip install "bioio-ome-tiff"

# --- Apple Silicon note -------------------------------------------------
# If 'pip install -e .' fails on the omnipose step on an M1/M2/M3 Mac,
# comment out "omnipose" in pyproject.toml, rerun this line, then install
# omnipose from source in the same env:
#   pip install git+https://github.com/kevinjohncutler/omnipose.git
#   pip install git+https://github.com/kevinjohncutler/cellpose-omni.git
# -----------------------------------------------------------------------

if [ -d "${NAPARI_PATH}" ]; then
    echo "==> Installing napari-buddy plugin from ${NAPARI_PATH}"
    conda run --no-capture-output -n "${ENV_NAME}" pip install -e "${NAPARI_PATH}"
else
    echo "==> napari-buddy not found at '${NAPARI_PATH}' - skipping."
    echo "    Install it later with: conda activate ${ENV_NAME} && pip install -e <path-to-napari-buddy>"
fi

# GPU (Linux + NVIDIA): pip installs a PyTorch build that may lack CUDA. With an NVIDIA GPU, install the
# CUDA build so Omnipose and the bud model run on the GPU. (macOS has no CUDA: CPU is used.)
if [ "${USE_GPU}" != "cpu" ] && command -v nvidia-smi >/dev/null 2>&1; then
    echo "==> NVIDIA GPU found: installing PyTorch with CUDA support"
    # --no-deps: only swap torch (+ torchvision), keep everything else; retries: the file is ~2.5 GB
    conda run --no-capture-output -n "${ENV_NAME}" pip install --force-reinstall --no-deps --no-cache-dir --timeout 120 --retries 10 "torch==2.5.1" "torchvision==0.20.1" --index-url https://download.pytorch.org/whl/cu121
    conda run --no-capture-output -n "${ENV_NAME}" pip install "numpy<2"
else
    echo "==> No NVIDIA GPU found (or 'cpu' given): keeping the CPU version of PyTorch"
fi

echo "==> Smoke test (imports only)"
conda run --no-capture-output -n "${ENV_NAME}" python -c "import buddy, bigfish, omnipose, bioio_ome_tiff, segmentation_models_pytorch, torch; print('buddy imports OK | GPU available:', torch.cuda.is_available())"
conda run --no-capture-output -n "${ENV_NAME}" java -version

echo ""
echo "Done. Next steps:"
echo "  conda activate ${ENV_NAME}"
echo "  pytest buddy/tests/test_image.py -v    # full pipeline test on example data"
echo "  python -m napari                       # launch viewer; check Plugins -> buddy"
echo "  python -m notebook                    # run the analysis notebooks"
echo ""
echo "Tip: always launch with 'python -m napari' (not bare 'napari') so it"
echo "runs THIS env's napari and can't get bounced to base conda."
