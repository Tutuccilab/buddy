# Installing buddy

## Quick install (recommended)

From the `buddy` repo root:

**Windows (Anaconda Prompt):**
```
powershell -ExecutionPolicy Bypass -File install.ps1
```

**macOS / Linux (Terminal):**
```
bash install.sh
```

The script creates a `buddy` conda environment (Python 3.10), installs
buddy with all its dependencies (including the napari viewer and Jupyter),
adds the `bioio-ome-tiff` writer plugin, installs the `napari-buddy` plugin
if it sits next to this repo, and runs a quick import check.

## GPU (recommended)

Omnipose and the bud model run much faster on an NVIDIA GPU. pip installs the **CPU-only** PyTorch
by default; the install scripts replace it with the CUDA version when `nvidia-smi` is found
(skip with `-CPU` on Windows / `cpu` as third argument on macOS/Linux). To do it by hand:

```
conda activate buddy
python -m pip install --force-reinstall --no-deps --no-cache-dir --timeout 120 --retries 10 "torch==2.5.1" "torchvision==0.20.1" --index-url https://download.pytorch.org/whl/cu121
python -m pip install "numpy<2"
python -c "import torch; print('GPU available:', torch.cuda.is_available())"
```

Then set `"use_gpu": true` in the `"parameters"` block of your config file (the default). Without a
GPU everything still runs, on the CPU.

With a GPU, the background filter, spot detection and the denoising step of spot decomposition also run on it (`"gpu_spot_detection": true`, the
default). They give the same spots as Big-FISH. If the GPU runs out of memory, buddy falls back to
Big-FISH on the CPU for that channel. The automatic threshold (`"threshold": null`) works on both.
Which one was used is written to `img.json` (`filter_backend`, `detection_backend`, `decomposition_backend`).

## Bud-model weights

The two trained bud models (`bud_model_DIC_DAPI_only.pt`, `bud_model_with_ASH1CLB2.pt`, ~100 MB each)
download automatically to `buddy/models/` the first time automatic bud classification runs.
Offline? Download them from the GitHub release page of this repository (assets of release
`bud-models-v1`) and put them in `buddy/models/`, or in any folder you give as
`"weights_dir"` in the config file.

## Manual install

```
conda create --name buddy python=3.10
conda activate buddy
python -m pip install -e ".[test,napari,notebooks]"   # buddy + deps + pytest + napari + jupyter
python -m pip install bioio-ome-tiff                   # OME-TIFF writer (see note below)
# optional, NVIDIA GPU: see "GPU" above for the CUDA version of PyTorch
python -m pip install -e ../napari-buddy               # the napari plugin (adjust the path)
```

Python **must** be 3.9 or 3.10 — omnipose does not work on 3.11+.
Use `python -m pip` so installs land in the active env, not base conda.

## Test it

```
conda activate buddy
pytest buddy/tests/test_image.py -v
python -m napari      # Plugins -> buddy   (never bare 'napari')
python -m notebook    # run the analysis notebooks (from the repo root)
```

The pipeline test runs on the bundled example data and should finish with
no errors.

## Good to know

- **bioio OME-TIFF writer.** bioio 3.x split its writers into separate
  packages, so `OmeTiffWriter` now lives in `bioio-ome-tiff`, and
  `buddy/image.py` imports it as
  `from bioio_ome_tiff.writers import OmeTiffWriter`. The installers pull
  this in automatically; if you install by hand, run
  `pip install bioio-ome-tiff`.
- **Launch with `python -m`.** Always use `python -m napari` and
  `python -m notebook`, never bare `napari` / `jupyter`. The bare commands
  can bounce to a broken base-conda copy that happens to be first on your
  PATH (the cause of the numpy/`_ssl` DLL errors).
- **First `.vsi` read is slow.** Reading Olympus `.vsi` files needs Java.
  bioio-bioformats fetches a JRE and Maven automatically the first time
  (`Fetching zulu-jre:11 using cjdk...`). This needs internet once; after
  that it's cached. On locked-down/offline machines, uncomment
  `openjdk=11` and `maven` in `environment.yml` and build the env with
  `conda env create -f environment.yml` instead.
- **Updating the code.** buddy is installed in editable mode (`pip install -e`), so the environment
  uses the code in this folder directly. After pulling or editing code, restart the Jupyter kernel;
  no reinstall needed (only when `pyproject.toml` gets new dependencies).
- **No GPU?** Nothing to do: buddy falls back to the CPU (or set `"use_gpu": false` in the config).
- **Apple Silicon + omnipose.** If the omnipose step fails to build,
  comment `omnipose` out of `pyproject.toml`, reinstall, then:
  ```
  pip install git+https://github.com/kevinjohncutler/omnipose.git
  pip install git+https://github.com/kevinjohncutler/cellpose-omni.git
  ```
