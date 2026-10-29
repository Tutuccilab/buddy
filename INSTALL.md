# Installing buddy

This is the only place where installation is described; the README, the tutorial and napari-buddy point here.
You need [Miniconda or Anaconda](https://docs.conda.io/en/latest/miniconda.html) and Git. The installer makes a
conda environment called `buddy` with Python 3.12 and everything in it (buddy, napari + the napari-buddy plugin,
Jupyter, Java for reading `.vsi` files); internet is needed once; an NVIDIA GPU is optional.

Every package version is pinned in `constraints/<platform>.txt` (the versions buddy is tested with), and the
installers pass that file to pip, so an install on another machine gets the same packages as ours.

## Quick install (recommended)

Put the two repositories side by side (`<folder>/buddy` and `<folder>/napari-buddy`; the zip from Zenodo is
already laid out like this), then run **from the `buddy` repo root**:

**Windows** (Anaconda Prompt):
```
powershell -ExecutionPolicy Bypass -File install.ps1
```

**macOS** (Terminal):
```
bash install.sh
```

**Linux** (Terminal):
```
bash install.sh
```
On Linux the installer picks the CPU build of PyTorch when no NVIDIA GPU is found (the default download is the
2.5 GB CUDA build), uses `~/.cache` as temporary space (`/tmp` is too small on some machines) and installs napari
with Qt5 (`napari[pyqt5]`; Qt6 does not start on every Linux desktop).

The scripts do, in this order: create the environment (Python 3.12, Java 11) · `pip install -c
constraints/<platform>.txt -e ".[test,napari,notebooks]"` · `pip install --no-deps omnipose` (see
"Omnipose" below) · `pip install -e ../napari-buddy` · Windows with an NVIDIA GPU: swap in the CUDA build of PyTorch
· an import check and `java -version`.

On every platform the script removes an existing `buddy` environment first, so it can simply be run again after a
failed attempt. Options: `-CPU` (Windows) or `cpu` as third argument (macOS / Linux) to skip the GPU build of
PyTorch; the first argument is the path to napari-buddy (default `../napari-buddy`), the second the environment
name. The script ends with an import check and `java -version`.

## GPU (recommended)

Omnipose, the bud model and spot detection run much faster on an NVIDIA GPU. PyTorch is pinned to 2.5.1.
- **Linux:** the normal PyPI wheel of torch 2.5.1 is the CUDA 12.4 build and brings its CUDA libraries along
  (`nvidia-*-cu12` packages); it works with any NVIDIA driver from 525 on. Nothing to swap. Without a GPU the
  installer takes the much smaller CPU wheel instead.
- **Windows:** the PyPI wheel is CPU-only; the installer swaps in the self-contained CUDA 12.1 build when
  `nvidia-smi` is found (skip with `-CPU`). By hand:
  ```
  conda activate buddy
  python -m pip install --force-reinstall --no-deps --no-cache-dir --timeout 120 --retries 10 "torch==2.5.1" "torchvision==0.20.1" --index-url https://download.pytorch.org/whl/cu121
  python -c "import torch; print('GPU available:', torch.cuda.is_available())"
  ```
- **macOS:** CPU only.

Do not mix: a CUDA build from the PyTorch index installed with `--no-deps` on **Linux** lacks its `nvidia-*`
libraries and will not import. On Linux, use the PyPI wheel.

Then set `"use_gpu": true` in the `"hardware"` block of your config file (the default). Without a
GPU everything still runs, on the CPU.

With a GPU, the background filter, spot detection and the denoising step of spot decomposition also run on it (`"gpu_spot_detection": true`, the
default). They give the same spots as Big-FISH. If the GPU runs out of memory, buddy falls back to
Big-FISH on the CPU for that channel. The automatic threshold (`"threshold": null`) works on both.
Which one was used is written to `img.json` (`filter_backend`, `detection_backend`, `decomposition_backend`).

## Bud-model weights

The two trained bud models (`bud_model_DIC_DAPI_only_v2.pt`, `bud_model_with_ASH1CLB2_v2.pt`, ~100 MB each)
download automatically to `buddy/models/` the first time automatic bud classification runs.
Offline? Download them from the GitHub release page of this repository (assets of release
`bud-models`) and put them in `buddy/models/`, or in any folder you give as
`"weights_dir"` in the config file.

## Manual install

```
conda create --name buddy python=3.12
conda activate buddy
python -m pip install -c constraints/windows.txt -e ".[test,napari,notebooks]"   # buddy + deps + pytest + napari + jupyter
python -m pip install --no-deps omnipose==1.1.4                                  # see "Omnipose" below
# Windows with an NVIDIA GPU: see "GPU" above for the CUDA version of PyTorch
python -m pip install -c constraints/windows.txt -e ../napari-buddy              # the napari plugin (adjust the path)
```
(`constraints/linux.txt` or `constraints/macos.txt` on those platforms.) Leaving out `-c` works too, but then pip
takes the newest version of every package, and new releases of Big-FISH, scikit-image or Omnipose change the
results or break the install.

Python 3.10, 3.11 or 3.12 (3.12 is what the installers use and the constraints are made for; 3.13 has no PyTorch 2.5.1).
Use `python -m pip` so installs land in the active env, not base conda.

## Test it

```
conda activate buddy
pytest -m "not slow"   # fast tests (~1 min); pytest -m slow: Omnipose, bud model, one real .vsi
python -m napari      # Plugins -> buddy   (never bare 'napari')
python -m notebook    # run the analysis notebooks (in notebooks/)
```

The tests download a small test field on first use (release `test-data-v3`, 70 MB; the slow `.vsi` test also `test-data-vsi`, 400 MB) and should finish with no errors. For a private repository the download goes through the GitHub CLI (`gh auth login` once).

## Good to know

- **Omnipose.** `pip install omnipose` pulls in about a hundred packages (aicsimageio, networkit, dask-image, ...)
  of which segmentation uses a dozen. buddy therefore lists those dozen itself in `pyproject.toml` and installs
  `omnipose` with `--no-deps`. The omnipose wheel includes its own copy of `cellpose_omni`; the separate
  `cellpose-omni` package on PyPI (0.9.1) is older and, installed next to it, overwrites that copy - Omnipose then
  silently runs as plain Cellpose (`>>cyto<< model set to be used` in the log, 0 cells, a `niter` error on the
  nuclei). Never install it; `pip uninstall cellpose-omni` + `pip install --force-reinstall --no-deps
  omnipose==1.1.4` repairs an environment that has it. `buddy.segmentation.masks.check_omnipose()` (run by the
  installers) checks that the real Omnipose is in place. (One of the unused packages, `aicsimageio`, is imported by
  `cellpose_omni.io` at import time without being needed; `masks.py` registers an empty stand-in for it.) (Omnipose's own GUI, `omnipose` on the command line, needs its `gui` extra: `pip install
  "omnipose[gui]"`.)
- **Updating the pinned versions.** After changing `pyproject.toml`, regenerate the three constraint files (the
  command is in their first lines; `pip install uv` gives the `uv` tool), run the tests, commit. Only change
  `torch==` / `torchvision==` together with the CUDA index URLs in the installers.
- **Launch with `python -m`.** Always use `python -m napari` and
  `python -m notebook`, never bare `napari` / `jupyter`. The bare commands
  can bounce to a broken base-conda copy that happens to be first on your
  PATH (the cause of the numpy/`_ssl` DLL errors).
- **Java for `.vsi` files.** Reading Olympus `.vsi` files goes through Bio-Formats (Java). Bio-Formats itself is a
  pip package (`bioformats_jar`, the 6.7.0 jar + JPype), so nothing is downloaded on first use; Java comes from
  conda-forge (`openjdk=11`, put in the environment by the installers; activating the environment sets `JAVA_HOME`).
  Another Java works too: JPype finds it through `JAVA_HOME`.
- **Updating the code.** buddy is installed in editable mode (`pip install -e`), so the environment
  uses the code in this folder directly. After pulling or editing code, restart the Jupyter kernel;
  no reinstall needed (only when `pyproject.toml` gets new dependencies).
- **No GPU?** Nothing to do: buddy falls back to the CPU (or set `"use_gpu": false` in the config).
- **Apple Silicon + Omnipose.** If `mahotas` or `edt` fail to build (no wheel for your Python), install the
  Xcode command-line tools (`xcode-select --install`) and run the installer again; if Omnipose itself fails, see
  Troubleshooting.

## Troubleshooting

- **Omnipose on macOS.** If the Omnipose step fails, install it from source
  (see [this issue](https://github.com/kevinjohncutler/omnipose/issues/14)):
  ```
  conda activate buddy
  pip install --no-deps git+https://github.com/kevinjohncutler/omnipose.git
  ```
- **`import torch` fails on Linux with a missing `libcudnn` / `libcublas`:** a CUDA build from the PyTorch index
  was installed with `--no-deps` (older installers did this). `pip install --force-reinstall "torch==2.5.1"
  "torchvision==0.20.1"` from PyPI brings the libraries along.
- **Java not found when reading `.vsi` files** (e.g. an environment made by hand, or with an older installer):
  `conda install -c conda-forge openjdk=11`, then deactivate and reactivate the environment. Check with
  `java -version`. If it is still not found, set `JAVA_HOME` once: `conda env config vars set JAVA_HOME=%CONDA_PREFIX%\Library`
  (Windows) or `conda env config vars set JAVA_HOME=$CONDA_PREFIX/lib/jvm` (macOS / Linux), and reactivate.
- **A `.vsi` file does not read** (buddy <= 0.5.1 said "Bio-Formats does not support this image" when the
  Bio-Formats download of the first read had failed; since 0.5.2 Bio-Formats is installed with buddy and there is
  no download). To see what Bio-Formats makes of a file:
  `python -c "from buddy.reading.vsi import BioFormatsFile; f = BioFormatsFile(r'path/to/file.vsi'); print(f.series_names, f.shape(0))"`.
  Another Bio-Formats version can be tried without reinstalling: set `BIOFORMATS_JAR` to the path of a
  `bioformats_package.jar` (from downloads.openmicroscopy.org) before starting Python.
- **CUDA out of memory in step 2** (small GPU, e.g. 4 GB): set `"segmentation": {"tile": true}` in the config file
  (Omnipose on tiles) or `"use_gpu": false`; the error message says so too. See TUTORIAL.md, section 2.1.
- **Omnipose GUI** (optional, handy for finding segmentation settings): run `omnipose` from the command line.
