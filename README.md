# buddy

An end-to-end pipeline for quantifying single-molecule FISH (smFISH) in
yeast cells — from raw microscope files to per-cell mRNA counts.
`buddy` orchestrates:

- **Read** — VSI fluorescence stacks + DIC cell images
- **Segment** — cells and nuclei (Omnipose/Cellpose)
- **Detect** — spot detection and dense-region decomposition (Big-FISH, https://big-fish.readthedocs.io/)
- **Align** — registers the DIC/segmentation frame to the fluorescence frame
- **Assign buds** — a trained deep-learning model assigns every bud to its mother automatically
  and flags the cells worth checking (see *Automatic bud classification* below)
- **Correct** — interactive mother/bud annotation in napari (via the
  companion [`napari-buddy`](https://github.com/Tutuccilab/napari-buddy) plugin)
- **Quantify** — per-cell and per-compartment (mother vs bud) counts of
  mRNAs, dense regions, nascent RNAs, transcription sites, and nuclei,
  with cell morphology
- **Track** - Processing parameters are recorded per image (JSON) so runs stay reproducible.

Goal is to automate processing and record processing parameters.

What changed per version: [CHANGELOG.md](CHANGELOG.md). Version 0.1.0 adds automatic bud classification and GPU spot detection; the manuscript analysis used 0.0.1.

# Expected inputs
The module expects a directory containing the following files:
- Fluorescent channels: .vsi microscopy files and the corresponding vsi directories, **or** multi-page TIFF
  stacks (OME-TIFF or Fiji/ImageJ hyperstacks with the channels as z-stacks, e.g. `image_01.ome.tif`). The
  channel names in the file are matched to the `"filter"` names in the config; a file without channel names
  needs `"tiff_channel": <index>` in each `"channels"` entry, a file without axis information `"tiff_axes"`
  (e.g. `"ZCYX"`) in `"parameters"`.
- Cell outlines: 2D .tif files (e.g. DIC images), named `<image>_DIC.tif`; for TIFF stacks the DIC image may
  also be a channel of the stack named `DIC` (its middle z plane is used)
- Configuration: a json file containing channel and processing parameters (all settings are explained in
  `INSTALLATION_TUTORIAL.md`, section 2.1; or build one in napari: Plugins → napari-buddy → Create config file).

# Download code and example data
Go to https://doi.org/10.5281/zenodo.22900927 to download napari, napari-buddy and some test data. 
The folder structure of the zipped file should allow you to directly start with the tutorial after installation.
Note: newer versions of the code can be downloaded by cloning this repo.

# Installation

From the **`buddy` repo root**:

**Windows (Anaconda Prompt):**
```
powershell -ExecutionPolicy Bypass -File install.ps1
```

**macOS / Linux (Terminal):**
```
bash install.sh
```
This will install buddy and all its dependencies. If an NVIDIA GPU is found, the GPU version of
PyTorch is installed too, so Omnipose and the bud model run on the GPU (add `-CPU` on Windows, or
`cpu` as third argument on macOS/Linux, to skip that). The bud-model weights download automatically
the first time they are used.

# Dependencies

Full list in `pyproject.toml`. The key ones:

**Core processing**
- `big-fish` — smFISH spot detection and dense-region decomposition
- `omnipose` + `cellpose-omni` — cell and nucleus segmentation
- `scikit-image` — image I/O, morphology, region properties
- `segmentation-models-pytorch` (+ PyTorch, installed with omnipose) — the bud model
- `numpy`, `scipy` — arrays; alignment (distance transform) and focus (Savitzky–Golay)
- `pandas` — per-cell quantification tables
- `jsonpickle` — serialising image objects and parameters to JSON
- `natsort` — natural sorting of image lists

**Reading VSI microscopy files** (requires a working Java environment)
- `bioio` + `bioio-bioformats`
- `scyjava`

**Interactive mask/bud correction**
- `napari` (with a Qt backend, e.g. `napari[pyqt6]`)
- `napari-buddy` — the companion plugin (its reader is imported in the workflow notebook)

**Development**
- `pytest` — test suite
Details:
 - Omnipose

   On MacOS: install from git source or git clone on MacOS in a conda env with python<=3.10.11 (see [this issue](https://github.com/kevinjohncutler/omnipose/issues/14))

   First add `jupyterlab` to `omnipose_mac_environment.yml` from [Omnipose](https://omnipose.readthedocs.io/installation.html) repository, then:
   ```
   conda env create --name myenv --file /Volumes/DataDrive/omnipose_mac_environment.yml
   conda activate myenv
   pip install git+https://github.com/kevinjohncutler/omnipose.git
   pip install git+https://github.com/kevinjohncutler/cellpose-omni.git
   ```

- Omnipose GUI (nice to have, not required)

   Run `omnipose` from the command line.

Omnipose, the bud model, the background filter, spot detection and the denoising step of spot decomposition use the GPU when `"use_gpu": true` is set in the config file (the default) and PyTorch can see a CUDA GPU; otherwise they run on the CPU (slower, but works). The GPU versions give the same spots as Big-FISH; switch them off with `"gpu_spot_detection": false`. The install scripts install the GPU version of PyTorch automatically when an NVIDIA GPU is present (see INSTALL.md).

- Other
   ```
   conda install -c conda-forge scyjava
   pip install "napari[pyqt6]"
   pip install bioio bioio_bioformats big-fish jsonpickle pathlib  
   ```
- Might be needed:
   - `export JAVA_HOME=/Users/elton/miniconda3/envs/myenv/lib/jvm`
   - deactivate and reactivate myenv for `scyjava` to work

# Typical workflow
1. Batch process inputs to TIF files: `buddy/01_fish_bulk_processing.ipynb`
2. Find good segmentation parameters using Omnipose GUI
3. Batch segment DIC and DAPI pictures: `buddy/01_fish_bulk_processing.ipynb`
4. Find good spot detection parameters using napari-buddy plugin
   1. Load an img.json file in napari
   2. Tweak spot detection parameters
5. Batch detect spots, align, **assign buds automatically** and count spots: `buddy/01_fish_bulk_processing.ipynb`
6. Check the flagged buds (and anything else) in napari, step 4 of the same notebook

Instead of the notebook you can run all steps from napari: **Plugins → napari-buddy → buddy pipeline**
(choose the config file and folders, then one button per step, with a progress bar and time estimate).
Both run the same per-image functions (`buddy/pipeline.py`).

Running a step again skips images already processed with the same settings; for images processed with other
settings it lists what changed and asks before overwriting (config: `"if_already_processed": "ask" | "skip" | "overwrite"`).

The notebook reads the .vsi files of several images at the same time, and runs the background filter, spot
detection and spot decomposition on the GPU when one is available.

# Automatic bud classification

After the alignment step, a trained model decides for every segmented cell whether it is
unbudded, budded or itself a bud, draws the bud, and assigns it to its mother. The result is
written in the normal buddy files (`corrected/DIC_corrected_masks.tif` with each bud merged into
its mother, `corrected/bud_corrected_masks.tif` with 1 = bud, 2 = background, 3 = mother), so
spot counting per mother and bud works as before.

**Switch it on / choose the model** in the config file:

```json
"parameters": { "...": "...", "use_gpu": true, "gpu_spot_detection": true },
"bud_classification": {
    "enabled": true,
    "model": "with_ASH1CLB2",
    "ash_channel": "ASH1CLB2",
    "overwrite_manual_corrections": false
}
```

| `"model"` | input | use for |
|---|---|---|
| `"DIC_DAPI_only"` | DIC + DAPI | every strain, including she2/she3 mutants |
| `"with_ASH1CLB2"` | DIC + DAPI + ASH1/CLB2 smFISH | strains with normal ASH1 mRNA transport (not she2/she3): fewer false pairs, best overall |

* `"ash_channel"`: name of the ASH1/CLB2 channel in `"channels"` (only used by `with_ASH1CLB2`).
* `"overwrite_manual_corrections"`: bud masks you corrected by hand are never replaced, unless this is `true`.
* `"snap"` (default true): a bud the model found that lies mostly inside one Omnipose object becomes that whole object
  (the model says which bud belongs to which mother, Omnipose gives the full outline; helps for large buds).
* `"weights_dir"` (optional): folder with your own trained weights (default: `buddy/models`, downloaded automatically).

**Check the result in napari** (step 4 of the bulk-processing notebook). Cells the model was
unsure about are boxed in the `check these buds` layer:

* red — *close call*: two cells claim the same bud almost equally strongly
* orange — *bud without mother*: looks like a bud, but no mother claimed it
* yellow — *weak pair* / *possible missed bud*: just above or below the model's cut-off

Unbudded cells with high ASH1/CLB2 (≥ 8× a typical unbudded cell in the image) are boxed too, with or without the
bud model, whenever an ASH1/CLB2 channel is counted (`cell_flags.py`, listed in `cell_flags.csv`):

* magenta — *late-M pair?*: the cell and an unbudded neighbour each have their own full nucleus: mother + daughter
  in late M (pair them if they belong together), or two cells that have just divided
* green — *high ASH1/CLB2, no bud*: a newborn daughter, or a cell in early S whose small bud is not marked
  (hover text gives a rough DNA-content hint from DAPI: ~1 = G1-like, higher = S/G2-like)

Settings (optional): `"cell_flags": {"enabled": true, "ash_channel": null, "fold": 8, "nucleus_area": 0.5,
"nucleus_intensity": 0.7, "max_gap_px": 10}` (`ash_channel` null = the one of `bud_classification`).

The `bud → mother links` layer draws a line from each bud to its mother, and the **Layer groups** panel
(napari-buddy ≥ 0.1.0) shows / hides whole groups of layers. Flagged cells are also listed, with the reason, in `bud_flags.csv` in each image folder (`bud_pairs.csv`
lists the pairs found, `bud_model_cells.csv` the model output per cell).

**How good is it?** On a blind, hand-annotated test set (198 cells from test images the models never
saw, chosen to over-represent the hard cases) the models find mother-bud pairs better than the original manual
curation of those cells did: pairing F1 0.82 (`with_ASH1CLB2`) and 0.80 (`DIC_DAPI_only`) vs 0.72, with 86 % of
the very small buds (< 1 µm²) found vs 39 % by the curation. Large buds (> 4 µm²) remain the weakest class
(69-81 % found vs 96 %), so always check the flagged cells. Details: `methods.md`, scores per model:
`buddy/models/bud_model_<model>.json`.

**Train your own model** on your corrected images with `buddy/05_train_bud_model.ipynb`
(code: `buddy/bud_training.py`; the model itself: `buddy/bud_model.py`).

## Issues

If you encounter a problem, please [file an issue] with a detailed description.

[napari]: https://github.com/napari/napari
[pytest]: https://docs.pytest.org/
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[file an issue]: https://github.com/Tutuccilab/buddy/issues