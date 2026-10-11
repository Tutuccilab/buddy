# buddy

[![tests](https://github.com/Tutuccilab/buddy/actions/workflows/tests.yml/badge.svg)](https://github.com/Tutuccilab/buddy/actions/workflows/tests.yml)
[![slow tests](https://github.com/Tutuccilab/buddy/actions/workflows/slow-tests.yml/badge.svg)](https://github.com/Tutuccilab/buddy/actions/workflows/slow-tests.yml)

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
  [TUTORIAL.md](TUTORIAL.md), section 2.1; or build one in napari: Plugins → napari-buddy → Create config file).

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

Full list of dependencies: `pyproject.toml`. GPU set-up, manual install and troubleshooting (macOS, Java):
[INSTALL.md](INSTALL.md).

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
written in the normal buddy files, so spot counting per mother and bud works as before.
Switch it on in the config file:

```json
"bud_classification": { "enabled": true, "model": "with_ASH1CLB2", "ash_channel": "ASH1CLB2" }
```

| `"model"` | input | use for |
|---|---|---|
| `"DIC_DAPI_only"` | DIC + DAPI | every strain, including she2/she3 mutants |
| `"with_ASH1CLB2"` | DIC + DAPI + ASH1/CLB2 smFISH | strains with normal ASH1 mRNA transport (not she2/she3): fewer false pairs, best overall |

Tested on 198 hand-annotated cells from images the models never saw, the models assign buds to the right mother
with a pairing F1 of 0.82 (`with_ASH1CLB2`) and 0.80 (`DIC_DAPI_only`); large buds remain the weakest class, so always
check the flagged cells in napari.

All settings, the flags in napari and the output files: [BUD_MODEL.md](BUD_MODEL.md). How the models were trained
and evaluated: [methods.md](methods.md). Train your own model: `buddy/05_train_bud_model.ipynb`.

## Issues

If you encounter a problem, please [file an issue] with a detailed description.

[file an issue]: https://github.com/Tutuccilab/buddy/issues
