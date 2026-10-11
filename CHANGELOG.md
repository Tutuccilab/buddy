# Changelog

## 0.1.0 (2026-10-09)

Added after the analysis in the manuscript; results in the manuscript were obtained with 0.0.1.

**Automatic bud classification**
- GPU background filter: the result is stored as uint16 block by block (no full-size int32 copy): about 1 GB less peak RAM per 2304 x 2304 x 40 stack.
- `pipeline.progress_callback`: an optional hook (image, step) that a GUI can set to show what is being processed (used by napari-buddy's pipeline panel).
- A trained model assigns every bud to its mother and writes the bud masks (`bud_model.py`, "model D": the model
  sees DIC, DAPI and a soft dot on the cell's centre instead of its Omnipose outline, and predicts the cell, its
  bud and its class). Two models: `DIC_DAPI_only` (any strain) and `with_ASH1CLB2` (not for she2/she3 mutants).
  Switch on with the `bud_classification` block in the config file. Weights: release `bud-models`.
- "Snap" (`"snap": true`): a bud that lies mostly inside one Omnipose object takes that whole object as its outline.
- Cells the model is unsure about are flagged in napari (`check these buds` layer) and listed in `bud_flags.csv`.
- Bud masks corrected by hand are never overwritten.
- Unbudded cells with high ASH1/CLB2 are flagged in napari as *late-M pair?* or *high ASH1/CLB2, no bud*
  (`cell_flags.py`, `cell_flags.csv`, config block `cell_flags`); computed after the spot counts of steps 3 and 5.
- napari (needs napari-buddy 0.1.0): a `bud → mother links` layer shows which bud belongs to which mother
  (replaces the magenta boxes); layers load grouped, with a **Layer groups** panel to show / hide them.
- Step 4 of the bulk-processing notebooks is one cell: `review_images(exp)` opens a **Correct images** panel in
  napari (previous / save / save + next / skip).
- napari panel **buddy pipeline** (napari-buddy 0.1.0): runs steps 1-5 with buttons, in the background, with
  a progress bar, a time estimate and "Stop after this image". The notebook and the panel use the same
  per-image functions (`pipeline.py`).
- Train your own model: `05_train_bud_model.ipynb` (`bud_training.py`).
- Model weights download automatically from the `bud-models` release.

**Input**
- Multi-page TIFF stacks (OME-TIFF, Fiji/ImageJ hyperstacks) are accepted as input next to .vsi files
  (`tiff_input.py`; config: `tiff_channel`, `tiff_axes`, `dic_channel`).

**Tests**
- `tests/`: unit tests (counting, alignment, cell flags, bud-model rules, config / re-run check, TIFF input) and a
  regression test on a 512 x 512 px crop of a real image (`tests/data/`, downloaded from the `test-data-v1`
  release); GitHub Actions runs the fast tests on Linux and Windows.

**Speed**
- GPU versions of the background filter, spot detection and the denoising filter of spot decomposition
  (`gpu_filters.py`); same spots and dense regions as Big-FISH.
  Config: `"gpu_spot_detection"` (default true). Omnipose and the bud model use the GPU via `"use_gpu"`.
- Background filter only on the in-focus slices (same result, faster).
- Steps 1 and 3 also save 2D maximum projections (`projections/`), so images open fast in napari.
- Segmentation reads only DAPI; the intermediate `img.05.pkl` is no longer written.
- Step 1 of `01_fish_bulk_processing.ipynb` reads the .vsi files of several images at once (the number of
  workers follows from the CPU cores and free memory).

**Edge cells**
- `"remove_edge_cells": true` (parameters block) removes cells touching the image edge in step 2. Default `false`.

**Re-running steps**
- Each step skips images already processed with the same settings, and lists what changed and asks before
  overwriting images processed with other settings (`"if_already_processed": "ask" | "skip" | "overwrite"`).
  The settings each step used are recorded in `img.json` (`results` → `settings`).

**Fixes**
- Re-running step 3 no longer shifts the cell masks a second time. The alignment shift is estimated on the
  Omnipose masks with a fixed random seed, so it is the same every run; hand corrections are kept.
- `img.json` `parameters` now show the settings of the latest run (before, they kept the values from step 1).

**Install**
- Install scripts install the CUDA version of PyTorch when an NVIDIA GPU is found.
- New dependencies: `segmentation-models-pytorch`, `psutil`.

## 0.0.1

Version used for the manuscript (Zenodo: https://doi.org/10.5281/zenodo.22900927).
