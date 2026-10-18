# Changelog

## 0.4.0 (2026-10-18)
- DIC turn (still measured per image): buddy now warns when the DIC lighting direction of an image cannot be measured
  clearly (consistency < 0.9, e.g. blurry DIC) and writes the measurement to `img.json`; `bud_training.prepare_image`
  warns the same way. Optional config setting `"bud_classification"` -> `"dic_rotation"` (null = measured per image,
  default; 0-3 = fixed turn for every image of the experiment).
- `bud_training.training_rows` leaves out the she2/she3 strains (YET563, YET572) for `with_ASH1CLB2` by default
  (`EXCLUDED_STRAINS`; before, only notebook 05 passed them), and matches strain names as whole names.
- Training files from `bud_training.prepare_image` now include the imaging day in their name (`<day>__<strain>__<image>`):
  image names repeat between days, and the second one used to be skipped silently. Prepare the images again.
- **Shipped bud models replaced by the models reported in the paper** (`bud_model_DIC_DAPI_only_v2.pt`,
  `bud_model_with_ASH1CLB2_v2.pt`, release `bud-models`): same recipe, trained without imaging day 240909, last epoch (30)
  and claim cut-off 0.5 for both models, fixed in advance (was: epoch 22 / cut-off 0.6 and epoch 16 / cut-off 0.5, chosen
  on the gold set). On the held-out day: pair F1 0.75 (DIC + DAPI) and 0.80 (with ASH1/CLB2); details in
  `buddy/models/*.json`. Bud pairs can change slightly compared with the previous models; the new file names make buddy
  download the new weights even where the old ones are already present. The old weights stay in the release.
- Notebook 05 (train your own bud model): holds out whole imaging days (`TEST_DAYS`) instead of folder names, uses the
  last epoch and cut-off 0.5 like the shipped models, and no longer suggests picking an earlier epoch on the test images.
- Test data: reference outputs of the slow bud-model test renewed with the new model (`tests/data/field_A`, release
  `test-data-v2`).
- Bud model, mother-bud assignment (`BudModel.assign`) simplified: claims are accepted from strongest to weakest and a
  claim is skipped when its cell is already a bud, its bud is already taken, or its bud is an accepted mother. The
  old rule that let a weaker claim take a bud by scoring higher on that bud's pixels (and knock out the earlier mother)
  is gone; it never occurred on the 16 held-out test fields, which give identical pairs before and after. A close call
  is now: another cell claimed the same bud with a strength within 0.15 of the winner.
- Spot decomposition (step 3) is now repeatable: Big-FISH builds the reference spot from up to 2000 randomly chosen
  spots, so channels with more spots could give slightly different decomposed spots on every run. The random generator
  is now seeded for this step (`DECOMPOSITION_SEED = 0` in `buddy/spots/detection.py`) and restored afterwards.
  Channels with more than 2000 spots can differ slightly from results made before this change; rerun step 3 to update.
- Install: Java 11 and Maven (conda-forge `openjdk`, `maven`) are now part of the environment (installers and
  `environment.yml`), so reading `.vsi` files no longer depends on Bio-Formats downloading a Java on first use
  (which failed offline or behind a proxy). The installers also print `java -version` at the end.
- INSTALL.md: troubleshooting for "Bio-Formats does not support this image" (incomplete Bio-Formats download).

## 0.3.0 (2026-10-12)
- **Cell-cycle phase of every cell** (`buddy/quantification/cell_cycle.py`): after the spots are counted (step 3) and
  again after your corrections (step 5), every cell gets G1, S, G2 or M from its bud size, the nuclei in mother and bud
  and its ASH1/CLB2 mRNAs, written to `cell_cycle.csv` (with the reason for each call) and shown in napari-buddy 0.2.0.
  Same rules as the validated cell-cycle assignment; a 2 µm² bud with a nucleus now counts as M (it fell between S/G2
  and M). Without an ASH1/CLB2 channel unbudded cells are called "unbudded" instead of G1.
- New optional config block `"cell_cycle"` (thresholds, ASH1/CLB2 channel; `CELL_CYCLE_DEFAULTS` in `config.py`).
- Needs napari-buddy ≥ 0.2.0 to show the phases; buddy warns at import when an older napari-buddy is installed.

## 0.2.0 (2026-10-11)
- Code reorganised by pipeline step into sub-packages: `reading/` (step 1), `segmentation/` (step 2), `spots/` (step 3),
  `quantification/` (steps 3b/5) and `buds/` (the bud model); `image.py` only ties them together. No change in results
  (regression tests on the test field).
- **Import paths changed** for everything except `Experiment`, `Image` and `pipeline`: e.g. `buddy.bud_model` →
  `buddy.buds.bud_model`, `buddy.bud_training` → `buddy.buds.bud_training`, `buddy.tiff_input` → `buddy.reading.tiff_input`,
  `buddy.gpu_filters` → `buddy.spots.gpu`. `utils.py` and `Masks_alignment.py` are gone: `spot_assignment` is in
  `buddy.quantification.spot_assignment`, the alignment functions in `buddy.spots.alignment`.
- Config: unknown settings are reported in the log, missing essentials (`scale`, `spot_radius`, `sigma`, a channel's
  `"filter"`) raise a clear error (`config.py`).
- buddy logs to the `"buddy"` logger instead of configuring Python's root logger (`buddy.log.setup_logging`).
- Step 2 also writes the raw Omnipose cells (`DIC_masks.tif`), used by the bud-model training.
- Bud-model training reads the pixel size from each image's `img.json` (`scale`) instead of assuming 65 nm.
- Removed `Image.save` / `unpickle` (`img.05.pkl`): nothing read it.
- The notebooks moved from `buddy/` to `notebooks/` in the repo root, with their data files (`budSegmentationData20260311.csv`,
  its README, `test_data.csv`). Notebook 01 reads the example config from `..\buddy\config_files`; notebook 05 writes
  your own weights to `notebooks/models/`.
- Needs napari-buddy ≥ 0.1.1; buddy warns at import when an older napari-buddy is installed.

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
