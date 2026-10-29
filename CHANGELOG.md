# Changelog

## 0.6.0 (2026-10-29)
**The config file has one block per pipeline step.** `"parameters"` held spot detection, segmentation, clean-up,
GPU and TIFF settings together, and the Omnipose settings were split between it and `"segmentation"`. Now:
`channels` (step 1), `spots` (step 3: scale, spot_radius, sigma, patch_size, decomposition), `segmentation` (step 2:
organism, models, tile, min sizes, use_preset, cells / nuclei, clean-up), `bud_classification`, `cell_cycle`, then
the rarely changed `hardware` (use_gpu, gpu_spot_detection), `tiff_input` (step 1, TIFF stacks) and `experiments`.
Each step reads one block (`buddy.where()`, README code map).
- **Omnipose settings in the open.** `segmentation` has `use_preset` (true in the example: the organism's preset is
  used and `cells` / `nuclei` spell out its diameter, mask_threshold and flow_threshold, so switching `organism`
  is enough; false or absent, as in every older config: `cells` / `nuclei` override the preset as before). With
  `use_preset` true, differing `cells` / `nuclei` values are reported in the log and ignored. The napari panel's
  Omnipose tab has the same switch.
- **Decomposition settings in the config.** `spots.decomposition` = `{"alpha": 0.5, "beta": 2, "gamma": 1}`,
  the Big-FISH `decompose_dense` settings, the same for every channel (were constants in `spots/detection.py`);
  recorded for `which_images` only when changed, so earlier results are not reported as different.
- `cell_flags` is no longer in the example and is only written when it differs from the defaults: its settings
  are not meant to be tuned.
- Fixed: Python's `faulthandler` is left off once the Java VM has started (it was switched back on after the
  start). The JVM handles SIGSEGV itself (safepoints, null checks); with faulthandler's handler in place the process
  aborted on Linux, typically during a garbage collection while reading a large `.vsi` - the crash of the tests on
  GitHub's Linux runner. Windows was not affected.
- Fixed: the GPU decomposition multiplied its filter kernel as `tuple * gamma`, which repeats the tuple instead of
  scaling it (harmless at the fixed `gamma=1`, wrong for any other value, an error for a float such as `1.0`).
- **Old configs keep working.** A file with a `"parameters"` block is read as before (the blocks win where both
  are given); nothing changes in memory (`exp.parameters`, `img.json`, `which_images`), so old results are not
  reported as "processed with other settings". `buddy.config.write_config` saves any config in the new layout.
- **Readable files.** `write_config` / `to_json` write 2-space JSON with short lists on one line and a blank line
  between blocks; keys that start with `_` are comments and ignored. `config_files/config_example.json` is
  regenerated from `config.example_config()` (`test_config_layout.py` keeps it in step with `config.py`): every
  setting with its default, one `_about` comment per block, the example set's channels.
- Relative `inputdir` / `outputdir` in a config file are relative to the config file's folder (was: the current
  directory); the example points at `../../test yourself/...` from `config_files/`.
- A real `.vsi` in the test data: release `test-data-vsi` (`vsi.zip`, ~400 MB: one image of the example set with
  its `_..._` folder and `_DIC.tif`) is fetched for the slow `tests/test_vsi_file.py`, which checks the series
  names, every channel stack (checksums in `vsi_reference.json`, made by `tests/make_vsi_reference.py`) and the
  DIC pairing. The fast Bio-Formats test keeps using a small OME-TIFF.
- CI: `slow_tests.yml` runs the slow tests (Omnipose and the bud model on the CPU, the real `.vsi`) every Monday
  and on demand; `tests.yml` keeps running the fast tests on every push.
- The aliases of 0.5.0 for napari-buddy <= 0.3 are gone: `buddy/rerun_check.py`, `buddy/spots/gpu.py`,
  `Image.load_data`, `Image.save_metadata` (napari-buddy 0.4.0 uses the real names).
- Error messages and the code map name the new blocks (`"hardware": "use_gpu"`, ...). napari-buddy 0.4.0 writes
  the new layout.

## 0.5.2 (2026-10-29)
Two dependencies replaced by buddy's own code; fewer packages, and Python 3.12 works.
- **Bio-Formats without bioio.** `buddy/reading/vsi.py` reads microscope files through Bio-Formats directly (JPype);
  `bioio` + `bioio-bioformats` (58 packages: dask, xarray, ome-types, pydantic, ...) are gone. Bio-Formats itself now
  comes as the pip package `bioformats_jar` (the 6.7.0 jar, the same version bioio used), so Maven is no longer
  installed and nothing is downloaded on the first `.vsi` read (the `.jgo` problems of the troubleshooting table
  are history). The planes are read into one reused Java array viewed by numpy - faster than bioio's dask path.
  `BUDDY_VSI_READER=bioio` still uses bioio when it is installed, for comparing. Series names, channel order and
  the arrays are the same; verified on the example set.
- **The bud model's U-Net in plain PyTorch.** `buddy/buds/unet.py` is segmentation_models_pytorch's
  `Unet("resnet34")` written out with torchvision's resnet34 as the encoder, with the same module names: the released
  weights load unchanged (strict) and the outputs are bit-identical (test_unet.py checks against smp when it is
  installed). `segmentation-models-pytorch` (+ timm, huggingface-hub, safetensors, httpx, ...) is gone. Training
  (notebook 05) works the same; `pretrained_encoder=True` takes torchvision's ImageNet weights.
- Test data `test-data-v3`: the step-3 / step-5 reference outputs of the regression test were still those of the
  detection before 0.4.0 (only the bud-model reference had been renewed), so `pytest tests/test_regression.py`
  failed everywhere. Renewed with the current code (SRL1 on the crop: 363 -> 507 spots, the dim bottom-of-focus
  spots 0.4.0 keeps); the test, `conftest.py` and CI point at `test-data-v3`.
- CI installs with the constraint files too.
- **Fix of 0.5.1: Omnipose silently ran as plain Cellpose.** The omnipose wheel ships its own `cellpose_omni`; the
  installers also installed the separate `cellpose-omni==0.9.1` from PyPI, which overwrote it with an older copy
  whose Omnipose check fails -> `>>cyto<< model set to be used`, 0 cells, `niter` error on the nuclei. Only
  `omnipose` is installed now, and `masks.check_omnipose()` (installers, `load_models`) verifies that the Omnipose
  models are really available. To repair an environment made by 0.5.1: `pip uninstall cellpose-omni`,
  `pip install --force-reinstall --no-deps omnipose==1.1.4`.
- Fix of 0.5.1: the list of what Omnipose needs was too short - `import cellpose_omni.models` also pulls in torchvf,
  cmap, colour-science, pyinstrument, imagecodecs and aicsimageio, so step 2 failed on a fresh install. The first five
  are in `pyproject.toml` now; for `aicsimageio` (~25 packages, used by nothing buddy calls) `masks.py` registers an
  empty stand-in module when it is not installed. Checked by importing and segmenting in a bare environment. A core
  install is ~70 packages (0.4.0: ~150), a full one ~200 (0.4.0: ~260).
- **Python 3.12.** Python 3.10 reaches end of life in October 2026, and Omnipose's "3.10 only" is history. The
  installers, the constraint files and CI now use 3.12 (`requires-python >=3.10,<3.13`; 3.13 has no PyTorch 2.5.1).
  The package versions are the same ones as before (our 3.10 environment, to the version); checked on 3.12: all
  tests, Omnipose masks and spots identical to the 3.10 reference. numpy 2 (with the newest scipy / scikit-image /
  numba) was checked too and gives the same masks and spots, so `numpy<3` is allowed, but the constraints keep 1.26.
  One drift caught on the way: the newest `ncolor` breaks Omnipose 1.1.4 - the constraints pin 1.5.3.
- **Output folders can move.** `Image.from_json` takes the folder the `img.json` is in as the image folder, whatever
  the stored `savepath` says (an absolute path of the computer that made it). A processed dataset copied to another
  drive, computer or OS now just works (steps, napari, notebooks); the next save writes the new location.

## 0.5.1 (2026-10-28)
Installation: the same packages on every machine, and fewer of them.
- **Pinned versions.** `constraints/windows.txt`, `linux.txt`, `macos.txt` hold the version of every package buddy
  is tested with (the Windows file is our working environment to the version); `install.sh` / `install.ps1` pass
  them to pip (`-c`). Until now nothing was pinned, so every install resolved to whatever was newest that day -
  new releases of Big-FISH, scikit-image or Omnipose changed the spots or broke the install. PyTorch is pinned to
  2.5.1 in `pyproject.toml`.
- **GPU on Linux fixed.** The installers swapped in the CUDA 12.1 PyTorch with `--no-deps`; on Linux that wheel
  needs the `nvidia-*` packages that `--no-deps` left out, so `import torch` failed. On Linux the PyPI wheel of
  torch 2.5.1 is already a CUDA build with those libraries as dependencies: no swap. Windows keeps the swap (the
  cu121 wheel is self-contained); macOS is CPU.
- **Fewer packages.** `omnipose` on PyPI pulls ~100 packages (aicsimageio, networkit, dask-image, pyinstrument, ...);
  segmentation uses a dozen. Those are now listed in `pyproject.toml` and the installers add `omnipose` +
  `cellpose-omni` with `--no-deps`. The OME-TIFF channel stacks are written with `tifffile` (`ome=True`, same
  axes, channel name and compression) instead of `bioio-ome-tiff`; `jsonpickle` and `natsort` (never imported) and
  `jupyterlab` (notebook is enough) are dropped. A full install is ~220 packages instead of ~280, a core one ~110.
- `install.ps1` now does the same steps as `install.sh` (removes an existing env first, stops on the first error).
- `masks.load_models` says what to install when `cellpose_omni` is missing.
- (Corrected in 0.5.2: 0.5.1's installers also installed the `cellpose-omni` package from PyPI, which overwrites the
  copy inside the omnipose wheel and turns Omnipose into plain Cellpose. Only `omnipose` is installed now.)

## 0.5.0 (2026-10-28)
Changes after Zsloth's review of the code and the installation on Linux - thank you.
- **One place per thing.** The processing steps are written out in `buddy/pipeline.py` (`segment_cells`,
  `find_focus`, `classify_buds`, ... one function per sub-step, each reading one block of the config file); `Image`
  (`buddy/image.py`) only holds an image's files, arrays and `img.json`. The Omnipose code moved from
  `segmentation/omnipose.py` (the name of the library) to `segmentation/masks.py`, the GPU versions of the Big-FISH
  steps from `spots/gpu.py` to `spots/gpu_detection.py` (`spots/gpu.py` stays as an alias for one version), and `rerun_check.py` -
  which images a step still has to do - to `which_images.py` (alias kept); no function name exists twice any more. `buddy.where()` prints where each step lives (also the "Where things are" table in README.md).
  The outputs are unchanged (checked file by file against 0.4.0 on the test field); the public functions
  (`pipeline.read_image`, `segment_image`, `spots_image`, `buds_and_counts`, `reassign_image`, `Experiment.init_omnipose`)
  kept their names.
- **Errors say where and what to do.** An error inside a step is raised as `pipeline.StepError`: "step 2 (segment)
  of image X failed in buddy/segmentation/masks.py:71 in run_omnipose(): ..." with a hint for known causes (GPU out
  of memory -> `"tile": true` in step 2, `gpu_spot_detection` / `use_gpu` false in step 3; a file of an earlier step missing -> which step writes it). `Image.load()` raises a
  `MissingFileError` naming the step that makes a file instead of returning None; files a step can do without are asked
  for with `optional=True` and logged.
- **Omnipose settings in the config file**: new optional block `"segmentation"` (`cell_model`, `nucleus_model`,
  `tile`, `cells`, `nuclei`; defaults and presets in `buddy/config.py`). `"tile": true` is the fix for *CUDA out of
  memory* in step 2 on small GPUs. The settings used are recorded in `img.json` and in the re-run check.
  napari-buddy 0.3.2 has an *Omnipose* tab in *Create config file* for it.
- Install: one `install.sh` for macOS and Linux (temporary files under `~/.cache`, an existing environment is
  replaced, CPU PyTorch first on Linux without a GPU, `napari[pyqt5]` on Linux); INSTALL.md is the only place
  installation is described. Notebooks: a `ROOT` cell (the folder above the repositories) and forward-slash paths,
  so they run on every operating system and from any copy of the folder; `DATA` / `CFG_FILE` for your own data;
  warnings shown by default (`SHOW_WARNINGS`); notebook 04 reads the example table.

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
- Spot detection (step 3): the background filter and the LoG detection run on the whole stack, and the in-focus range
  is applied to the detected spots afterwards. Before, the stack was cut at the in-focus range first, which made the
  lowest in-focus plane an edge of the LoG filter and a ridge of false local maxima there (hidden by dropping the two
  lowest in-focus planes, `DROP_BOTTOM_SLICES`, now gone). Spot counts change: dim spots near the bottom of the focus
  range are kept and the false ones are gone; rerun step 3 (and 5) to update existing results.
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
