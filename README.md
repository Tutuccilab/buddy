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
- **Cell cycle** — the cell-cycle phase (G1 / S / G2 / M) of every cell (see *Cell-cycle phases* below)
- **Track** - Processing parameters are recorded per image (JSON) so runs stay reproducible.

What changed per version: [CHANGELOG.md](CHANGELOG.md). Version 0.1.0 adds automatic bud classification and GPU spot detection; the manuscript analysis used 0.0.1.

# Expected inputs
The module expects a directory containing the following files:
- Fluorescent channels: .vsi microscopy files and the corresponding vsi directories, **or** multi-page TIFF
  stacks (OME-TIFF or Fiji/ImageJ hyperstacks with the channels as z-stacks, e.g. `image_01.ome.tif`). The
  channel names in the file are matched to the `"filter"` names in the config; a file without channel names
  needs `"tiff_channel": <index>` in each `"channels"` entry, a file without axis information `"tiff_axes"`
  (e.g. `"ZCYX"`) in the `"tiff_input"` block of the config.
- Cell outlines: 2D .tif files (e.g. DIC images), named `<image>_DIC.tif`; for TIFF stacks the DIC image may
  also be a channel of the stack named `DIC` (its middle z plane is used)
- Configuration: a json file containing channel and processing parameters: start from `config_files/config_example.json`
  (every setting written out; explained in [TUTORIAL.md](TUTORIAL.md), section 2.1), or build one in napari:
  Plugins → napari-buddy → Create config file.

# Download code and example data
Go to https://doi.org/10.5281/zenodo.22900927 to download buddy, napari-buddy and some test data.
The folder structure of the zipped file should allow you to directly start with the tutorial after installation.
Note: newer versions of the code can be downloaded by cloning this repo.

# Installation

One command per operating system, from the `buddy` repo root; everything about installing (GPU, Linux and macOS
notes, manual install, troubleshooting) is in **[INSTALL.md](INSTALL.md)** and nowhere else.

# Where things are

One step = one function in `buddy/pipeline.py`, written out as a few named sub-steps; each sub-step reads one block
of the config file and writes one or two files of the image folder. `python -c "import buddy; buddy.where()"`
prints this table (or `buddy.where("segment")` for one step); errors inside a step say the step, the image, the
file and function in buddy, and what to do when the cause is known.

| step | function in `pipeline.py` | the work is done in | config block | writes |
|---|---|---|---|---|
| 1 read | `read_image` | `reading/readers.py`, `reading/vsi.py` (.vsi, Bio-Formats), `reading/tiff_input.py` | `channels` (`tiff_input` for TIFF stacks) | `{channel}.tif`, `DIC.tif`, `projections/` |
| 2 segment | `segment_cells`, `segment_nuclei` | `segmentation/masks.py` (Omnipose) | `segmentation` (organism, models, tile, min sizes, cells / nuclei) | `DIC_masks.tif`, `DAPI_masks.tif` |
|  | `clean_up_masks` | `segmentation/cleanup.py` | `segmentation` (min_cell_area, expand_px, remove_edge_cells) | `DIC_masks_pp.tif`, `DAPI_masks.tif` |
| 3 spots | `find_focus`, `filter_background`, `detect_spots`, `decompose_spots` | `spots/detection.py` (Big-FISH; `spots/gpu_detection.py` on the GPU) | `spots` (scale, spot_radius, sigma, patch_size, decomposition), `channels` (threshold), `hardware` | `{channel}_spots.npy`, `_filtered.tif`, `_decomposed_spots.npy`, `_dense_regions.npy` |
|  | `align_masks` | `spots/alignment.py` | - | `aligned_DIC.tif`, `aligned_DIC_masks.tif`, `corrected/` masks |
|  | `classify_buds` | `buds/bud_model.py`, `buds/unet.py` | `bud_classification` | `corrected/bud_corrected_masks.tif`, `bud_flags.csv`, `bud_pairs.csv` |
|  | `count_spots`, `flag_cells`, `assign_cell_cycle` | `quantification/spot_assignment.py`, `cell_flags.py`, `cell_cycle.py` | `cell_flags`, `cell_cycle` | `{channel}.csv`, `cell_flags.csv`, `cell_cycle.csv` |
| 4 check | - | napari-buddy (Correct images, Buds) | - | `corrected/` masks |
| 5 recount | `count_spots`, `flag_cells`, `assign_cell_cycle` | as in step 3 | as in step 3 | `{channel}.csv`, `cell_flags.csv`, `cell_cycle.csv` |

The config file has one block per step (`config_files/config_example.json` shows every setting with its default and
a comment per block). Everything else: `config.py` (defaults, checks, the readable writer), `experiment.py` (the config + the loaded models),
`image.py` (one image's files, arrays and `img.json`), `paths.py` (file names), `which_images.py` (which images a
step still has to do), `parallel.py` (several images at once).

# Typical workflow
1. Batch process inputs to TIF files: `notebooks/01_fish_bulk_processing.ipynb`
2. Find good segmentation parameters using Omnipose GUI
3. Batch segment DIC and DAPI pictures: `notebooks/01_fish_bulk_processing.ipynb`
4. Find good spot detection parameters using napari-buddy plugin
   1. Load an img.json file in napari
   2. Tweak spot detection parameters
5. Batch detect spots, align, **assign buds automatically** and count spots: `notebooks/01_fish_bulk_processing.ipynb`
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

On a whole imaging day the models had never seen, they assign buds to the right mother with a pairing F1 of 0.80
(`with_ASH1CLB2`) and 0.75 (`DIC_DAPI_only`); large buds remain the weakest class, so always check the flagged cells in napari.

All settings, the flags in napari and the output files: [BUD_MODEL.md](BUD_MODEL.md). How the models were trained
and evaluated: [methods.md](methods.md). Train your own model: `notebooks/05_train_bud_model.ipynb`.

# Cell-cycle phases

After counting (step 3, and again in step 5 with your corrected masks) buddy gives every cell a phase, written to
`cell_cycle.csv` (with the reason for each call) and shown in napari as the `cell cycle` layer (S, G2 and M written on the cells):

| phase | rule |
|---|---|
| G1 | no bud, ≤ 5 ASH1/CLB2 mRNAs |
| S | bud < 2 µm², no nucleus in the bud |
| G2 | bud 2–15 µm², no nucleus in the bud |
| M | bud ≥ 2 µm² with a nucleus in it |

Cells with a mother that does not have exactly one nucleus, or a length outside 2–20 µm, are not assessed; other
combinations are `NA`. Without an ASH1/CLB2 channel unbudded cells are `unbudded`.
The thresholds are in the optional `"cell_cycle"` block of the config file. The S / G2 split rests on the bud size,
so check the buds first.

## Issues

If you encounter a problem, please [file an issue] with a detailed description.

[file an issue]: https://github.com/Tutuccilab/buddy/issues
