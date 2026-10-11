# buddy — Tutorial

A practical, step-by-step guide to installing the smFISH pipeline (`buddy` +
`napari-buddy`) and running a full analysis, from raw `.vsi` images to a per-cell
dataframe with mother/bud RNA counts and cell-cycle phases.

`buddy` is a scripted wrapper around [Big-FISH](https://big-fish.readthedocs.io/)
that automates segmentation, spot detection, dense-region decomposition and
spot-to-cell assignment, recording every parameter so runs are reproducible.
`napari-buddy` is a companion napari plugin that gives you a GUI for tuning
detection parameters, correcting masks, and annotating and reassigning buds.

---

## Part 1 — Installation

Install buddy and napari-buddy as described in [INSTALL.md](INSTALL.md): one command on Windows, macOS and
Linux; GPU set-up, manual install and troubleshooting are there too. You need Anaconda or Miniconda and Git;
Python 3.10 is installed for you, internet is needed on the first run, and a GPU is optional.

### 1.1 Expected folder layout

The two repos must sit **side by side** in the same parent folder — the plugin is
installed by relative path and several notebooks reach across with `..\..`:

```
<parent>/
├── buddy/               <- the buddy repo
│   ├── buddy/           <- the importable package (image.py, experiment.py, …)
│   │   ├── config_files/
│   │   └── models/
│   └── notebooks/       <- 01_… to 05_… notebooks
└── napari-buddy/        <- the plugin repo
    └── napari_buddy/    (_reader.py, _bud_widget.py, _config_widget.py, …)
```

### 1.2 Two things that are slow *once*

- **First `.vsi` read** downloads a Java runtime (`Fetching zulu-jre:11 using
  cjdk…`). Needs internet once; cached afterwards. Offline/behind a proxy?
  Uncomment `openjdk=11` and `maven` in `environment.yml` to get Java from conda
  instead.
- **First segmentation** downloads the Omnipose/Cellpose model weights.

---

## Part 2 — Set up an experiment

### 2.1 The configuration file

Everything the pipeline needs to know about an experiment lives in one JSON
config (see `config_files/…/config…json`). The key fields:

- `parameters.scale` — voxel size in nm `[z, y, x]` (e.g. `[200, 65, 65]`).
- `parameters.spot_radius` — expected spot radius in nm `[z, y, x]`
  (e.g. `[880, 120, 120]`).
- `parameters.sigma` — Gaussian background-subtraction σ `[z, y, x]`.
- `parameters.patch_size` — focus-detection patch, e.g. `[200, 200]`.
- `parameters.organism` — `"S.cerevisiae"` or `"E.coli"` (selects segmentation
  parameters).
- `channels` — one entry per channel keyed by transcript, each with a `filter`
  (the microscope filter name, **must match the `.vsi` scene exactly**, e.g.
  `"CY3.5 NAR"`), an `mrna` name, a `colormap`, and an optional detection
  `threshold`.
- `parameters.min_cell_area` / `parameters.expand_px` — step 2: remove cells smaller than this
  (px, default 20) / grow every cell by this many px (default 0).
- `parameters.remove_edge_cells` — `true`: step 2 removes cells touching the image edge (they are
  incomplete: too few RNAs, bud possibly outside the image). Default `false`.
- `parameters.use_gpu` — `true` (default): run Omnipose and the bud model on the GPU
  when one is available.
- `parameters.gpu_spot_detection` — `true` (default): with a GPU, also run the background
  filter and spot detection on it (same spots as Big-FISH).
- `if_already_processed` — what a step does with images it already processed with *other* settings:
  `"ask"` (default: lists what changed and asks once), `"skip"` or `"overwrite"`. Images already
  processed with the same settings are always skipped. Re-running segmentation moves the image's
  `corrected/` masks to `corrected_replaced_<date-time>/`, since they no longer fit the new cell masks.
- `bud_classification` — automatic mother/bud assignment:
  ```json
  "bud_classification": {"enabled": true, "model": "with_ASH1CLB2",
                         "ash_channel": "ASH1CLB2", "overwrite_manual_corrections": false}
  ```
  `model` is `"DIC_DAPI_only"` (DIC + DAPI; any strain, incl. she2/she3) or
  `"with_ASH1CLB2"` (also uses the ASH1/CLB2 channel; best for small buds; not for
  she2/she3 mutants). Leave the block out or set `"enabled": false` to annotate buds by hand.

You can hand-edit the JSON, or build one in napari: **Plugins → napari-buddy →
Create config file**, fill in the channels and parameters (including the segmentation, GPU,
bud classification and re-run settings below), and save.

> **Common gotcha:** if the log says *"ignoring unknown channel token '…'"* during
> reading, a channel's `filter` in the config doesn't match the filter name the
> microscope wrote into the `.vsi` scene. Fix the `filter` string to match
> exactly (case- and spelling-sensitive).

there is also `config_test.json` in parent folder.

### 2.2 Point the notebook at your data

Open `01_fish_bulk_processing.ipynb` (this is the main pipeline notebook). In the
first configuration cell, set the config path and the input/output folders:

```python
exp = Experiment.from_cfg_file(r"..\buddy\config_files\220601\config…json")
exp.inputdir  = r"…\your data\STRAIN BRn"
exp.outputdir = r"…\your data\STRAIN BRn\output"
exp.create_image_list()
```

`create_image_list()` scans `inputdir` for `.vsi` files and their matching DIC
TIFFs and builds the list of fields to process.

---

## Part 3 — Run the pipeline (notebook 1)

The notebook is organised into five numbered stages. Run them top to bottom;
each stage writes to the per-image `output/<field>/` folder and records what it
did in `img.json`, so you can stop and resume between stages.

The same five stages can also be run from napari without the notebook: **Plugins → napari-buddy → buddy
pipeline** (one button per stage, with a progress bar and time estimate).

### Stage 1 — Read in microscope images
Reads each `.vsi` (fluorescence channels + DAPI) and the DIC image (several images
at the same time), builds the DIC+DAPI composite used for segmentation, and writes
`{channel}.tif` and `DIC.tif` per field.

### Stage 2 — Segment images
Loads `init_omnipose()` once (so the models aren't reloaded per image), then for
each field segments cells (from the DIC+DAPI composite) and nuclei (from the DAPI
projection), post-processes the masks (drops cells with no DAPI, objects < 20 px,
and nuclei with no cell; optionally cells touching the image edge, see
`remove_edge_cells`), and writes `DIC_masks_pp.tif`.

### Stage 3 — Detect spots
For each field: detects the in-focus z-range, background-subtracts each channel,
detects spots (both on the GPU when available), decomposes dense regions, then **aligns** the DIC/cell masks to
the detected spots and writes the aligned masks into `corrected/`. It copies the
segmented DAPI into `corrected/DAPI_corrected_masks.tif` (if not already there).
If `bud_classification` is enabled, the **bud model** then assigns every bud to its
mother and writes `corrected/DIC_corrected_masks.tif` (buds merged into mothers),
`corrected/bud_corrected_masks.tif` and `bud_flags.csv` (cells worth checking).
Bud masks you corrected by hand are never overwritten. Finally it runs a first
spot-to-cell assignment. Per-channel spot tables
(`{channel}_spots.npy/.csv`) and results are written.

> **Tuning tip:** if spot counts look wrong, tune the detection **before** batch
> processing using the napari-buddy widgets (Filter out background → Detect spot →
> Threshold spots → Decompose spots) on a single field, then put the chosen
> `threshold` into the config.

### Stage 4 — Check and correct *(interactive)*
Run the cell: napari opens the first image with a **Correct images** panel
(◀ previous · save · save + next ▶ · skip ▶). For each image:

1. **Check.** With automatic bud classification, the `bud masks` layer shows the model's result, the
   `bud → mother links` layer draws a red line from each bud to its mother (white circle = bud without mother),
   and `check these buds` boxes the cells the model was unsure about — red: close call between two
   mothers, orange: bud without mother, yellow: weak pair / possible missed bud (hover a box for the
   reason). Without it, small cells are pre-marked as buds (yellow boxes) and non-round cells are circled
   (cyan).
2. **Fix** what is wrong: cell/nucleus masks with the labels brush; buds by painting in `bud masks`
   (1 = bud, 2 = background, 3 = mother), or in the **Reassign buds to mothers** panel (opens automatically)
   (draw a line from a bud to its mother, click *Apply links*).
3. **Save + next.** Saving links newly painted buds to the cell they touch most, merges them into that
   mother and writes `corrected/DIC_corrected_masks.tif`, `DAPI_corrected_masks.tif` and
   `bud_corrected_masks.tif`.

The **Layer groups** panel shows / hides groups of layers. Reopening an image later loads your corrected masks.

### Stage 5 — Reassign spots
Re-runs spot-to-cell assignment on the **corrected** masks for every field,
rewriting the per-cell `{channel}.csv` tables (now with mother/bud compartments).

---

## Part 4 — Downstream analysis (notebooks 2–4)

(Notebook 05 trains your own bud model on corrected images; see the README.)

- **`02_data_wrangling_and_plotting.ipynb`** — concatenates the per-field,
  per-channel `.csv` tables into one master dataframe, attaches metadata (strain,
  condition, replicate, field, mRNA), builds the per-cell unique identifier
  `strain_replicate_cellID_fov`, and makes summary plots.
- **`03_verify_dataframe_in_napari.ipynb`** — pick any cell by its `uid` and jump
  straight back to it in napari to eyeball the mask, spots and counts.
- **`04_generate_cell_cycle_crops_from_uid.ipynb`** — **cell-cycle assignment.**
  Converts measurements to microns, restricts to sensibly sized cells (major axis
  2–20 µm), and assigns each cell to **G1 / S / G2 / M** from bud area, mother/bud
  nucleus counts and the *ASH1/CLB2* level:
  - **G1** — no bud, one mother nucleus, low *ASH1/CLB2* (≤ 5 mRNAs)
  - **S** — small bud (< 2 µm²), one mother nucleus, no bud nucleus
  - **G2** — larger bud (2–15 µm²), one mother nucleus, no bud nucleus
  - **M** — bud present with a nucleus in it (one mother + one bud nucleus)

  It also generates single-cell image crops from each `uid`/bounding box, which
  were scored by eye to validate the automatic phase calls.

---

## Part 5 — Troubleshooting quick reference

| Symptom | Likely cause / fix |
|---|---|
| `numpy` / `_ssl` DLL errors on launch | You launched bare `napari`/`jupyter`. Use `python -m napari` / `python -m notebook` inside the `buddy` env. |
| "ignoring unknown channel token '…'" | A channel `filter` in the config doesn't match the `.vsi` scene name — fix the string exactly. |
| First `.vsi` read hangs / "Fetching zulu-jre" | Java is being downloaded on first use. Let it finish (needs internet once), or provide `openjdk=11`/`maven` via conda. |
| Buds disappear before you can annotate | The DAPI-presence filter drops cells with no nucleus of their own. Paint the missing bud in the `bud masks` layer and link it with the Reassign-buds widget. |
| `assign_spots` can't find corrected masks | Stage 4's save button (and the DAPI copy in Stage 3) must have run — they create the `corrected/` masks that assignment reads. |
| "no CUDA GPU is available to PyTorch" | The CPU-only PyTorch is installed. See INSTALL.md → GPU, or ignore (runs on CPU). |
| "Could not find or download bud_model_….pt" | No internet on first use: download the weights from the GitHub release and put them in `buddy/models/` (INSTALL.md). |
| Omnipose won't install | Check Python is 3.9/3.10 (not 3.11+). On Apple Silicon see *Troubleshooting* in [INSTALL.md](INSTALL.md). |

---

## Appendix — Output files per field (`output/<field>/`)

| File | Written by | Contents |
|---|---|---|
| `DIC.tif`, `{channel}.tif` | Stage 1 | DIC image and each fluorescence channel |
| `DIC_masks_pp.tif` | Stage 2 | post-processed cell masks |
| `DAPI_masks.tif` | Stage 2 | nucleus masks (cleaned) |
| `{channel}_filtered.tif` | Stage 3 | background-subtracted channel |
| `projections/*_max.tif` | Stage 1/3 | 2D maximum projections, used by napari to open images quickly |
| `{channel}_spots.npy/.csv` | Stage 3 | detected spots (+ intensities) |
| `{channel}_decomposed_spots.npy`, `{channel}_dense_regions.npy` | Stage 3 | decomposition output |
| `aligned_DIC.tif`, `aligned_DIC_masks.tif` | Stage 3 | DIC/masks aligned to spots |
| `corrected/DIC_corrected_masks.tif` | Stage 3/4 | working cell masks (aligned, then hand-corrected + bud-merged) |
| `corrected/DAPI_corrected_masks.tif` | Stage 3/4 | working nucleus masks |
| `corrected/bud_corrected_masks.tif` | Stage 3/4 | 3-class bud annotation (1 bud, 2 bg, 3 mother), by the bud model and/or by hand |
| `bud_flags.csv`, `bud_pairs.csv`, `bud_model_cells.csv` | Stage 3 | bud model: cells to check (with reason), pairs found, output per cell |
| `{channel}.csv` | Stage 3/5 | per-cell counts + morphology (mother/bud split) |
| `img.json` | every stage | metadata, parameters, step times, which stage ran last |
