# buddy — Installation & Usage Tutorial

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

### 1.1 Prerequisites

- **Anaconda or Miniconda** (<https://docs.conda.io/en/latest/miniconda.html>).
- **Git**, to clone the two repositories.
- **Python 3.10** — pinned for you by the install step. Do **not** use 3.11+:
  Omnipose (segmentation) is only tested up to 3.10.11 and breaks on newer
  Python.
- **Internet on first run.** The first time you open an Olympus `.vsi` image,
  Bio-Formats downloads a Java runtime automatically (once; cached afterwards).
- **A GPU is optional** — Omnipose runs on CPU, just slower.
- **Disk:** a few GB for the environment, the Java runtime and the model weights
  that download on first use.

Supported platforms: Windows 10/11, macOS (Intel and Apple Silicon), Linux.

### 1.2 Expected folder layout

The two repos must sit **side by side** in the same parent folder — the plugin is
installed by relative path and several notebooks reach across with `..\..`:

```
<parent>/
├── buddy/               <- the buddy repo
│   └── buddy/           <- the importable package (image.py, experiment.py, …)
│       ├── bud_tools.py
│       ├── config_files/
│       └── 01_… to 05_… notebooks
└── napari-buddy/        <- the plugin repo
    └── napari_buddy/    (_reader.py, _bud_widget.py, _config_widget.py, …)
```

### 1.3 Quick install (recommended)

From the **`buddy` repo root**:

**Windows (Anaconda Prompt):**
```
powershell -ExecutionPolicy Bypass -File install.ps1
```

**macOS / Linux (Terminal):**
```
bash install.sh
```

The script creates a `buddy` conda environment (Python 3.10), installs `buddy`
with all dependencies (napari + Jupyter included), adds the `bioio-ome-tiff`
writer plugin, installs `napari-buddy` if it sits next to the repo, and runs an
import check.

### 1.4 Manual install (if you prefer)

```
conda create --name buddy python=3.10
conda activate buddy
python -m pip install -e ".[test,napari,notebooks]"   # buddy + deps + pytest + napari + jupyter
python -m pip install bioio-ome-tiff                   # OME-TIFF writer (moved out of bioio core in 3.x)
python -m pip install -e ../napari-buddy               # the napari plugin (adjust the path)
```

Use `python -m pip` so installs land in the **active** env, not base conda.

> **Apple Silicon:** if `pip install -e .` fails on the Omnipose step, comment out
> `omnipose` in `pyproject.toml`, rerun the install, then:
> ```
> pip install git+https://github.com/kevinjohncutler/omnipose.git
> pip install git+https://github.com/kevinjohncutler/cellpose-omni.git
> ```

### 1.5 Verify the install

```
conda activate buddy
python -c "import buddy, bigfish, omnipose, bioio_ome_tiff; print('buddy imports OK')"
pytest buddy/tests/test_image.py -v      # full pipeline test on bundled example data
python -m napari                         # launch viewer; check Plugins → napari-buddy
```

### 1.6 Two things that are slow *once*

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

You can hand-edit the JSON, or build one in napari: **Plugins → napari-buddy →
Create config file**, fill in the channels/parameters, and save.

> **Common gotcha:** if the log says *"ignoring unknown channel token '…'"* during
> reading, a channel's `filter` in the config doesn't match the filter name the
> microscope wrote into the `.vsi` scene. Fix the `filter` string to match
> exactly (case- and spelling-sensitive).

there is also `config_test.json` in parent folder.

### 2.2 Point the notebook at your data

Open `01_fish_bulk_processing.ipynb` (this is the main pipeline notebook). In the
first configuration cell, set the config path and the input/output folders:

```python
exp = Experiment.from_cfg_file(r"config_files\220601\config…json")
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

### Stage 1 — Read in microscope images
Reads each `.vsi` (fluorescence channels + DAPI) and the DIC image, builds the
DIC+DAPI composite used for segmentation, and writes `{channel}.tif` and
`DIC.tif` per field.

### Stage 2 — Segment images
Loads `init_omnipose()` once (so the models aren't reloaded per image), then for
each field segments cells (from the DIC+DAPI composite) and nuclei (from the DAPI
projection), post-processes the masks (drops cells with no DAPI, objects < 20 px,
and nuclei with no cell), and writes `DIC_masks_pp.tif`.

### Stage 3 — Detect spots
For each field: detects the in-focus z-range, background-subtracts each channel,
detects spots, decomposes dense regions, then **aligns** the DIC/cell masks to
the detected spots and writes the aligned masks into `corrected/`. It copies the
segmented DAPI into `corrected/DAPI_corrected_masks.tif` (if not already there)
and runs a first spot-to-cell assignment. Per-channel spot tables
(`{channel}_spots.npy/.csv`) and results are written.

> **Tuning tip:** if spot counts look wrong, tune the detection **before** batch
> processing using the napari-buddy widgets (Filter out background → Detect spot →
> Threshold spots → Decompose spots) on a single field, then put the chosen
> `threshold` into the config.

### Stage 4 — Correct cell masks and add bud mask *(interactive)*
This is the manual QC step, done in napari. Pick a field (`i = 0`) and run the
loader cell — it opens napari with the DIC, masks, channels and spots loaded, and
spots coloured by cell. Then:

1. **Correct the masks.** Fix cell/nucleus masks with the labels brush as needed.
2. **Seed the bud layer.** Run the bud-annotation cell. It creates a `bud masks`
   layer (`1 = bud, 2 = background, 3 = mother`), pre-seeds small cells as buds
   (yellow boxes), and flags non-round cells as possible unsplit mother–bud pairs
   (cyan ellipses). Paint to re-classify: select label 1/2/3 and paint.
3. **Link buds to mothers.** Run the linking cell to auto-link each painted bud to
   the cell it most adjoins, merge it into that mother in the DIC mask, and box
   each mother–bud pair (magenta) and any orphan bud (red). For anything the
   automatic pass missed, use **Plugins → napari-buddy → Reassign buds to
   mothers**: click *New line layer*, draw a line from a bud to its mother (either
   direction; a bud you painted but that has no cell of its own is created on the
   spot), and click *Apply links*.
4. **Save.** Run the save cell to write
   `corrected/DIC_corrected_masks.tif`, `DAPI_corrected_masks.tif` and
   `bud_corrected_masks.tif`.

Reopening a field later reloads your corrected masks automatically.

### Stage 5 — Reassign spots
Re-runs spot-to-cell assignment on the **corrected** masks for every field,
rewriting the per-cell `{channel}.csv` tables (now with mother/bud compartments).

---

## Part 4 — Downstream analysis (notebooks 2–4)

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
| `assign_spots` can't find corrected masks | Stage 4's save cell (and the DAPI copy in Stage 3) must have run — they create the `corrected/` masks that assignment reads. |
| Omnipose won't install | Check Python is 3.9/3.10 (not 3.11+). On Apple Silicon see the note in §1.4. |

---

## Appendix — Output files per field (`output/<field>/`)

| File | Written by | Contents |
|---|---|---|
| `DIC.tif`, `{channel}.tif` | Stage 1 | DIC image and each fluorescence channel |
| `DIC_masks_pp.tif` | Stage 2 | post-processed cell masks |
| `DAPI_masks.tif` | Stage 2 | nucleus masks (cleaned) |
| `{channel}_filtered.tif` | Stage 3 | background-subtracted channel |
| `{channel}_spots.npy/.csv` | Stage 3 | detected spots (+ intensities) |
| `{channel}_decomposed_spots.npy`, `{channel}_dense_regions.npy` | Stage 3 | decomposition output |
| `aligned_DIC.tif`, `aligned_DIC_masks.tif` | Stage 3 | DIC/masks aligned to spots |
| `corrected/DIC_corrected_masks.tif` | Stage 3/4 | working cell masks (aligned, then hand-corrected + bud-merged) |
| `corrected/DAPI_corrected_masks.tif` | Stage 3/4 | working nucleus masks |
| `corrected/bud_corrected_masks.tif` | Stage 4 | 3-class bud annotation (1 bud, 2 bg, 3 mother) |
| `{channel}.csv` | Stage 3/5 | per-cell counts + morphology (mother/bud split) |
| `img.json` | every stage | metadata + parameters + which stage ran last |
