# Automatic bud classification — user guide

How to switch the bud model on, what its settings do, and how to check its result in napari. How it was trained and evaluated: [methods.md](methods.md), section *Automatic bud classification*.

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
* `"dic_rotation"` (default null): the DIC images are turned by a multiple of 90° so their lighting matches the training
  images. By default the lighting direction is measured on each image; buddy warns when that measurement is unclear
  (consistency < 0.9, e.g. blurry DIC; also written to `img.json`). The lighting is set by the microscope, so for one
  imaging session you can fix the turn instead: 0, 1, 2 or 3 (× 90°), e.g. the value most images of that session got
  (`"dic_rotation_k"` in their `img.json`).

**Check the result in napari** (step 4 of the bulk-processing notebook). Cells the model was
unsure about are boxed in the `check these buds` layer:

* red — *close call*: two cells claim the same bud almost equally strongly
* orange — *bud without mother*: looks like a bud, but no mother claimed it
* yellow — *weak pair* / *possible missed bud*: just above or below the model's cut-off

Unbudded cells with high ASH1/CLB2 (≥ 8× a typical unbudded cell in the image) are boxed too, with or without the
bud model, whenever an ASH1/CLB2 channel is counted (`buddy/quantification/cell_flags.py`, listed in `cell_flags.csv`):

* magenta — *late-M pair?*: the cell and an unbudded neighbour each have their own full nucleus: mother + daughter
  in late M (pair them if they belong together), or two cells that have just divided
* green — *high ASH1/CLB2, no bud*: a newborn daughter, or a cell in early S whose small bud is not marked
  (hover text gives a rough DNA-content hint from DAPI: ~1 = G1-like, higher = S/G2-like)

Settings (optional): `"cell_flags": {"enabled": true, "ash_channel": null, "fold": 8, "nucleus_area": 0.5,
"nucleus_intensity": 0.7, "max_gap_px": 10}` (`ash_channel` null = the one of `bud_classification`).

The `bud → mother links` layer draws a line from each bud to its mother, and the **Layer groups** panel
(napari-buddy ≥ 0.1.0) shows / hides whole groups of layers. Flagged cells are also listed, with the reason, in `bud_flags.csv` in each image folder (`bud_pairs.csv`
lists the pairs found, `bud_model_cells.csv` the model output per cell).

**How good is it?** Tested on every cell of 14 fields from an imaging day the models never saw (240909), with the
epoch and cut-off fixed in advance, the models assign buds to the right mother with a pairing F1 of 0.80
(`with_ASH1CLB2`, 95% CI 0.76–0.83; precision 0.91, recall 0.72) and 0.75 (`DIC_DAPI_only`, 0.71–0.79; precision 0.80,
recall 0.72). Large buds (> 4 µm²) remain the weakest class (61–65% found), so always check the flagged cells.
Details: `methods.md`, scores per model: `buddy/models/bud_model_<model>.json`.

**Train your own model** on your corrected images with `notebooks/05_train_bud_model.ipynb`
(code: `buddy/buds/bud_training.py`; the model itself: `buddy/buds/bud_model.py`). Perhaps handy if you got your own bud data, perhaps with a different 
optical setup to which you want to train a better suited model.  
