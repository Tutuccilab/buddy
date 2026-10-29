"""
The processing steps for one image - the only place where a step is written out. 01_fish_bulk_processing.ipynb and
the napari "buddy pipeline" panel (napari-buddy) call these functions, so both always do exactly the same.

    read_image(params, exp)          step 1   read the .vsi + DIC, write {channel}.tif and DIC.tif
    segment_image(json_file, exp)    step 2   Omnipose cell + nucleus masks, clean-up     (first: exp.init_omnipose())
    spots_image(json_file, exp)      step 3a  focus, background filter, spot detection, decomposition, alignment
    buds_and_counts(json_file, exp)  step 3b  bud model, count spots per cell, cell flags, cell-cycle phases
    reassign_image(json_file, exp)   step 5   count spots and assign phases again with the corrected masks of step 4

Each step is a few named sub-steps (the functions below it: find_focus, detect_spots, classify_buds, ...). Each
sub-step reads its settings from one block of the config file and writes one or two files of the image folder;
CODE_MAP lists, per sub-step, the function here, the module that does the work and the config block - printed by
buddy.where("segment"). Every step records its times and settings in the image's img.json.

An error inside a step is re-raised as StepError("step 2 (segment) of image X, in buddy/segmentation/masks.py: ...")
with a hint when the cause is known (GPU out of memory, a file of an earlier step missing).
"""
import logging
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import io

from buddy.image import Image, MissingFileError
from buddy.paths import ImageFiles

log = logging.getLogger("buddy")

STEP_TIMES = {
    'read': ['01-configure'],
    'segment': ['02-segment-cells', '02-segment-dapi', '02-segment-pp'],
    'spots': ['03-detect-spots', '04-decompose-spots', 'align', 'bud-classification', '05-assign-spots'],
    'reassign': ['05-assign-spots'],
}

# step -> (sub-step, function here, module that does the work, config block it reads, files it writes)
CODE_MAP = {
    "1 read": [
        ("read the microscope files", "read_image", "buddy/reading/readers.py, reading/vsi.py (.vsi: Bio-Formats), reading/tiff_input.py", '"channels" ("tiff_input" for TIFF stacks)', "{channel}.tif, DIC.tif, projections/"),
    ],
    "2 segment": [
        ("cell masks", "segment_cells", "buddy/segmentation/masks.py (Omnipose)", '"segmentation": organism, cell_model, tile, min_size, cells', "DIC_masks.tif"),
        ("nucleus masks", "segment_nuclei", "buddy/segmentation/masks.py (Omnipose)", '"segmentation": nucleus_model, nuclei_min_size, nuclei', "DAPI_masks.tif"),
        ("clean-up", "clean_up_masks", "buddy/segmentation/cleanup.py", '"segmentation": min_cell_area, expand_px, remove_edge_cells', "DIC_masks_pp.tif, DAPI_masks.tif"),
    ],
    "3 spots": [
        ("in-focus slices", "find_focus", "buddy/spots/detection.py", '"spots": patch_size', "img.json (results)"),
        ("background filter", "filter_background", "buddy/spots/detection.py (+ spots/gpu_detection.py)", '"spots": sigma; "hardware": gpu_spot_detection', "-"),
        ("spot detection", "detect_spots", "buddy/spots/detection.py (Big-FISH; + spots/gpu_detection.py)", '"channels": threshold; "spots": scale, spot_radius', "{channel}_spots.npy/.csv, {channel}_filtered.tif"),
        ("dense regions", "decompose_spots", "buddy/spots/detection.py (Big-FISH)", '"spots": scale, spot_radius, decomposition', "{channel}_decomposed_spots.npy, _dense_regions.npy, _reference_spot.tif"),
        ("DIC alignment", "align_masks", "buddy/spots/alignment.py", "-", "aligned_DIC.tif, aligned_DIC_masks.tif, corrected/*_masks.tif"),
        ("bud model", "classify_buds", "buddy/buds/bud_model.py, buds/unet.py", '"bud_classification"', "corrected/bud_corrected_masks.tif, bud_flags.csv, bud_pairs.csv"),
        ("spots per cell", "count_spots", "buddy/quantification/spot_assignment.py", "-", "{channel}.csv"),
        ("cell flags", "flag_cells", "buddy/quantification/cell_flags.py", '"cell_flags"', "cell_flags.csv"),
        ("cell-cycle phases", "assign_cell_cycle", "buddy/quantification/cell_cycle.py", '"cell_cycle"', "cell_cycle.csv"),
    ],
    "4 check": [
        ("correct masks and buds in napari", "-", "napari-buddy (Correct images, Buds tabs)", "-", "corrected/*_masks.tif"),
    ],
    "5 recount": [
        ("spots per cell, flags, phases with the corrected masks", "count_spots, flag_cells, assign_cell_cycle", "as in step 3", "as in step 3", "{channel}.csv, cell_flags.csv, cell_cycle.csv"),
    ],
}

# What is happening right now, for a GUI (napari-buddy sets this): progress_callback(image stem, text). Optional.
progress_callback = None

STEP_LABELS = {
    'read': 'reading the microscope files', 'grgb': 'saving channels + projections',
    '02-segment-cells': 'segmenting cells (Omnipose)', '02-segment-dapi': 'segmenting nuclei (Omnipose)',
    '02-segment-pp': 'cleaning up the masks', 'load': 'loading stacks + masks', 'focus': 'finding the in-focus slices',
    'filter': 'background filter', 'detect': 'detecting spots', '04-decompose-spots': 'decomposing dense spots',
    'align': 'aligning masks to the spots', 'bud-classification': 'bud model', '05-assign-spots': 'counting spots per cell',
}


class StepError(RuntimeError):
    """An error inside a pipeline step, with the step, the image and the file it happened in; .cause is the original."""

    def __init__(self, message, cause):
        super().__init__(message)
        self.cause = cause


def _hint(error, step=None):
    """What to do about an error whose cause is known; '' otherwise. step: the step number, for GPU errors."""
    text = str(error).lower()
    if "out of memory" in text and ("cuda" in text or "gpu" in text):
        if step == 2:
            return ('The GPU ran out of memory. Set "tile": true in the "segmentation" block of the config file '
                    '(Omnipose on tiles), or "use_gpu": false in "hardware" (CPU).')
        return ('The GPU ran out of memory. In "hardware" of the config file set "gpu_spot_detection": false '
                '(filter, detection and decomposition on the CPU) and/or "use_gpu": false (bud model on the CPU).')
    if isinstance(error, MissingFileError):
        return "Run the step that writes this file first (the steps skip images they already did: see which_images.py)."
    if isinstance(error, FileNotFoundError):
        return "A file is missing; the path above says which. Check inputdir / outputdir in the config file."
    return ""


def _where(error):
    """buddy file and function where the error was raised (the innermost buddy frame of the traceback)."""
    frames = [f for f in traceback.extract_tb(error.__traceback__) if "buddy" in f.filename.replace("\\", "/")]
    if not frames:
        return "a library buddy calls"
    f = frames[-1]
    path = f.filename.replace("\\", "/")
    return f"{path[path.rfind('buddy/'):]}:{f.lineno} in {f.name}()"


class _step:
    """with _step(2, "segment", image): ... -> errors inside become a StepError that says where and what to do."""

    def __init__(self, number, name, image):
        self.number, self.name, self.image = number, name, image

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is None or isinstance(exc, (StepError, KeyboardInterrupt)):
            return False
        stem = getattr(self.image, "stem", "?")
        message = f"step {self.number} ({self.name}) of image {stem} failed in {_where(exc)}: {exc.__class__.__name__}: {exc}"
        hint = _hint(exc, self.number)
        if hint:
            message += f"\n  -> {hint}"
        log.error(message)
        raise StepError(message, exc) from exc


def _report(image, key):
    if progress_callback is not None:
        try:
            progress_callback(getattr(image, 'stem', ''), STEP_LABELS.get(key, key))
        except Exception:                 # a GUI problem must not stop the processing
            pass


def _timed(image, key, function, *args):
    _report(image, key)
    tic = time.time()
    function(image, *args)
    image.time[key] = time.time() - tic


# =================================================================================================
# step 1: read
# =================================================================================================
def read_image(params, exp, mode='serial'):
    """Step 1: read the microscope files of one field (params = an entry of exp.images). Returns its img.json."""
    image = Image.from_dict(params, exp)
    with _step(1, "read", image):
        tic = time.time()
        _report(image, 'read')
        image.read_image()                                   # .vsi / TIFF stack: fluorescence channels + DAPI
        image.read_cells()                                   # the DIC image
        _report(image, 'grgb')
        save_input_layers(image)                             # {channel}.tif, DIC.tif and max projections
        image.time['01-configure'] = time.time() - tic
        image.time['processing_mode'] = mode
        image.record_settings('read')
        image.save_json('configure')
    return image.files.img_json


def save_input_layers(image):
    """Write every channel as an OME-TIFF z-stack (+ its maximum projection for napari) and the DIC image."""
    from buddy.reading.readers import save_channel_stacks
    save_channel_stacks(image.savepath, {ch: image.mrna[ch]["data"] for ch in image.mrna}, image.cells)


# =================================================================================================
# step 2: segment
# =================================================================================================
def segment_image(json_file, exp):
    """Step 2: cell masks (DIC + DAPI composite) and nucleus masks (DAPI) with Omnipose, then clean them up."""
    image = Image.from_json(json_file, exp)
    with _step(2, "segment", image):
        image.load('dapi')                                   # only the DAPI stack is needed here
        image.load('dic')
        _timed(image, '02-segment-cells', segment_cells)
        _timed(image, '02-segment-dapi', segment_nuclei)
        _timed(image, '02-segment-pp', clean_up_masks)       # -> DIC_masks_pp.tif, DAPI_masks.tif (not yet aligned)
        image.record_settings('segment')
        image.save_json('segment')
    return json_file


def _omnipose_models(exp):
    """The models loaded by exp.init_omnipose(), loaded here when the notebook / GUI did not do it."""
    if getattr(exp, "cell_model", None) is None:
        exp.init_omnipose()
    return exp.cell_model, exp.nuclei_model


def segment_cells(image):
    """Omnipose on the (empty, DIC, DAPI projection, empty) composite -> image.cell_masks, DIC_masks.tif (+ named copy)."""
    from buddy.paths import CELL_MASKS_RAW
    from buddy.segmentation import masks
    cell_model, _ = _omnipose_models(image.experiment)
    params, _ = masks.omnipose_settings(image.experiment)
    composite = masks.composite(image.DIC, image.max_projection("DAPI"))
    image.cell_masks = masks.run_omnipose(composite, cell_model, params, "cells")
    _segmentation_settings(image)["cells"] = {**params, "model_type": image.experiment.segmentation["cell_model"]}
    masks.save_masks(Path(image.savepath), image.cell_masks, CELL_MASKS_RAW, "DIC_masks.",
                     image.experiment.segmentation["cell_model"], params)


def segment_nuclei(image):
    """Omnipose on the DAPI maximum projection -> image.dapi_masks, DAPI_masks.tif (+ named copy)."""
    from buddy.paths import NUCLEUS_MASKS
    from buddy.segmentation import masks
    _, nucleus_model = _omnipose_models(image.experiment)
    _, params = masks.omnipose_settings(image.experiment)
    image.dapi_masks = masks.run_omnipose(image.max_projection("DAPI"), nucleus_model, params, "nuclei")
    _segmentation_settings(image)["dapi"] = {**params, "model_type": image.experiment.segmentation["nucleus_model"]}
    masks.save_masks(Path(image.savepath), image.dapi_masks, NUCLEUS_MASKS, "DAPI_masks_",
                     image.experiment.segmentation["nucleus_model"], params)


def clean_up_masks(image):
    """Drop cells without a nucleus / too small / at the edge, grow cells, drop nuclei without a cell -> DIC_masks_pp.tif, DAPI_masks.tif."""
    from buddy.segmentation import cleanup
    cells, image.dapi_masks, settings = cleanup.clean_up_masks(image.cell_masks, image.dapi_masks, image.experiment.parameters)
    image.cell_masks_by_area = cells
    _segmentation_settings(image)["postprocess"] = settings
    io.imsave(image.files.nucleus_masks, image.dapi_masks, check_contrast=False)
    io.imsave(image.files.cell_masks, cells, check_contrast=False)


def _segmentation_settings(image):
    if not hasattr(image, "segmentation"):
        image.segmentation = {"cells": {}, "dapi": {}}
    return image.segmentation


# =================================================================================================
# step 3a: spots
# =================================================================================================
def spots_image(json_file, exp, mode='serial'):
    """Step 3a: in-focus planes, background filter, spot detection + decomposition, align the masks to the spots."""
    image = Image.from_json(json_file, exp)
    with _step(3, "spots", image):
        _report(image, 'load')
        image.load('masks')
        image.load('mrna')

        tic = time.time()
        _report(image, 'focus')
        find_focus(image)
        _report(image, 'filter')
        filter_background(image)
        _report(image, 'detect')
        detect_spots(image)
        image.time['03-detect-spots'] = time.time() - tic
        _timed(image, '04-decompose-spots', decompose_spots)
        _timed(image, 'align', align_masks, True)            # -> aligned_DIC(_masks).tif, corrected/ masks

        f = ImageFiles(image.savepath)
        f.corrected.mkdir(parents=True, exist_ok=True)
        if not f.corrected_nuclei.is_file() and f.nucleus_masks.is_file():
            shutil.copy2(f.nucleus_masks, f.corrected_nuclei)

        image.time['processing_mode'] = mode
        image.record_settings('spots')
        image.save_json('spots')
    return json_file


def spot_device(image):
    """'cuda' for the GPU versions of filter / detection / decomposition (spots/gpu_detection.py), else None (Big-FISH, CPU)."""
    if not hasattr(image, "_spot_device"):
        from buddy.experiment import gpu_available
        wanted = image.parameters.get("gpu_spot_detection", True) and getattr(image.experiment, "use_gpu", True)
        image._spot_device = "cuda" if wanted and gpu_available(True) else None
    return image._spot_device


def find_focus(image):
    """The in-focus z range per channel, measured on the patch with the most cells -> img.json (ifx_1, ifx_2, z_max_focus)."""
    from buddy.spots import detection
    log.info("finding focus")
    patch_size = image.parameters["patch_size"]
    image.selected_patch = detection.find_high_density_patch(image.cell_masks_uncorrected, patch_size=patch_size)
    for ch in image.fluorescence_channels:
        image.results[ch] = {"colormap": image.experiment.channels[ch]["colormap"]}
        image.results[ch].update(detection.find_focus(image.mrna[ch]["data"], image.selected_patch, patch_size))
        r = image.results[ch]
        log.info(f"..channel {ch}: [{r['ifx_1']}, {r['ifx_2']}], max focus slice {r['z_max_focus']}")


def filter_background(image):
    """Gaussian background removal of every fluorescence stack (Big-FISH, or spots/gpu_detection.py on the GPU) -> mrna[ch]['filtered']."""
    from buddy.spots import detection
    log.info("filtering: remove_background_gaussian")
    for ch in image.fluorescence_channels:
        r = image.results[ch]
        image.mrna[ch]["filtered"], r["filter_backend"] = detection.filter_background(
            image.mrna[ch]["data"], image.parameters["sigma"], spot_device(image), f"channel {ch} filter")
        log.info(f"..channel {ch} ({r['filter_backend']})")


def detect_spots(image):
    """LoG + local maxima + threshold on the whole stack, kept within the in-focus range -> {channel}_spots.npy/.csv, _filtered.tif."""
    from buddy.reading.readers import save_projection
    from buddy.spots import detection
    log.info("detecting spots")
    f = image.files
    for ch in image.fluorescence_channels:
        r, data = image.results[ch], image.mrna[ch]
        found, r["threshold"], r["detection_backend"] = detection.detect_spots(
            data["data"], data["filtered"], r["ifx_1"], r["ifx_2"], image.experiment.channels[ch]["threshold"],
            image.parameters["scale"], image.parameters["spot_radius"], spot_device(image), f"channel {ch} detection")
        selected = data["filtered"][r["ifx_1"]:r["ifx_2"]]
        saved = np.zeros_like(data["filtered"])                        # slices outside the focus range are 0
        saved[r["ifx_1"]:r["ifx_2"]] = selected
        io.imsave(f.filtered(ch), saved.astype("uint16"), check_contrast=False)
        save_projection(f.filtered(ch), selected)                      # for napari
        np.save(f.spots(ch), found)
        pd.DataFrame(found, columns=["z", "y", "x", "intensity", "filtered_intensity"]).to_csv(f.spots_csv(ch))
        data["spots"] = found
        r["nr_spots"] = len(found)
        log.info(f"..channel {ch}: found {len(found)} spots, threshold={r['threshold']} ({r['detection_backend']})")


def decompose_spots(image):
    """Big-FISH decomposition of dense regions into single spots -> {channel}_decomposed_spots.npy, _dense_regions.npy, _reference_spot.tif."""
    from buddy.spots import detection
    log.info("decomposing spots")
    f = image.files
    for ch in image.fluorescence_channels:
        data = image.mrna[ch]
        decomposed, dense, reference, image.results[ch]["decomposition_backend"] = detection.decompose_spots(
            data["data"], data["spots"], image.parameters["scale"], image.parameters["spot_radius"], spot_device(image),
            f"channel {ch} decomposition", decomposition=image.parameters.get("decomposition"))
        data["decomposed_spots"], data["dense_regions"], data["reference_spot"] = decomposed, dense, reference
        np.save(f.decomposed_spots(ch), decomposed)
        np.save(f.dense_regions(ch), dense)
        io.imsave(f.reference_spot(ch), reference, check_contrast=False)
        log.info(f"..channel {ch}: found {len(dense)} dense regions, {len(decomposed)} decomposed spots (from {len(data['spots'])})")


def align_masks(image, overwrite_corrected=False, n_iter=4000, max_shift=200, radius=20, cutoff=50):
    """Shift the DIC frame (DIC + Omnipose masks) onto the spots (buddy/spots/alignment.py) -> aligned_*.tif, corrected/ masks, img.json (alignment)."""
    from buddy.spots import alignment
    log.info("aligning DIC / cell masks to detected spots")
    found = []
    for ch in image.fluorescence_channels:
        if "spots" in image.mrna.get(ch, {}):
            found.append(np.asarray(image.mrna[ch]["spots"])[:, :3])
        elif image.files.spots(ch).is_file():
            found.append(np.load(image.files.spots(ch))[:, :3])
    result = alignment.align_image(image.files, np.concatenate(found) if found else np.zeros((0, 3)),
                                   previous=image.results.get("alignment"), overwrite_corrected=overwrite_corrected,
                                   n_iter=n_iter, max_shift=max_shift, radius=radius, cutoff=cutoff)
    if result is not None:
        image.results["alignment"], image.cell_masks_aligned, image.aligned_DIC = result


# =================================================================================================
# step 3b: bud model and counting   /   step 5: counting again with the corrected masks
# =================================================================================================
def _load_for_counting(json_file, exp):
    image = Image.from_json(json_file, exp)
    image.load('mrna')       # channel data incl. DAPI (bud model, nuclear z-projection)
    image.load('spots')      # detected spots
    image.load('dense')      # decomposed dense regions
    return image


def buds_and_counts(json_file, exp):
    """Step 3b: bud model (if enabled in the config) -> bud masks, then count spots per cell, mother and bud."""
    image = _load_for_counting(json_file, exp)
    with _step(3, "buds and counts", image):
        if exp.bud_classification['enabled']:
            _timed(image, 'bud-classification', classify_buds)
        _timed(image, '05-assign-spots', count_spots)        # -> {channel}.csv
        flag_cells(image)                                    # -> cell_flags.csv (napari)
        assign_cell_cycle(image)                             # -> cell_cycle.csv (napari)
        image.record_settings('spots')
        image.save_json('spots')
    return json_file


def reassign_image(json_file, exp):
    """Step 5: count spots (and assign cell-cycle phases) again with the (hand-)corrected masks in corrected/."""
    image = _load_for_counting(json_file, exp)
    with _step(5, "recount", image):
        _timed(image, '05-assign-spots', count_spots)
        flag_cells(image)                                    # -> cell_flags.csv (napari)
        assign_cell_cycle(image)                             # -> cell_cycle.csv (napari)
        image.save_json('spots')
    return json_file


def classify_buds(image):
    """
    Automatic mother/bud assignment with the bud model (buddy/buds/bud_model.py), after the alignment and before
    counting. Writes corrected/DIC_corrected_masks.tif (buds merged into their mother), corrected/bud_corrected_masks.tif,
    bud_flags.csv, bud_pairs.csv and bud_model_cells.csv. Bud masks you corrected by hand are never overwritten
    (unless "overwrite_manual_corrections": true).
    """
    from scipy.ndimage import shift as ndi_shift

    from buddy.reading.readers import md5
    from buddy.spots import alignment
    exp, settings, f = image.experiment, image.experiment.bud_classification, image.files
    written_by_model = image.results.get("bud_classification", {}).get("bud_mask_md5")
    if f.corrected_buds.is_file() and md5(f.corrected_buds) != written_by_model and not settings["overwrite_manual_corrections"]:
        log.warning("..corrected/bud_corrected_masks.tif was made or edited by hand - not overwritten "
                    '(set "overwrite_manual_corrections": true to replace it)')
        return
    if getattr(exp, "bud_model", None) is None:
        exp.init_bud_model()
    model = exp.bud_model
    log.info(f"classifying buds (model: {model.name})")
    record = image.results.get("alignment")
    if record is None:
        log.warning("..no alignment found - run step 3 (spots) first; skipping bud classification")
        return
    h, w = record["target_shape"]
    dy, dx = record["mask_shift"]
    cells = ndi_shift(alignment.pad_to(io.imread(f.cell_masks), h, w), shift=(dy, dx), order=0, mode="constant", cval=0)
    dic = io.imread(f.aligned_dic) if f.aligned_dic.is_file() else \
        ndi_shift(alignment.pad_to(io.imread(f.dic), h, w), shift=(dy, dx), order=0)
    dapi = image.max_projection("DAPI")
    ash = image.max_projection(settings["ash_channel"]) if model.uses_ash else None
    result = model.classify(dic, dapi, cells, ash)
    f.corrected.mkdir(parents=True, exist_ok=True)
    io.imsave(f.corrected_cells, result["cell_masks"].astype(np.uint16), check_contrast=False)
    io.imsave(f.corrected_buds, result["bud_masks"], check_contrast=False)
    result["flags"].to_csv(f.bud_flags, index=False)
    result["pairs"].to_csv(f.bud_pairs, index=False)
    result["cells"].to_csv(f.bud_model_cells, index=False)
    image.results["bud_classification"] = {
        "model": model.name, "cut_off": model.cut_off, "snap": model.snap, "device": model.device,
        "dic_rotation_k": int(result["dic_rotation"]), "dic_rotation_fixed": model.dic_rotation is not None,
        "dic_light": result.get("dic_light"),          # measured lighting direction (angle, consistency 0-1, cells)
        "n_cells": int(len(result["cells"])),
        "n_pairs": int(len(result["pairs"])), "n_flags": int(len(result["flags"])),
        "bud_mask_md5": md5(f.corrected_buds),          # lets a later run recognise hand-made edits
    }
    log.info(f"..{len(result['pairs'])} mother-bud pairs, {len(result['flags'])} cells flagged for checking")


def count_spots(image):
    """Spots and dense regions per cell, mother and bud, with the corrected masks (buddy/quantification/spot_assignment.py) -> {channel}.csv."""
    from buddy.buds.bud_tools import BUD
    from buddy.quantification import spot_assignment
    log.info("assign spots")
    f = image.files
    image.cell_masks = image._read("corrected_cells")                # re-read: may have been corrected by hand
    image.dapi_masks = image._read("corrected_nuclei")
    image.bud_masks = image._read("corrected_buds", optional=True)   # step 3b or 4 may not have made buds yet
    if image.bud_masks is not None:
        bud_labels = np.where(image.bud_masks == BUD, image.cell_masks, 0)
    else:
        bud_labels = None
        log.info("..no corrected/bud_corrected_masks.tif: counting without bud compartment")
    nucleus_labels = np.where(image.dapi_masks > 0, image.cell_masks, 0)
    dapi_projection = image.max_projection("DAPI")
    for ch in image.fluorescence_channels:
        table = spot_assignment.spot_assignment(image.cell_masks, bud_labels, nucleus_labels, dapi_projection,
                                                image.mrna[ch]["spots"][:, :3], image.mrna[ch]["dense_regions"])
        table.rename(columns={"label": "image_cell_id"}, inplace=True)
        table.to_csv(f.counts(ch), index=False)
        log.info(f"..channel {ch}: saving results to {f.counts(ch)}")


def flag_cells(image):
    """Flags for napari: unbudded cells with high ASH1/CLB2 (buddy/quantification/cell_flags.py) -> cell_flags.csv."""
    from buddy.quantification.cell_flags import compute_flags
    settings, f = dict(image.experiment.cell_flags), image.files
    if not settings["enabled"]:
        return
    ch = settings["ash_channel"] or image.experiment.bud_classification["ash_channel"]
    if not f.counts(ch).is_file() or not f.corrected_cells.is_file():
        log.info(f"..no {ch} channel counts - no cell flags")
        if f.cell_flags.is_file():
            f.cell_flags.unlink()
        return
    flags = compute_flags(pd.read_csv(f.counts(ch)), io.imread(f.corrected_cells), settings)
    flags.to_csv(f.cell_flags, index=False)
    image.results["cell_flags"] = {"ash_channel": ch, "n_flags": int(len(flags)),
                                   **{k: settings[k] for k in ("fold", "nucleus_area", "nucleus_intensity", "max_gap_px")}}
    log.info(f"..{len(flags)} cells flagged for high {ch} (cell_flags.csv)")


def assign_cell_cycle(image):
    """Cell-cycle phase per cell (buddy/quantification/cell_cycle.py) -> cell_cycle.csv (napari 'cell cycle' layer)."""
    from buddy.quantification.cell_cycle import assign_phases, phase_counts
    settings, f = dict(image.experiment.cell_cycle), image.files
    channels = [ch for ch in image.fluorescence_channels if f.counts(ch).is_file()]
    if not settings["enabled"] or not channels:
        if f.cell_cycle.is_file():
            f.cell_cycle.unlink()
        return
    ash_channel = settings["ash_channel"] or image.experiment.bud_classification["ash_channel"]
    ash = None
    if f.counts(ash_channel).is_file():
        ash = pd.read_csv(f.counts(ash_channel)).set_index("image_cell_id")["total_RNAs"]
    else:
        log.info(f"..no {ash_channel} counts: unbudded cells are called 'unbudded', not G1")
    table = pd.read_csv(f.counts(ash_channel if ash is not None else channels[0]))
    pixel_um = float(image.experiment.parameters["scale"][1]) / 1000
    phases = assign_phases(table, settings, pixel_um, ash)
    phases.to_csv(f.cell_cycle, index=False)
    counts = phase_counts(phases)
    image.results["cell_cycle"] = {"ash_channel": ash_channel if ash is not None else None, "counts": counts,
                                   **{k: settings[k] for k in ("g1_max_ash", "s_g2_bud_um2", "max_bud_um2",
                                                               "min_length_um", "max_length_um")}}
    log.info("..cell-cycle phases (cell_cycle.csv): " + ", ".join(f"{k} {v}" for k, v in counts.items()))


# =================================================================================================
# where is what
# =================================================================================================
def where(step=None):
    """
    Print, per step (or for one step: "segment", 2, "spots", ...), what buddy does, in which function here, which
    module does the work, which block of the config file it reads and which files it writes.
    """
    rows = {k: v for k, v in CODE_MAP.items() if step is None or str(step).lower() in k.lower()}
    if not rows:
        print(f"no step matches {step!r}; steps: {list(CODE_MAP)}")
        return
    for name, subs in rows.items():
        print(f"step {name}")
        for what, func, module, config, files in subs:
            print(f"  {what:<48} pipeline.{func:<40}")
            print(f"  {'':<48} does the work: {module}")
            print(f"  {'':<48} config: {config};  writes: {files}")
