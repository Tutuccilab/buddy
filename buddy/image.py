"""
One image (field of view) of an Experiment and the steps that process it. The work is done by the modules
reading, segmentation, spots, quantification and buds; this class holds the image's data
and settings, calls them and writes the results. pipeline.py runs the steps in order.

    image = Image.from_json(folder / "img.json", exp)
    image.load_data("masks"); image.load_data("mrna")
    image.find_focus(); image.filter(); image.detect_spots(); ...

Attributes: experiment, parameters, vsi_file, cell_file, stem, savepath (the image folder), mrna
({channel: {"data": z-stack, "filtered", "spots", "dense_regions", ...}}), cells / DIC (2D), the masks
(cell_masks, dapi_masks, bud_masks, ...), results (per channel + alignment + settings, saved in img.json),
time (seconds per step), segmentation (Omnipose settings used), last_run.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import io

from buddy.buds.bud_tools import BUD
from buddy.paths import ImageFiles
from buddy.quantification import spot_assignment
from buddy.reading import readers
from buddy.segmentation import cleanup, omnipose
from buddy.spots import alignment, detection

log = logging.getLogger("buddy")


class Image:

    def __init__(self, exp, img):
        """
        exp : the Experiment (config); img : dict with at least vsi_file and cell_file (step 1) or the contents
        of an img.json (later steps: savepath, parameters, results, ...).
        """
        self.experiment = exp
        self.parameters = exp.parameters
        self.time = {}
        self.results = {}
        self.cell_file = None
        for key, value in img.items():
            setattr(self, key, value)
        if "savepath" not in img:
            self.stem = exp.stem_of(img["vsi_file"])
            self.savepath = str(Path(exp.outputdir) / self.stem)
            Path(self.savepath).mkdir(parents=True, exist_ok=True)
            log.info(f"created output dir {self.savepath}")
        elif Path(img["savepath"]).is_dir():
            log.info(f"found output dir {img['savepath']}")
        else:
            log.warning(f"could not find output dir {img['savepath']}")

    @classmethod
    def from_json(cls, filename, exp):
        return cls(exp, readers.read_json(filename))

    @classmethod
    def from_dict(cls, params, exp):
        return cls(exp, params)

    @property
    def files(self):
        return ImageFiles(self.savepath)

    @property
    def fluorescence_channels(self):
        return [ch for ch in self.mrna if ch != "DAPI"]

    # ---- step 1: read --------------------------------------------------------------------------
    def read_image(self):
        """The fluorescence channels: from the .vsi file, or from a multi-page TIFF (buddy/tiff_input.py)."""
        channels, self._dic_from_stack = readers.read_channels(self.experiment, self.vsi_file)
        self.mrna = {mrna: {"data": data} for mrna, data in channels.items()}
        for mrna in self.fluorescence_channels:
            self.results[mrna] = {}

    def read_cells(self):
        if self.cell_file is None and getattr(self, "_dic_from_stack", None) is not None:
            log.info("cell image: the DIC channel of the TIFF stack")
            self.cells = self._dic_from_stack
        else:
            self.cells = readers.read_dic(Path(self.experiment.inputdir) / self.cell_file)
        self.DIC = self.cells                     # create_grgb() / align() read self.DIC

    def create_grgb(self):
        log.info("creating GRGB composite image")
        self.grgb = omnipose.composite(self.DIC, np.max(self.mrna["DAPI"]["data"], axis=0))

    def save_input_layers(self):
        readers.save_channel_stacks(self.savepath, {ch: self.mrna[ch]["data"] for ch in self.mrna}, self.cells)

    # ---- loading what earlier steps wrote --------------------------------------------------------
    def load_data(self, layer):
        f = self.files
        read = lambda p: io.imread(p) if Path(p).is_file() else None
        if layer == "masks":
            self.cell_masks = read(f.corrected_cells)
            self.cell_masks_uncorrected = io.imread(f.cell_masks)
            self.bud_masks = read(f.corrected_buds)
            self.cell_masks_aligned = read(f.aligned_cell_masks)
            self.dapi_masks = read(f.corrected_nuclei)
            self.dapi_masks_uncorrected = io.imread(f.nucleus_masks)
        elif layer == "mrna":
            self.mrna = {ch: {"data": io.imread(f.channel_stack(ch))} for ch in self.experiment.channels}
        elif layer == "dapi":                     # only the DAPI stack (all that segmentation needs)
            self.mrna = {"DAPI": {"data": io.imread(f.channel_stack("DAPI"))}}
        elif layer == "spots":
            for ch in self.fluorescence_channels:
                self.mrna[ch]["spots"] = np.load(f.spots(ch))
        elif layer == "dense":
            for ch in self.fluorescence_channels:
                self.mrna[ch]["dense_regions"] = np.load(f.dense_regions(ch))
        elif layer == "DIC":
            self.DIC = io.imread(f.dic)
            self.aligned_DIC = io.imread(f.aligned_dic)
        else:
            raise ValueError(f"unknown layer {layer!r}")

    # ---- step 2: segmentation ----------------------------------------------------------------------
    def segment_cells(self):
        exp = self.experiment
        self.cell_masks, settings = omnipose.segment_cells(
            self.grgb, exp.parameters, model=getattr(exp, "cell_model", None), use_gpu=getattr(exp, "use_GPU", False),
            folder=Path(self.savepath))
        self._segmentation_settings()["cells"] = settings

    def segment_dapi(self):
        exp = self.experiment
        self.dapi_masks, settings = omnipose.segment_nuclei(
            np.max(self.mrna["DAPI"]["data"], axis=0), exp.parameters, model=getattr(exp, "nuclei_model", None),
            use_gpu=getattr(exp, "use_GPU", False), folder=Path(self.savepath))
        self._segmentation_settings()["dapi"] = settings

    def postprocess_masks(self):
        cells, self.dapi_masks, settings = cleanup.clean_up_masks(self.cell_masks, self.dapi_masks, self.experiment.parameters)
        self.cell_masks_by_area = cells
        self._segmentation_settings()["postprocess"] = settings
        io.imsave(self.files.nucleus_masks, self.dapi_masks, check_contrast=False)
        io.imsave(self.files.cell_masks, cells, check_contrast=False)

    def _segmentation_settings(self):
        if not hasattr(self, "segmentation"):
            self.segmentation = {"cells": {}, "dapi": {}}
        return self.segmentation

    # ---- step 3a: spots ---------------------------------------------------------------------------
    def _device(self):
        """'cuda' for the GPU versions of the filter / detection / decomposition, else None."""
        if not hasattr(self, "_filter_device"):
            from buddy.experiment import gpu_available
            wanted = self.parameters.get("gpu_spot_detection", True) and getattr(self.experiment, "use_gpu", True)
            self._filter_device = "cuda" if wanted and gpu_available(True) else None
        return self._filter_device

    def find_focus(self):
        log.info("finding focus")
        patch_size = self.parameters["patch_size"]
        self.selected_patch = detection.find_high_density_patch(self.cell_masks_uncorrected, patch_size=patch_size)
        for ch in self.fluorescence_channels:
            self.results[ch] = {"colormap": self.experiment.channels[ch]["colormap"]}
            self.results[ch].update(detection.find_focus(self.mrna[ch]["data"], self.selected_patch, patch_size))
            r = self.results[ch]
            log.info(f"..channel {ch}: [{r['ifx_1']}, {r['ifx_2']}], max focus slice {r['z_max_focus']}")

    def filter(self):
        log.info("filtering: remove_background_gaussian")
        for ch in self.fluorescence_channels:
            r = self.results[ch]
            self.mrna[ch]["filtered"], r["filter_backend"] = detection.filter_background(
                self.mrna[ch]["data"], self.parameters["sigma"], self._device(), f"channel {ch} filter")
            log.info(f"..channel {ch} ({r['filter_backend']})")

    def detect_spots(self):
        log.info("detecting spots")
        f = self.files
        for ch in self.fluorescence_channels:
            r, data = self.results[ch], self.mrna[ch]
            found, r["threshold"], r["detection_backend"] = detection.detect_spots(
                data["data"], data["filtered"], r["ifx_1"], r["ifx_2"], self.experiment.channels[ch]["threshold"],
                self.parameters["scale"], self.parameters["spot_radius"], self._device(), f"channel {ch} detection")
            selected = data["filtered"][r["ifx_1"]:r["ifx_2"]]
            saved = np.zeros_like(data["filtered"])                        # slices outside the focus range are 0
            saved[r["ifx_1"]:r["ifx_2"]] = selected
            io.imsave(f.filtered(ch), saved.astype("uint16"), check_contrast=False)
            readers.save_projection(f.filtered(ch), selected)              # for napari
            np.save(f.spots(ch), found)
            pd.DataFrame(found, columns=["z", "y", "x", "intensity", "filtered_intensity"]).to_csv(f.spots_csv(ch))
            data["spots"] = found
            r["nr_spots"] = len(found)
            log.info(f"..channel {ch}: found {len(found)} spots, threshold={r['threshold']} ({r['detection_backend']})")

    def decompose_spots(self):
        log.info("decomposing spots")
        f = self.files
        for ch in self.fluorescence_channels:
            data = self.mrna[ch]
            decomposed, dense, reference, self.results[ch]["decomposition_backend"] = detection.decompose_spots(
                data["data"], data["spots"], self.parameters["scale"], self.parameters["spot_radius"], self._device(),
                f"channel {ch} decomposition")
            data["decomposed_spots"], data["dense_regions"], data["reference_spot"] = decomposed, dense, reference
            np.save(f.decomposed_spots(ch), decomposed)
            np.save(f.dense_regions(ch), dense)
            io.imsave(f.reference_spot(ch), reference, check_contrast=False)
            log.info(f"..channel {ch}: found {len(dense)} dense regions, {len(decomposed)} decomposed spots (from {len(data['spots'])})")

    def align(self, overwrite_corrected=False, n_iter=4000, max_shift=200, radius=20, cutoff=50):
        """Align the DIC frame (DIC + Omnipose masks) to the spots; see buddy.spots.alignment.align_image."""
        log.info("aligning DIC / cell masks to detected spots")
        found = []
        for ch in self.fluorescence_channels:
            if "spots" in self.mrna.get(ch, {}):
                found.append(np.asarray(self.mrna[ch]["spots"])[:, :3])
            elif self.files.spots(ch).is_file():
                found.append(np.load(self.files.spots(ch))[:, :3])
        result = alignment.align_image(self.files, np.concatenate(found) if found else np.zeros((0, 3)),
                                       previous=self.results.get("alignment"), overwrite_corrected=overwrite_corrected,
                                       n_iter=n_iter, max_shift=max_shift, radius=radius, cutoff=cutoff)
        if result is not None:
            self.results["alignment"], self.cell_masks_aligned, self.aligned_DIC = result

    # ---- step 3b: bud model, counting, flags -------------------------------------------------------
    def classify_buds(self):
        """
        Automatic mother/bud assignment with the bud model (buddy/bud_model.py), after align() and before
        assign_spots(). Writes corrected/DIC_corrected_masks.tif (buds merged into their mother), corrected/
        bud_corrected_masks.tif, bud_flags.csv, bud_pairs.csv and bud_model_cells.csv. Bud masks you corrected by
        hand are never overwritten (unless "overwrite_manual_corrections": true).
        """
        from scipy.ndimage import shift as ndi_shift
        settings, f = self.experiment.bud_classification, self.files
        written_by_model = self.results.get("bud_classification", {}).get("bud_mask_md5")
        if f.corrected_buds.is_file() and readers.md5(f.corrected_buds) != written_by_model and not settings["overwrite_manual_corrections"]:
            log.warning("..corrected/bud_corrected_masks.tif was made or edited by hand - not overwritten "
                        '(set "overwrite_manual_corrections": true to replace it)')
            return
        model = getattr(self.experiment, "bud_model", None)
        if model is None:
            self.experiment.init_bud_model()
            model = self.experiment.bud_model
        log.info(f"classifying buds (model: {model.name})")
        record = self.results.get("alignment")
        if record is None:
            log.warning("..no alignment found - run align() first; skipping bud classification")
            return
        h, w = record["target_shape"]
        dy, dx = record["mask_shift"]
        cells = ndi_shift(alignment.pad_to(io.imread(f.cell_masks), h, w), shift=(dy, dx), order=0, mode="constant", cval=0)
        dic = io.imread(f.aligned_dic) if f.aligned_dic.is_file() else \
            ndi_shift(alignment.pad_to(io.imread(f.dic), h, w), shift=(dy, dx), order=0)
        dapi = self._max_projection("DAPI")
        ash = self._max_projection(settings["ash_channel"]) if model.uses_ash else None
        result = model.classify(dic, dapi, cells, ash)
        f.corrected.mkdir(parents=True, exist_ok=True)
        io.imsave(f.corrected_cells, result["cell_masks"].astype(np.uint16), check_contrast=False)
        io.imsave(f.corrected_buds, result["bud_masks"], check_contrast=False)
        result["flags"].to_csv(f.bud_flags, index=False)
        result["pairs"].to_csv(f.bud_pairs, index=False)
        result["cells"].to_csv(f.bud_model_cells, index=False)
        self.results["bud_classification"] = {
            "model": model.name, "cut_off": model.cut_off, "snap": model.snap, "device": model.device,
            "dic_rotation_k": int(result["dic_rotation"]), "dic_rotation_fixed": model.dic_rotation is not None,
            "dic_light": result.get("dic_light"),          # measured lighting direction (angle, consistency 0-1, cells)
            "n_cells": int(len(result["cells"])),
            "n_pairs": int(len(result["pairs"])), "n_flags": int(len(result["flags"])),
            "bud_mask_md5": readers.md5(f.corrected_buds),          # lets a later run recognise hand-made edits
        }
        log.info(f"..{len(result['pairs'])} mother-bud pairs, {len(result['flags'])} cells flagged for checking")

    def _max_projection(self, channel):
        if getattr(self, "mrna", None) and channel in self.mrna and "data" in self.mrna[channel]:
            return np.max(self.mrna[channel]["data"], axis=0)
        return np.max(io.imread(self.files.channel_stack(channel)), axis=0)

    def assign_spots(self):
        """Count spots per cell, mother and bud with the corrected masks -> <channel>.csv."""
        log.info("assign spots")
        f = self.files
        self.cell_masks = io.imread(f.corrected_cells)          # re-read: may have been corrected by hand
        self.dapi_masks = io.imread(f.corrected_nuclei)
        if f.corrected_buds.is_file():
            self.bud_masks = io.imread(f.corrected_buds)
            bud_labels = np.where(self.bud_masks == BUD, self.cell_masks, 0)
        else:
            self.bud_masks, bud_labels = None, None
            log.info("..no bud_corrected_masks.tif; assigning without bud compartment")
        nucleus_labels = np.where(self.dapi_masks > 0, self.cell_masks, 0)
        dapi_projection = np.max(self.mrna["DAPI"]["data"], axis=0)
        for ch in self.fluorescence_channels:
            table = spot_assignment.spot_assignment(self.cell_masks, bud_labels, nucleus_labels, dapi_projection,
                                             self.mrna[ch]["spots"][:, :3], self.mrna[ch]["dense_regions"])
            table.rename(columns={"label": "image_cell_id"}, inplace=True)
            table.to_csv(f.counts(ch), index=False)
            log.info(f"..channel {ch}: saving results to {f.counts(ch)}")

    def flag_cells(self):
        """Flags for napari: unbudded cells with high ASH1/CLB2 (buddy/cell_flags.py) -> cell_flags.csv."""
        from buddy.quantification.cell_flags import compute_flags
        settings, f = dict(self.experiment.cell_flags), self.files
        if not settings["enabled"]:
            return
        ch = settings["ash_channel"] or self.experiment.bud_classification["ash_channel"]
        if not f.counts(ch).is_file() or not f.corrected_cells.is_file():
            log.info(f"..no {ch} channel counts - no cell flags")
            if f.cell_flags.is_file():
                f.cell_flags.unlink()
            return
        flags = compute_flags(pd.read_csv(f.counts(ch)), io.imread(f.corrected_cells), settings)
        flags.to_csv(f.cell_flags, index=False)
        self.results["cell_flags"] = {"ash_channel": ch, "n_flags": int(len(flags)),
                                      **{k: settings[k] for k in ("fold", "nucleus_area", "nucleus_intensity", "max_gap_px")}}
        log.info(f"..{len(flags)} cells flagged for high {ch} (cell_flags.csv)")

    def assign_cell_cycle(self):
        """Cell-cycle phase per cell (buddy/quantification/cell_cycle.py) -> cell_cycle.csv (napari 'cell cycle' layer)."""
        from buddy.quantification.cell_cycle import assign_phases, phase_counts
        settings, f = dict(self.experiment.cell_cycle), self.files
        channels = [ch for ch in self.fluorescence_channels if f.counts(ch).is_file()]
        if not settings["enabled"] or not channels:
            if f.cell_cycle.is_file():
                f.cell_cycle.unlink()
            return
        ash_channel = settings["ash_channel"] or self.experiment.bud_classification["ash_channel"]
        ash = None
        if f.counts(ash_channel).is_file():
            ash = pd.read_csv(f.counts(ash_channel)).set_index("image_cell_id")["total_RNAs"]
        else:
            log.info(f"..no {ash_channel} counts: unbudded cells are called 'unbudded', not G1")
        table = pd.read_csv(f.counts(ash_channel if ash is not None else channels[0]))
        pixel_um = float(self.experiment.parameters["scale"][1]) / 1000
        phases = assign_phases(table, settings, pixel_um, ash)
        phases.to_csv(f.cell_cycle, index=False)
        counts = phase_counts(phases)
        self.results["cell_cycle"] = {"ash_channel": ash_channel if ash is not None else None, "counts": counts,
                                      **{k: settings[k] for k in ("g1_max_ash", "s_g2_bud_um2", "max_bud_um2",
                                                                  "min_length_um", "max_length_um")}}
        log.info("..cell-cycle phases (cell_cycle.csv): " + ", ".join(f"{k} {v}" for k, v in counts.items()))

    # ---- metadata --------------------------------------------------------------------------------
    def record_settings(self, step):
        """Record in img.json (results -> settings -> step) the config settings this step used (see rerun_check.py)."""
        from buddy.rerun_check import step_settings
        self.results.setdefault("settings", {})[step] = step_settings(self.experiment, step)
        self.parameters = dict(self.experiment.parameters)

    def save_metadata(self, stage):
        log.info(f"saving metadata to file: {self.files.img_json}")
        keys = ["vsi_file", "cell_file", "stem", "savepath", "parameters", "segmentation", "results", "time"]
        metadata = {k: getattr(self, k) for k in keys if hasattr(self, k)}
        metadata["last_run"] = stage
        readers.write_json(self.files.img_json, metadata)
        return metadata
