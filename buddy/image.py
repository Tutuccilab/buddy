"""
One image (field of view) of an Experiment: its files, its arrays in memory and its img.json. Nothing is computed
here - the processing steps are the functions of pipeline.py, which read and write through this class.

    image = Image.from_json(folder / "img.json", exp)
    image.load("masks"); image.load("mrna")          # arrays of earlier steps (LAYERS says which files, made by which step)
    image.load("spots", optional=True)                 # None when not there yet, instead of an error
    image.save_json("spots")                           # img.json: settings, results, timings, last step

Attributes: experiment, parameters, vsi_file, cell_file, stem, savepath (the image folder), files (ImageFiles),
mrna ({channel: {"data": z-stack, "filtered", "spots", "dense_regions", ...}}), cells / DIC (2D), the masks
(cell_masks, dapi_masks, bud_masks, ...), results (per channel + alignment + settings, saved in img.json),
time (seconds per step), segmentation (Omnipose settings used), last_run.
"""
import logging
from pathlib import Path

import numpy as np
from skimage import io

from buddy.paths import ImageFiles
from buddy.reading import readers

log = logging.getLogger("buddy")


class MissingFileError(FileNotFoundError):
    """A file an earlier step should have written is not there; the message says which step makes it."""


# what load() reads: layer -> list of (attribute, ImageFiles attribute or callable(files, channel), made by)
STEP_OF = {
    "channel_stack": "step 1 (read)", "dic": "step 1 (read)",
    "cell_masks_raw": "step 2 (segment)", "cell_masks": "step 2 (segment)", "nucleus_masks": "step 2 (segment)",
    "spots": "step 3 (spots)", "dense_regions": "step 3 (spots)", "aligned_dic": "step 3 (spots, alignment)",
    "aligned_cell_masks": "step 3 (spots, alignment)", "corrected_cells": "step 3 (spots, alignment) or step 4 (napari)",
    "corrected_nuclei": "step 3 (spots) or step 4 (napari)", "corrected_buds": "step 3b (bud model) or step 4 (napari)",
}
LAYERS = {
    "dic": [("DIC", "dic")],
    "dapi": [("mrna.DAPI.data", ("channel_stack", "DAPI"))],
    "mrna": "all channel stacks",                                   # mrna[ch]["data"] for every channel of the config
    "masks": [("cell_masks_uncorrected", "cell_masks"), ("dapi_masks_uncorrected", "nucleus_masks"),
              ("cell_masks", "corrected_cells"), ("dapi_masks", "corrected_nuclei"),
              ("bud_masks", "corrected_buds"), ("cell_masks_aligned", "aligned_cell_masks")],
    "aligned_dic": [("aligned_DIC", "aligned_dic")],
    "spots": "per fluorescence channel: mrna[ch]['spots']",
    "dense": "per fluorescence channel: mrna[ch]['dense_regions']",
}
# in "masks", the corrected and aligned files may not exist yet (steps 3 / 4 make them): None then, no error
MASKS_OPTIONAL = {"corrected_cells", "corrected_nuclei", "corrected_buds", "aligned_cell_masks"}


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
        self.mrna = {}
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
        """
        The image of an img.json. The folder the file is in is the image folder, whatever "savepath" inside says:
        an output folder can be copied or moved (another drive, another computer) and still works; the next
        save_json() writes the new location.
        """
        img = readers.read_json(filename)
        folder = Path(filename).resolve().parent
        stored = img.get("savepath")
        if stored is not None and Path(stored).resolve() != folder:
            log.info(f"image folder moved: img.json says {stored}, using {folder}")
            img["savepath"] = str(folder)
        return cls(exp, img)

    @classmethod
    def from_dict(cls, params, exp):
        return cls(exp, params)

    @property
    def files(self):
        return ImageFiles(self.savepath)

    @property
    def fluorescence_channels(self):
        return [ch for ch in self.mrna if ch != "DAPI"]

    # ---- the microscope files (step 1) ---------------------------------------------------------------
    def read_image(self):
        """The fluorescence channels (+ DAPI): from the .vsi file, or from a multi-page TIFF (buddy/reading/tiff_input.py)."""
        channels, self._dic_from_stack = readers.read_channels(self.experiment, self.vsi_file)
        self.mrna = {mrna: {"data": data} for mrna, data in channels.items()}
        for mrna in self.fluorescence_channels:
            self.results[mrna] = {}

    def read_cells(self):
        """The DIC image: the _DIC.tif next to the .vsi, or the DIC channel of a TIFF stack."""
        if self.cell_file is None and getattr(self, "_dic_from_stack", None) is not None:
            log.info("cell image: the DIC channel of the TIFF stack")
            self.cells = self._dic_from_stack
        else:
            self.cells = readers.read_dic(Path(self.experiment.inputdir) / self.cell_file)
        self.DIC = self.cells

    # ---- files of earlier steps ----------------------------------------------------------------------
    def _read(self, key, channel=None, optional=False):
        """Read one file of the image folder by its ImageFiles name; MissingFileError (or None) when it is not there."""
        f = self.files
        path = getattr(f, key)(channel) if channel is not None else getattr(f, key)
        if not Path(path).is_file():
            if optional:
                log.info(f"..{Path(path).name} not there yet (made by {STEP_OF.get(key, '?')})")
                return None
            raise MissingFileError(f"{path} is missing: it is written by {STEP_OF.get(key, 'an earlier step')} - "
                              f"run that step for this image first")
        return np.load(path) if str(path).endswith(".npy") else io.imread(path)

    def load(self, layer, optional=False):
        """
        Read the arrays of earlier steps into this object (see LAYERS). A missing file raises MissingFileError naming
        the step that makes it; optional=True gives None instead (for files a step can do without).
        """
        if layer == "dic":
            self.DIC = self._read("dic", optional=optional)
        elif layer == "aligned_dic":
            self.aligned_DIC = self._read("aligned_dic", optional=optional)
        elif layer == "dapi":                          # only the DAPI stack (all that segmentation needs)
            self.mrna = {"DAPI": {"data": self._read("channel_stack", "DAPI", optional)}}
        elif layer == "mrna":
            self.mrna = {ch: {"data": self._read("channel_stack", ch, optional)} for ch in self.experiment.channels}
        elif layer == "masks":
            for attribute, key in LAYERS["masks"]:
                setattr(self, attribute, self._read(key, optional=optional or key in MASKS_OPTIONAL))
        elif layer == "spots":
            for ch in self.fluorescence_channels:
                self.mrna[ch]["spots"] = self._read("spots", ch, optional)
        elif layer == "dense":
            for ch in self.fluorescence_channels:
                self.mrna[ch]["dense_regions"] = self._read("dense_regions", ch, optional)
        else:
            raise ValueError(f"unknown layer {layer!r}; one of {list(LAYERS)}")

    def max_projection(self, channel):
        """2D maximum projection of a channel: from memory when the stack is loaded, else from its file."""
        if channel in self.mrna and "data" in self.mrna[channel]:
            return np.max(self.mrna[channel]["data"], axis=0)
        return np.max(self._read("channel_stack", channel), axis=0)

    # ---- img.json ------------------------------------------------------------------------------------
    def record_settings(self, step):
        """Record in img.json (results -> settings -> step) the config settings this step used (see which_images.py)."""
        from buddy.which_images import step_settings
        self.results.setdefault("settings", {})[step] = step_settings(self.experiment, step)
        self.parameters = dict(self.experiment.parameters)

    def save_json(self, stage):
        """Write img.json: file names, settings, results and timings of every step so far, and the last step run."""
        log.info(f"saving metadata to file: {self.files.img_json}")
        keys = ["vsi_file", "cell_file", "stem", "savepath", "parameters", "segmentation", "results", "time"]
        metadata = {k: getattr(self, k) for k in keys if hasattr(self, k)}
        metadata["last_run"] = stage
        readers.write_json(self.files.img_json, metadata)
        return metadata

