"""
An Experiment = the images in one input directory that share one config file (processing parameters, channels).

    exp = Experiment.from_cfg_file("config.json")
    exp.create_image_list()            # the .vsi files / TIFF stacks in inputdir -> exp.images
    exp.read_image_list_from_jsons()   # the processed images in outputdir       -> exp.json_files

Settings are attributes: exp.parameters, exp.channels, exp.bud_classification, exp.cell_flags, exp.cell_cycle (dicts with every
default filled in, see buddy/config.py), exp.inputdir / exp.outputdir, exp.if_already_processed.
"""
import logging
import re
from pathlib import Path

from buddy.config import BUD_CLASSIFICATION_DEFAULTS, CELL_CYCLE_DEFAULTS, CELL_FLAG_DEFAULTS, load_config, validate  # noqa: F401 (re-exported)
from buddy.log import keep_logging, setup_logging
from buddy.reading.tiff_input import find_dic_file, is_stack

log = logging.getLogger("buddy")


def gpu_available(wanted=True):
    """True when a GPU is wanted and PyTorch can use one."""
    if not wanted:
        return False
    import torch
    if torch.cuda.is_available():
        return True
    log.warning("use_gpu is true, but no CUDA GPU is available to PyTorch - running on CPU "
                "(see INSTALL.md to install the GPU version of PyTorch)")
    return False


class Experiment:

    def __init__(self, cfg):
        cfg = validate(dict(cfg))
        for key, value in cfg.items():
            setattr(self, key, value)
        self.filter2mRNA = {v["filter"]: v["mrna"] for v in self.channels.values()}
        self.use_gpu = self.parameters["use_gpu"]
        self.images = {}
        self.json_files = []

    @classmethod
    def from_cfg_file(cls, cfg_file):
        setup_logging()
        try:
            cfg = load_config(cfg_file)
        except FileNotFoundError:
            log.warning(f"failed to find cfg file {cfg_file}")
            return None
        log.info(f"found cfg file {cfg_file}")
        return cls(cfg)

    # ---- models shared by all images -------------------------------------------------------------
    def init_omnipose(self, use_GPU=None):
        """
        Load the Omnipose models once (cells: cyto2_omni, nuclei), so they are not re-loaded for every image.
        use_GPU: None = follow "use_gpu" in the config file; falls back to the CPU without a GPU.
        """
        with keep_logging():
            from cellpose_omni import models  # torch import deferred to here
            self.use_GPU = gpu_available(self.use_gpu if use_GPU is None else use_GPU)
            log.info(f'initialising omnipose models (cell: cyto2_omni, nuclei) on {"GPU" if self.use_GPU else "CPU"}')
            self.cell_model = models.CellposeModel(gpu=self.use_GPU, model_type="cyto2_omni")
            self.nuclei_model = models.CellposeModel(gpu=self.use_GPU, model_type="nuclei")

    def init_bud_model(self):
        """Load the bud model chosen in the config once (None when bud classification is off)."""
        settings = self.bud_classification
        if not settings["enabled"]:
            log.info('automatic bud classification is off ("bud_classification" -> "enabled": false)')
            self.bud_model = None
            return
        from buddy.buds.bud_model import BudModel  # torch import deferred to here
        self.bud_model = BudModel.load(settings["model"], use_gpu=gpu_available(self.use_gpu),
                                       weights_dir=settings["weights_dir"], snap=settings["snap"])

    # ---- the images --------------------------------------------------------------------------------
    @staticmethod
    def stem_of(file_name):
        """Name of an image's output folder from its microscope file name (scene names and .ome stripped)."""
        stem = re.sub(r"_CY[A-Z0-9\s,._]+DAPI", "", Path(file_name).stem)
        return re.sub(r"\.ome$", "", stem)

    def create_image_list(self):
        """The images in inputdir: .vsi files (+ their _DIC.tif), and multi-page TIFF stacks (see buddy/tiff_input.py)."""
        self.images = {}
        self._add_tiff_stacks()
        log.info(f"getting .vsi files from {self.inputdir}")
        for f in Path(self.inputdir).glob("*.vsi"):
            log.info(f"examining image {f}")
            stem = self.stem_of(f.name)
            params = {"vsi_file": f.name, "cell_file": None, "valid": True}
            if (Path(self.inputdir) / f"_{f.stem}_").is_dir():
                log.info(f"..found VSI directory _{f.stem}_")
            else:
                log.warning(f"..failed to find VSI directory _{f.stem}_")
                params["valid"] = False
            dic_file = find_dic_file(self.inputdir, stem)
            if dic_file:
                params["cell_file"] = dic_file
                log.info(f"..found DIC file {dic_file}")
            else:
                log.warning(f"..failed to find a DIC file for {Path(self.inputdir) / stem}")
                params["valid"] = False
            if params["valid"]:
                log.info("..all good, adding image to image list")
                self.images[stem] = params

    def _add_tiff_stacks(self):
        for f in sorted(Path(self.inputdir).glob("*.tif*")):
            stem = self.stem_of(f.name)
            if stem.endswith("_DIC") or re.search(r"_DIC_\d\d$", stem):
                continue                                      # a DIC image, not a stack
            info = is_stack(f)
            if info is None:
                continue
            log.info(f'examining TIFF stack {f.name}: axes {info["axes"]}, shape {info["shape"]}, '
                     f'channels {info["channel_names"] or "unnamed"}')
            dic_file = find_dic_file(self.inputdir, stem)
            dic_name = str(self.parameters["dic_channel"])
            dic_in_stack = any(n.strip().upper() == dic_name.upper() for n in info["channel_names"]) or dic_name.isdigit()
            if dic_file is None and not dic_in_stack:
                log.warning(f"..no DIC image for {f.name} (expected {stem}_DIC.tif, or a channel named {dic_name!r}) - skipped")
                continue
            log.info(f'..DIC: {dic_file or "channel of the stack"}; adding image to image list')
            self.images[stem] = {"vsi_file": f.name, "cell_file": dic_file, "valid": True}

    def read_image_list_from_jsons(self):
        """The processed images: every <outputdir>/*/img.json."""
        self.json_files = sorted(Path(self.outputdir).glob("*/img.json"))
        for f in self.json_files:
            log.info(f"reading json file {f}")
        return self.json_files

