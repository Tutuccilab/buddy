"""
Step 2: cell masks (Omnipose on the DIC + DAPI composite) and nucleus masks (Omnipose on the DAPI projection).
The clean-up of the masks is in cleanup.py.

    from buddy.segmentation.omnipose import segment_cells, segment_nuclei
"""
import logging

import numpy as np
from skimage import io

from buddy.log import keep_logging

log = logging.getLogger("buddy")

CELL_MODEL = "cyto2_omni"
NUCLEUS_MODEL = "nuclei"

_OMNIPOSE_COMMON = {
    "rescale": None,            # upscale or downscale the image, None = no rescaling
    "invert": False,
    "transparency": True,       # transparency in flow output
    "omni": True,               # Omnipose mask reconstruction
    "cluster": True,            # DBSCAN clustering
    "resample": True,           # run dynamics on the rescaled grid or the original grid
    "verbose": False,
    "tile": False,              # average the outputs of flipped (augmented) images; slower, usually not needed
    "niter": None,              # None lets Omnipose choose the number of Euler iterations
    "augment": False,
    "affinity_seg": False,
}
# Omnipose settings for cells, per organism
CELL_PARAMS = {
    "S.cerevisiae": {**_OMNIPOSE_COMMON, "channels": [1, 2], "mask_threshold": -3, "flow_threshold": 1, "min_size": 10, "diameter": 50},
    "E.coli": {**_OMNIPOSE_COMMON, "channels": [1, 2], "mask_threshold": 0, "flow_threshold": 0, "min_size": 20, "diameter": 0},
}
NUCLEUS_PARAMS = {**_OMNIPOSE_COMMON, "channels": [0, 0], "mask_threshold": 0.0, "flow_threshold": 0.0, "min_size": 10, "diameter": 0.0}


def composite(dic, dapi_projection):
    """The 4-channel composite Omnipose segments the cells on: (empty, DIC, DAPI projection, empty)."""
    return np.stack([np.zeros(dic.shape), dic, dapi_projection, np.zeros(dic.shape)], axis=0)


def _mask_file_name(prefix, model_type, params):
    return (f"{prefix}model={model_type}_chan={str(params['channels']).replace(' ', '')}_diameter={params['diameter']}"
            f"_minsize={params['min_size']}_mask={params['mask_threshold']}_flow={params['flow_threshold']}.tif")


def segment_cells(grgb, parameters, model=None, use_gpu=False, folder=None):
    """
    Omnipose cell masks of the composite image. parameters: the config's "parameters" (organism, min_size).
    model: a loaded CellposeModel (Experiment.init_omnipose), else one is loaded here. Returns (masks, settings used).
    The masks are also written to <folder>/DIC_masks.model=... when a folder is given.
    """
    if model is None:
        with keep_logging():
            from cellpose_omni import models  # torch import deferred to here
            model = models.CellposeModel(gpu=use_gpu, model_type=CELL_MODEL)
    organism = parameters.get("organism", "S.cerevisiae")
    if organism not in CELL_PARAMS:
        raise ValueError(f'"organism" must be one of {list(CELL_PARAMS)}, not {organism!r}')
    params = dict(CELL_PARAMS[organism])
    params["min_size"] = parameters.get("min_size", params["min_size"])
    log.info("segmenting DIC image")
    masks, _flow, _style = model.eval(grgb, **params)
    log.info(f"Found {np.max(masks)} cells")
    settings = {**params, "model_type": CELL_MODEL}
    if folder is not None:
        from buddy.paths import CELL_MASKS_RAW
        f = folder / _mask_file_name("DIC_masks.", CELL_MODEL, params)
        io.imsave(f, masks, check_contrast=False)
        io.imsave(folder / CELL_MASKS_RAW, masks, check_contrast=False)              # the raw cells (also used to train the bud model)
        log.info(f"writing cell mask to {f}")
    return masks, settings


def segment_nuclei(dapi_projection, parameters, model=None, use_gpu=False, folder=None):
    """Omnipose nucleus masks of the DAPI maximum projection. Returns (masks, settings used); writes DAPI_masks.tif."""
    from buddy.paths import NUCLEUS_MASKS
    if model is None:
        with keep_logging():
            from cellpose_omni import models
            model = models.CellposeModel(gpu=use_gpu, model_type=NUCLEUS_MODEL)
    params = dict(NUCLEUS_PARAMS)
    params["min_size"] = parameters.get("nuclei_min_size", params["min_size"])
    log.info("segmenting DAPI image")
    log.info(f"nuclei model_type: {NUCLEUS_MODEL}, channels: {params['channels']}")
    masks, _flow, _style = model.eval(dapi_projection, **params)
    log.info(f"Found {np.max(masks)} cells")
    settings = {**params, "model_type": NUCLEUS_MODEL}
    if folder is not None:
        f = folder / _mask_file_name("DAPI_masks_", NUCLEUS_MODEL, params)
        io.imsave(f, masks, check_contrast=False)
        io.imsave(folder / NUCLEUS_MASKS, masks, check_contrast=False)
        log.info(f"writing DAPI mask to {f}")
    return masks, settings
