"""
Step 2, the Omnipose part: cell masks from the DIC + DAPI composite, nucleus masks from the DAPI projection.
The clean-up of the masks is in cleanup.py; the step that calls both is pipeline.segment_image.

    from buddy.segmentation import masks
    cell_params, nucleus_params = masks.omnipose_settings(exp)      # from the config: presets + "segmentation" block
    cell_model, nucleus_model = masks.load_models(exp)              # once per experiment (Experiment.init_omnipose)
    cells = masks.run_omnipose(masks.composite(dic, dapi_projection), cell_model, cell_params, "cells")
    nuclei = masks.run_omnipose(dapi_projection, nucleus_model, nucleus_params, "nuclei")

All Omnipose settings come from buddy/config.py (OMNIPOSE_PRESETS, SEGMENTATION_DEFAULTS); nothing is set here.
"""
import logging

import numpy as np
from skimage import io

from buddy.config import OMNIPOSE_NUCLEUS_PRESET, OMNIPOSE_PRESETS, omnipose_overrides
from buddy.log import keep_logging

log = logging.getLogger("buddy")


class GpuOutOfMemoryError(RuntimeError):
    """Omnipose ran out of GPU memory; the message says what to change in the config file."""


def omnipose_settings(exp):
    """
    The settings handed to Omnipose's model.eval(), for the cells and for the nuclei: the preset of the organism,
    "min_size" / "nuclei_min_size", the "cells" / "nuclei" overrides (ignored when "use_preset" is true) and "tile"
    (all in the "segmentation" block).
    """
    p, seg = exp.parameters, exp.segmentation
    cells_over, nuclei_over = omnipose_overrides(p, seg)
    cells = {**OMNIPOSE_PRESETS[p["organism"]], "min_size": p["min_size"], **cells_over, "tile": seg["tile"]}
    nuclei = {**OMNIPOSE_NUCLEUS_PRESET, "min_size": p["nuclei_min_size"], **nuclei_over, "tile": seg["tile"]}
    return cells, nuclei


def _stand_in_for_aicsimageio():
    """
    cellpose_omni/io.py does `from aicsimageio import AICSImage` at import time, but only uses it in its own imread()
    for formats buddy never reads with it. aicsimageio would bring ~25 packages (dask, xarray, ome-types, zarr, ...)
    for that one line, so Omnipose is installed without its dependency list (INSTALL.md) and this registers an empty
    stand-in module when aicsimageio is not installed. If it is installed, nothing happens.
    """
    import sys
    import types
    try:
        import aicsimageio  # noqa: F401
    except ImportError:
        module = types.ModuleType("aicsimageio")
        module.AICSImage = None     # cellpose_omni.io.imread would fail on it; buddy does not call imread
        sys.modules["aicsimageio"] = module


def check_omnipose():
    """
    Import Omnipose and make sure it is the real thing: `cellpose_omni` must be the copy inside the omnipose wheel
    and its Omnipose models (cyto2_omni, ...) available. The separate cellpose-omni package from PyPI, installed
    next to omnipose, overwrites that copy with an older one whose Omnipose check fails - cellpose_omni then runs
    as plain Cellpose ("cyto" model, 0 cells, a niter error on the nuclei). Returns the models module.
    """
    _stand_in_for_aicsimageio()
    repair = ("Repair: pip uninstall cellpose-omni, then pip install --force-reinstall --no-deps omnipose==1.1.4 "
              "(INSTALL.md -> Omnipose).")
    try:
        import omnipose.core  # noqa: F401 - a missing package surfaces here with its name
        from cellpose_omni import models
    except Exception as e:  # noqa: BLE001 - the PyPI cellpose_omni fails at import in several ways (downloads, ImportError)
        raise ImportError("Omnipose cannot be imported: it is not installed, one of its packages is missing (run "
                          "install.sh / install.ps1, or pip install --no-deps omnipose==1.1.4), or the cellpose-omni "
                          f"package from PyPI has replaced the copy that comes with omnipose. {repair} Cause: "
                          f"{e.__class__.__name__}: {e}") from e
    if not getattr(models, "OMNI_INSTALLED", False) or "cyto2_omni" not in getattr(models, "MODEL_NAMES", []):
        raise ImportError("cellpose_omni does not see Omnipose (OMNI_INSTALLED is false): the cellpose-omni package "
                          f"from PyPI has replaced the copy that comes with omnipose. {repair}")
    return models


def load_models(exp, use_gpu=None):
    """
    The two Omnipose models of the config ("segmentation": "cell_model", "nucleus_model"), loaded once so they are
    not re-loaded for every image. use_gpu: None = "use_gpu" of the config; falls back to the CPU without a GPU.
    Returns (cell_model, nucleus_model, on_gpu).
    """
    from buddy.experiment import gpu_available
    on_gpu = gpu_available(exp.use_gpu if use_gpu is None else use_gpu)
    seg = exp.segmentation
    with keep_logging():
        models = check_omnipose()           # torch import deferred to here; errors say what to install or repair
        log.info(f'loading Omnipose models (cells: {seg["cell_model"]}, nuclei: {seg["nucleus_model"]}) on {"GPU" if on_gpu else "CPU"}')
        cell_model = models.CellposeModel(gpu=on_gpu, model_type=seg["cell_model"])
        nucleus_model = models.CellposeModel(gpu=on_gpu, model_type=seg["nucleus_model"])
    return cell_model, nucleus_model, on_gpu


def composite(dic, dapi_projection):
    """The 4-channel composite the cell model segments: (empty, DIC, DAPI projection, empty)."""
    return np.stack([np.zeros(dic.shape), dic, dapi_projection, np.zeros(dic.shape)], axis=0)


def run_omnipose(image, model, params, what):
    """
    model.eval(image, **params) -> label mask (0 = background). what: "cells" or "nuclei", for the log and errors.
    A GPU out-of-memory error becomes a GpuOutOfMemoryError that names the config setting to change.
    """
    log.info(f"segmenting {what} (Omnipose {model.__class__.__name__}, tile={params.get('tile')})")
    try:
        masks, _flows, _styles = model.eval(image, **params)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            raise GpuOutOfMemoryError(
                f"Omnipose ran out of GPU memory while segmenting the {what}. Set \"tile\": true in the "
                '"segmentation" block of the config file (runs the network on tiles), or "use_gpu": false '
                '("hardware") to segment on the CPU.') from e
        raise
    log.info(f"..found {int(np.max(masks))} {what}")
    return masks


def mask_file_name(prefix, model_type, params):
    """DIC_masks.model=cyto2_omni_chan=[1,2]_diameter=50_...tif: the settings in the name (kept next to the plain file)."""
    return (f"{prefix}model={model_type}_chan={str(params['channels']).replace(' ', '')}_diameter={params['diameter']}"
            f"_minsize={params['min_size']}_mask={params['mask_threshold']}_flow={params['flow_threshold']}.tif")


def save_masks(folder, masks, plain_name, prefix, model_type, params):
    """Write the masks twice: under the plain name (what the pipeline reads) and with the settings in the name."""
    io.imsave(folder / plain_name, masks, check_contrast=False)
    named = folder / mask_file_name(prefix, model_type, params)
    io.imsave(named, masks, check_contrast=False)
    log.info(f"..written {folder / plain_name} (and {named.name})")
