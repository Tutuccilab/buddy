"""The "segmentation" block of the config: Omnipose settings assembled from the presets, "parameters" and the overrides."""
import pytest

from buddy.config import OMNIPOSE_PRESETS, SEGMENTATION_DEFAULTS, validate
from buddy.experiment import Experiment
from buddy.segmentation import masks


def _cfg(**segmentation):
    return {"inputdir": "in", "outputdir": "out", "channels": {"DAPI": {"filter": "DAPI"}},
            "parameters": {"scale": [200, 65, 65], "spot_radius": [880, 120, 120], "sigma": [0.75, 2.3, 2.3],
                           "min_size": 12, "nuclei_min_size": 8},
            "segmentation": segmentation}


def test_defaults_fill_in_and_presets_apply():
    exp = Experiment(_cfg())
    assert exp.segmentation == SEGMENTATION_DEFAULTS
    cells, nuclei = masks.omnipose_settings(exp)
    assert cells["diameter"] == OMNIPOSE_PRESETS["S.cerevisiae"]["diameter"] and cells["mask_threshold"] == -3
    assert cells["min_size"] == 12 and nuclei["min_size"] == 8          # from "parameters"
    assert cells["tile"] is False and nuclei["tile"] is False


def test_tile_and_overrides_reach_omnipose():
    exp = Experiment(_cfg(tile=True, cells={"diameter": 40, "flow_threshold": 0.5}, nuclei={"mask_threshold": 1}))
    cells, nuclei = masks.omnipose_settings(exp)
    assert cells["tile"] is True and nuclei["tile"] is True
    assert cells["diameter"] == 40 and cells["flow_threshold"] == 0.5 and cells["mask_threshold"] == -3
    assert nuclei["mask_threshold"] == 1 and nuclei["diameter"] == 0.0


def test_unknown_omnipose_setting_is_an_error():
    with pytest.raises(ValueError, match="segmentation/cells"):
        validate(_cfg(cells={"diamter": 40}))


def test_gpu_out_of_memory_names_the_setting():
    class Model:
        def eval(self, image, **params):
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    with pytest.raises(masks.GpuOutOfMemoryError, match='"tile": true'):
        masks.run_omnipose(None, Model(), {"tile": False}, "cells")
