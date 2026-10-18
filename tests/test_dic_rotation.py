"""DIC turn: measured per image, a fixed turn from the config, and the warning on unclear lighting."""
import json
import logging

import numpy as np
import pytest

from buddy.buds.bud_model import BudModel
from buddy.config import validate

GOOD = {"inputdir": "in", "outputdir": "out",
        "parameters": {"scale": [200, 65, 65], "spot_radius": [880, 120, 120], "sigma": [0.75, 2.3, 2.3]},
        "channels": {"SUN4": {"filter": "CY3"}}}


def test_config_dic_rotation():
    cfg = validate(json.loads(json.dumps(GOOD)))
    assert cfg["bud_classification"]["dic_rotation"] is None
    bad = json.loads(json.dumps(GOOD)); bad["bud_classification"] = {"dic_rotation": 5}
    with pytest.raises(ValueError, match="dic_rotation"):
        validate(bad)


class _Net:
    """stand-in network: records the inputs, returns empty maps"""
    def __init__(self):
        self.inputs = []

    def __call__(self, x):
        import torch
        self.inputs.append(x.clone())
        n, _, h, w = x.shape
        return torch.full((n, 2, 2 * h, 2 * w), -10.0), torch.tensor([[5.0, 0.0, 0.0]] * n)


def _field():
    """one round cell, flat DIC (no lighting direction to measure)"""
    yy, xx = np.mgrid[:200, :200]
    masks = (((yy - 100) ** 2 + (xx - 100) ** 2) < 30 ** 2).astype(np.int64)
    return np.zeros((200, 200), np.float32), masks


def test_fixed_turn_is_used_and_unclear_lighting_warns(caplog):
    dic, masks = _field()
    m = BudModel(_Net(), "DIC_DAPI_only", "cpu", crop=64, dic_rotation=3)
    m.predict_cells_normalised(dic, dic, masks)
    assert m.last_rotation == 3
    m = BudModel(_Net(), "DIC_DAPI_only", "cpu", crop=64)
    with caplog.at_level(logging.WARNING, logger="buddy"):
        m.predict_cells_normalised(dic, dic, masks)
    assert "DIC lighting direction unclear" in caplog.text
