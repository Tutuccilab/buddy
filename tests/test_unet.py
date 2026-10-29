"""The plain-torch U-Net of the bud model: same module names as segmentation_models_pytorch's Unet (so the released
weights load unchanged), and the same outputs (checked against smp 0.5.0 when it is installed)."""
from pathlib import Path

import pytest
import torch

from buddy.buds import bud_model
from buddy.buds.unet import Unet

WEIGHTS = Path(bud_model.DEFAULT_WEIGHTS_DIR) / bud_model.MODELS["DIC_DAPI_only"]["file"]


def test_shapes_and_keys():
    net = Unet(in_channels=4)
    keys = list(net.state_dict())
    assert keys[0] == "encoder.conv1.weight" and "decoder.blocks.0.conv1.0.weight" in keys
    assert keys[-1] == "classification_head.3.bias" and len(keys) == 280
    net.eval()
    with torch.no_grad():
        maps, classes = net(torch.zeros(1, 4, 64, 64))
    assert maps.shape == (1, 2, 64, 64) and classes.shape == (1, 3)


@pytest.mark.skipif(not WEIGHTS.exists(), reason="released bud-model weights not downloaded")
def test_released_weights_load_strictly():
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    bud_model.build_network(3).load_state_dict(state, strict=True)


def test_same_as_segmentation_models_pytorch():
    smp = pytest.importorskip("segmentation_models_pytorch")
    torch.manual_seed(0)
    ref = smp.Unet("resnet34", encoder_weights=None, in_channels=4, classes=2, aux_params=dict(classes=3, dropout=0.2))
    mine = Unet(in_channels=4)
    mine.load_state_dict(ref.state_dict(), strict=True)
    ref.eval(), mine.eval()
    x = torch.randn(2, 4, 128, 128)
    with torch.no_grad():
        (a, b), (c, d) = ref(x), mine(x)
    assert torch.equal(a, c) and torch.equal(b, d)
