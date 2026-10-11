"""buddy/config.py: defaults are filled in, typos are reported, missing essentials raise."""
import json
import logging

import pytest

from buddy.config import load_config, validate
from buddy.paths import ImageFiles

GOOD = {"inputdir": "in", "outputdir": "out",
        "parameters": {"scale": [200, 65, 65], "spot_radius": [880, 120, 120], "sigma": [0.75, 2.3, 2.3]},
        "channels": {"SUN4": {"filter": "CY3"}, "DAPI": {"filter": "DAPI", "threshold": None}}}


def test_defaults_are_filled_in():
    cfg = validate(json.loads(json.dumps(GOOD)))
    assert cfg["parameters"]["patch_size"] == [200, 200] and cfg["parameters"]["use_gpu"] is True
    assert cfg["channels"]["SUN4"] == {"filter": "CY3", "mrna": "SUN4", "threshold": None, "colormap": "gray", "tiff_channel": None}
    assert cfg["bud_classification"]["enabled"] is False and cfg["cell_flags"]["fold"] == 8
    assert cfg["if_already_processed"] == "ask" and cfg["experiments"] == []


def test_unknown_settings_are_reported(caplog):
    cfg = json.loads(json.dumps(GOOD))
    cfg["parameters"]["min_sizee"] = 5                      # a typo
    cfg["bud_classification"] = {"enabeld": True}
    with caplog.at_level(logging.WARNING, logger="buddy"):
        validate(cfg)
    assert "min_sizee" in caplog.text and "enabeld" in caplog.text
    assert cfg["bud_classification"]["enabled"] is False     # the typo did not switch it on
    assert "min_sizee" not in cfg["parameters"]


@pytest.mark.parametrize("remove, message", [("inputdir", "inputdir"), ("channels", "channels")])
def test_missing_blocks_raise(remove, message):
    cfg = json.loads(json.dumps(GOOD)); del cfg[remove]
    with pytest.raises(ValueError, match=message):
        validate(cfg)


def test_missing_required_parameters_raise():
    cfg = json.loads(json.dumps(GOOD)); del cfg["parameters"]["sigma"]
    with pytest.raises(ValueError, match="sigma"):
        validate(cfg)
    cfg = json.loads(json.dumps(GOOD)); del cfg["channels"]["SUN4"]["filter"]
    with pytest.raises(ValueError, match="filter"):
        validate(cfg)
    cfg = json.loads(json.dumps(GOOD)); cfg["if_already_processed"] = "maybe"
    with pytest.raises(ValueError, match="if_already_processed"):
        validate(cfg)


def test_load_config_keeps_the_file_name(tmp_path):
    f = tmp_path / "config.json"; json.dump(GOOD, open(f, "w"))
    cfg = load_config(f)
    assert cfg["cfg_file"] == str(f) and cfg["channels"]["DAPI"]["mrna"] == "DAPI"


def test_image_files(tmp_path):
    f = ImageFiles(tmp_path)
    assert f.channel_stack("SUN4") == tmp_path / "SUN4.tif"
    assert f.corrected_cells == tmp_path / "corrected" / "DIC_corrected_masks.tif"
    assert f.spots("SUN4").name == "SUN4_spots.npy" and f.counts("SUN4").name == "SUN4.csv"
    assert f.projection("ASH1CLB2_filtered.tif") == tmp_path / "projections" / "ASH1CLB2_filtered_max.tif"
    assert (f / "x.txt") == tmp_path / "x.txt"
