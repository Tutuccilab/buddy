"""Step 1 with a multi-page TIFF instead of a .vsi: OME-TIFF and ImageJ hyperstacks, with and without channel names."""
import json

import numpy as np
import pytest
import tifffile

from buddy.experiment import Experiment
from buddy.tiff_input import find_dic_file, is_stack, read_stack

Z, Y, X = 5, 64, 48


def _stack(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 4000, (Z, 3, Y, X), np.uint16)       # ZCYX: channels CY5, CY3, DAPI


def _experiment(tmp_path, **extra):
    cfg = {"inputdir": str(tmp_path / "in"), "outputdir": str(tmp_path / "out"),
           "parameters": {"scale": [200, 65, 65], "spot_radius": [880, 120, 120], "sigma": [0.75, 2.3, 2.3],
                          "patch_size": [200, 200], "organism": "S.cerevisiae", "min_size": 10, **extra.pop("parameters", {})},
           "channels": {"ASH1CLB2": {"filter": "CY5", "mrna": "ASH1CLB2", "threshold": 50},
                        "SUN4": {"filter": "CY3", "mrna": "SUN4", "threshold": 50},
                        "DAPI": {"filter": "DAPI", "mrna": "DAPI", "threshold": None}}, **extra}
    (tmp_path / "in").mkdir(exist_ok=True)
    f = tmp_path / "config.json"; json.dump(cfg, open(f, "w"))
    return Experiment.from_cfg_file(str(f))


def test_ome_tiff_with_channel_names(tmp_path):
    exp = _experiment(tmp_path)
    data = _stack()
    p = tmp_path / "in" / "img_01.ome.tif"
    tifffile.imwrite(p, data, ome=True, metadata={"axes": "ZCYX", "Channel": {"Name": ["CY5", "CY3", "DAPI"]}})
    info = is_stack(p)
    assert info["channel_names"] == ["CY5", "CY3", "DAPI"] and info["ome"]
    channels, dic = read_stack(p, exp)
    assert set(channels) == {"ASH1CLB2", "SUN4", "DAPI"} and dic is None
    assert np.array_equal(channels["SUN4"], data[:, 1]) and channels["DAPI"].shape == (Z, Y, X)


def test_imagej_hyperstack(tmp_path):
    exp = _experiment(tmp_path)
    data = _stack()                                                # ZCYX, as Fiji writes hyperstacks
    p = tmp_path / "in" / "img_02.tif"
    tifffile.imwrite(p, data, imagej=True, metadata={"axes": "ZCYX", "Labels": ["CY5", "CY3", "DAPI"] * Z})
    info = is_stack(p)
    assert info["imagej"] and info["axes"] == "ZCYX" and info["channel_names"] == ["CY5", "CY3", "DAPI"]
    channels, _ = read_stack(p, exp)
    assert np.array_equal(channels["ASH1CLB2"], data[:, 0]) and np.array_equal(channels["DAPI"], data[:, 2])


def test_other_axis_orders_are_handled(tmp_path):
    exp = _experiment(tmp_path)
    data = _stack()
    p = tmp_path / "in" / "img_08.ome.tif"
    tifffile.imwrite(p, np.transpose(data, (1, 0, 2, 3))[None], ome=True,        # TCZYX with T = 1
                     metadata={"axes": "TCZYX", "Channel": {"Name": ["CY5", "CY3", "DAPI"]}})
    channels, _ = read_stack(p, exp)
    assert np.array_equal(channels["SUN4"], data[:, 1])


def test_without_channel_names_needs_tiff_channel(tmp_path):
    exp = _experiment(tmp_path)
    data = _stack()
    p = tmp_path / "in" / "img_03.tif"
    tifffile.imwrite(p, data, imagej=True, metadata={"axes": "ZCYX"})
    assert is_stack(p)["channel_names"] == []
    with pytest.raises(ValueError, match="tiff_channel"):
        read_stack(p, exp)
    exp = _experiment(tmp_path)
    for key, idx in (("ASH1CLB2", 0), ("SUN4", 1), ("DAPI", 2)):
        exp.channels[key]["tiff_channel"] = idx
    channels, _ = read_stack(p, exp)
    assert np.array_equal(channels["SUN4"], data[:, 1])


def test_dic_as_a_channel_of_the_stack(tmp_path):
    exp = _experiment(tmp_path)
    data = np.concatenate([_stack(), np.full((Z, 1, Y, X), 7, np.uint16)], axis=1)
    data[Z // 2, 3] = 9                                            # the middle plane of the DIC channel
    p = tmp_path / "in" / "img_04.ome.tif"
    tifffile.imwrite(p, data, ome=True, metadata={"axes": "ZCYX", "Channel": {"Name": ["CY5", "CY3", "DAPI", "DIC"]}})
    channels, dic = read_stack(p, exp)
    assert dic.shape == (Y, X) and (dic == 9).all()
    assert set(channels) == {"ASH1CLB2", "SUN4", "DAPI"}


def test_image_list_and_dic_file(tmp_path):
    exp = _experiment(tmp_path)
    inp = tmp_path / "in"
    tifffile.imwrite(inp / "img_05.ome.tif", _stack(), ome=True, metadata={"axes": "ZCYX", "Channel": {"Name": ["CY5", "CY3", "DAPI"]}})
    tifffile.imwrite(inp / "img_05_DIC.tif", np.zeros((Y, X), np.uint16))
    tifffile.imwrite(inp / "img_06.ome.tif", _stack(), ome=True, metadata={"axes": "ZCYX", "Channel": {"Name": ["CY5", "CY3", "DAPI"]}})
    assert find_dic_file(inp, "img_05") == "img_05_DIC.tif" and find_dic_file(inp, "img_06") is None
    exp.create_image_list()
    assert list(exp.images) == ["img_05"]                          # img_06 has no DIC image -> skipped with a warning
    assert exp.images["img_05"] == {"vsi_file": "img_05.ome.tif", "cell_file": "img_05_DIC.tif", "valid": True}
    assert is_stack(inp / "img_05_DIC.tif") is None


def test_step1_from_tiff_writes_the_usual_files(tmp_path):
    """pipeline.read_image on a TIFF stack gives the same files as from a .vsi"""
    from buddy import pipeline
    exp = _experiment(tmp_path)
    inp = tmp_path / "in"
    data = _stack()
    tifffile.imwrite(inp / "img_07.ome.tif", data, ome=True, metadata={"axes": "ZCYX", "Channel": {"Name": ["CY5", "CY3", "DAPI"]}})
    tifffile.imwrite(inp / "img_07_DIC.tif", np.full((Y, X), 3, np.uint16))
    exp.create_image_list()
    json_file = pipeline.read_image(exp.images["img_07"], exp)
    out = tmp_path / "out" / "img_07"
    assert json_file == out / "img.json"
    for ch in ("ASH1CLB2", "SUN4", "DAPI"):
        assert tifffile.imread(out / f"{ch}.tif").shape == (Z, Y, X)
    assert np.array_equal(tifffile.imread(out / "SUN4.tif"), data[:, 1])
    assert (tifffile.imread(out / "DIC.tif") == 3).all()
    meta = json.load(open(json_file))
    assert meta["vsi_file"] == "img_07.ome.tif" and "01-configure" in meta["time"]
