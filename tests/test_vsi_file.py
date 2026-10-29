"""
Reading a real Olympus .vsi (tests/data/vsi, release test-data-vsi; slow: ~400 MB, Java): the series names, the
channel stacks and the DIC pairing come out as recorded in vsi_reference.json (tests/make_vsi_reference.py).
"""
import hashlib
import json

import numpy as np
import pytest

from buddy.config import example_config
from buddy.experiment import Experiment
from buddy.reading.readers import read_vsi
from buddy.reading.vsi import BioFormatsFile

pytestmark = pytest.mark.slow


def md5(array):
    return hashlib.md5(np.ascontiguousarray(array).tobytes()).hexdigest()


def test_series_and_channels_match_the_reference(vsi_folder):
    ref = json.load(open(vsi_folder / "vsi_reference.json", encoding="utf-8"))
    vsi = vsi_folder / ref["vsi"]
    with BioFormatsFile(vsi) as f:
        assert f.series_names == ref["series_names"]
        # T, C, Z, Y, X; bioio (the reference) adds a 6th axis for the RGB "macro image", which buddy never reads
        assert [list(f.shape(i)) for i in range(len(f.series_names))] == [shape[:5] for shape in ref["shapes"]]
    channels = {name: ch["filter"] for name, ch in example_config()["channels"].items() if not name.startswith("_")}
    stacks = read_vsi(vsi, {f: name for name, f in channels.items()})
    assert set(stacks) == set(ref["channels"])
    for ch, want in ref["channels"].items():
        got = stacks[ch]
        assert list(got.shape) == want["shape"] and str(got.dtype) == want["dtype"], ch
        assert md5(got) == want["md5"], f"{ch}: pixels differ from the reference"


def test_the_vsi_is_paired_with_its_dic_file(vsi_folder, tmp_path):
    ref = json.load(open(vsi_folder / "vsi_reference.json", encoding="utf-8"))
    cfg = example_config()
    cfg["inputdir"], cfg["outputdir"] = str(vsi_folder), str(tmp_path / "out")
    exp = Experiment(cfg)
    exp.create_image_list()
    images = list(exp.images.values())
    assert len(images) == 1 and images[0]["vsi_file"] == ref["vsi"] and images[0]["cell_file"] == ref["dic"]
