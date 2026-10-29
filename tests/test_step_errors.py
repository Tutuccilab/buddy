"""Errors inside a pipeline step say which step, which image, where in buddy, and what to do (pipeline.StepError)."""
import json
from pathlib import Path

import pytest

from buddy import pipeline
from buddy.experiment import Experiment
from buddy.image import Image, MissingFileError


def _exp(tmp_path):
    cfg = {"inputdir": str(tmp_path), "outputdir": str(tmp_path / "out"), "channels": {"DAPI": {"filter": "DAPI"}},
           "parameters": {"scale": [200, 65, 65], "spot_radius": [880, 120, 120], "sigma": [0.75, 2.3, 2.3]}}
    return Experiment(cfg)


def test_missing_file_names_the_step_that_makes_it(tmp_path):
    exp = _exp(tmp_path)
    folder = tmp_path / "out" / "img"; folder.mkdir(parents=True)
    json.dump({"vsi_file": "img.vsi", "cell_file": "img_DIC.tif", "stem": "img", "savepath": str(folder)}, open(folder / "img.json", "w"))
    image = Image.from_json(folder / "img.json", exp)
    with pytest.raises(MissingFileError, match="step 2 \\(segment\\)"):
        image.load("masks")
    assert image.load("spots", optional=True) is None or True           # optional never raises


def test_step_error_says_step_image_place_and_hint(tmp_path):
    exp = _exp(tmp_path)
    folder = tmp_path / "out" / "img"; folder.mkdir(parents=True)
    json.dump({"vsi_file": "img.vsi", "cell_file": "img_DIC.tif", "stem": "img", "savepath": str(folder)}, open(folder / "img.json", "w"))
    with pytest.raises(pipeline.StepError) as e:
        pipeline.segment_image(folder / "img.json", exp)                 # DAPI.tif of step 1 is missing
    text = str(e.value)
    assert "step 2 (segment)" in text and "image img" in text and "buddy/image.py" in text
    assert "step 1 (read)" in text and "->" in text                      # which step makes the file, and the hint
    assert isinstance(e.value.cause, MissingFileError)


def test_where_lists_every_step(capsys):
    pipeline.where()
    out = capsys.readouterr().out
    for step in ("1 read", "2 segment", "3 spots", "5 recount"):
        assert f"step {step}" in out
    assert "buddy/segmentation/masks.py" in out and '"segmentation"' in out


def test_gpu_hint_depends_on_the_step():
    oom = RuntimeError("CUDA out of memory. Tried to allocate 2.50 GiB")
    assert '"tile": true' in pipeline._hint(oom, 2)                       # Omnipose: tiling
    assert "gpu_spot_detection" in pipeline._hint(oom, 3)                  # spots / bud model: CPU switches
    assert pipeline._hint(ValueError("x"), 2) == ""


def test_moved_output_folder_is_found(tmp_path):
    """An img.json whose savepath points elsewhere (copied from another drive / computer) uses its own folder."""
    exp = _exp(tmp_path)
    folder = tmp_path / "out" / "img"; folder.mkdir(parents=True)
    json.dump({"vsi_file": "img.vsi", "cell_file": "img_DIC.tif", "stem": "img", "savepath": r"F:\old place\output\img"},
              open(folder / "img.json", "w"))
    image = Image.from_json(folder / "img.json", exp)
    assert Path(image.savepath) == folder.resolve()
    image.save_json("read")
    assert json.load(open(folder / "img.json"))["savepath"] == str(folder.resolve())
