"""The config file -> Experiment: defaults, and the re-run check that skips / overwrites processed images."""
import json

import pytest

from buddy.experiment import BUD_CLASSIFICATION_DEFAULTS, CELL_FLAG_DEFAULTS, Experiment
from buddy.which_images import select_images, step_settings

MINIMAL = {
    "inputdir": "in", "outputdir": "out",
    "parameters": {"scale": [200, 65, 65], "spot_radius": [880, 120, 120], "sigma": [0.75, 2.3, 2.3],
                   "patch_size": [200, 200], "organism": "S.cerevisiae", "min_size": 10},
    "channels": {"ASH1CLB2": {"filter": "CY5", "mrna": "ASH1CLB2", "threshold": 50},
                 "SUN4": {"filter": "CY3", "mrna": "SUN4"},
                 "DAPI": {"filter": "DAPI", "mrna": "DAPI", "threshold": None}},
}


def experiment(tmp_path, **extra):
    cfg = {**MINIMAL, **extra}
    cfg["inputdir"], cfg["outputdir"] = str(tmp_path / "in"), str(tmp_path / "out")
    f = tmp_path / "config.json"
    json.dump(cfg, open(f, "w"))
    return Experiment.from_cfg_file(str(f))


def test_defaults_when_blocks_are_missing(tmp_path):
    exp = experiment(tmp_path)
    assert exp.bud_classification == BUD_CLASSIFICATION_DEFAULTS and exp.bud_classification["enabled"] is False
    assert exp.bud_classification["snap"] is True
    assert exp.cell_flags == CELL_FLAG_DEFAULTS and exp.cell_flags["fold"] == 8
    assert exp.use_gpu is True
    assert exp.channels["SUN4"]["threshold"] is None                   # a missing threshold means automatic
    assert exp.filter2mRNA == {"CY5": "ASH1CLB2", "CY3": "SUN4", "DAPI": "DAPI"}


def test_config_overrides_defaults(tmp_path):
    exp = experiment(tmp_path, bud_classification={"enabled": True, "model": "with_ASH1CLB2"}, cell_flags={"fold": 12},
                     parameters={**MINIMAL["parameters"], "use_gpu": False})
    assert exp.bud_classification["enabled"] is True and exp.bud_classification["model"] == "with_ASH1CLB2"
    assert exp.bud_classification["overwrite_manual_corrections"] is False      # untouched default
    assert exp.cell_flags["fold"] == 12 and exp.cell_flags["max_gap_px"] == 10
    assert exp.use_gpu is False


def test_step_settings_ignore_gpu_but_see_the_model(tmp_path):
    a = experiment(tmp_path, parameters={**MINIMAL["parameters"], "use_gpu": True})
    b = experiment(tmp_path, parameters={**MINIMAL["parameters"], "use_gpu": False})
    for step in ("read", "segment", "spots"):
        assert step_settings(a, step) == step_settings(b, step)
    c = experiment(tmp_path, bud_classification={"enabled": True, "model": "with_ASH1CLB2", "ash_channel": "ASH1CLB2"})
    assert step_settings(c, "spots")["bud_model"] == "with_ASH1CLB2 (ASH1CLB2)"
    assert step_settings(a, "spots")["bud_model"] == "off"
    assert step_settings(a, "spots")["thresholds"] == {"ASH1CLB2": 50, "SUN4": "auto"}
    with pytest.raises(ValueError):
        step_settings(a, "nonsense")


def _image(folder, exp, last_run, settings=None):
    folder.mkdir(parents=True, exist_ok=True)
    meta = {"savepath": str(folder), "parameters": exp.parameters, "last_run": last_run,
            "results": {"settings": settings or {}}}
    json.dump(meta, open(folder / "img.json", "w"))
    return folder / "img.json"


def test_select_images_new_same_and_changed(tmp_path, capsys):
    exp = experiment(tmp_path, if_already_processed="skip")
    same = _image(tmp_path / "out" / "a", exp, "spots", {"spots": step_settings(exp, "spots")})
    other = step_settings(exp, "spots"); other["thresholds"] = {"ASH1CLB2": 99, "SUN4": "auto"}
    changed = _image(tmp_path / "out" / "b", exp, "spots", {"spots": other})
    not_yet = _image(tmp_path / "out" / "c", exp, "segment")                 # step 3 never ran on it
    todo = select_images(exp, "spots", candidates=[same, changed, not_yet])
    assert todo == [not_yet]                                                # skip policy: the changed one is left alone
    assert "1 new, 1 already processed" in capsys.readouterr().out
    exp.if_already_processed = "overwrite"
    assert select_images(exp, "spots", candidates=[same, changed, not_yet]) == [not_yet, changed]
    exp.if_already_processed = "ask"
    assert select_images(exp, "spots", candidates=[same, changed], ask=lambda text: "99" in text) == [changed]
    assert select_images(exp, "spots", candidates=[same, changed], ask=lambda text: False) == []


def test_segment_rerun_moves_corrections_aside(tmp_path):
    exp = experiment(tmp_path, if_already_processed="overwrite")
    old = step_settings(exp, "segment"); old["min_size"] = 99
    img = _image(tmp_path / "out" / "a", exp, "spots", {"segment": old})
    (img.parent / "corrected").mkdir(); (img.parent / "corrected" / "bud_corrected_masks.tif").write_bytes(b"x")
    assert select_images(exp, "segment", candidates=[img]) == [img]
    assert not (img.parent / "corrected").exists()
    assert len(list(img.parent.glob("corrected_replaced_*"))) == 1
