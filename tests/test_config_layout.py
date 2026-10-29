"""The config file layout of buddy 0.6: one block per step, the old "parameters" layout still read, "_" comments
ignored, write_config readable, config_example.json in step with config.py."""
import json
from pathlib import Path

import pytest

from buddy.config import (
    CELL_FLAG_DEFAULTS,
    OMNIPOSE_PRESETS,
    example_config,
    file_layout,
    load_config,
    omnipose_overrides,
    to_json,
    validate,
    write_config,
)

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = {"SUN4": {"filter": "CY3", "threshold": 50}, "DAPI": {"filter": "DAPI"}}
NEW = {"inputdir": "in", "outputdir": "out", "channels": CHANNELS,
       "segmentation": {"organism": "E.coli", "tile": True, "min_size": 15, "cells": {"diameter": 30}, "remove_edge_cells": True},
       "spots": {"scale": [200, 65, 65], "spot_radius": [880, 120, 120], "sigma": [0.75, 2.3, 2.3]},
       "hardware": {"use_gpu": False}}
OLD = {"inputdir": "in", "outputdir": "out", "channels": CHANNELS,
       "parameters": {"scale": [200, 65, 65], "spot_radius": [880, 120, 120], "sigma": [0.75, 2.3, 2.3],
                      "organism": "E.coli", "min_size": 15, "remove_edge_cells": True, "use_gpu": False},
       "segmentation": {"tile": True, "cells": {"diameter": 30}}}


def _copy(d):
    return json.loads(json.dumps(d))


def test_new_and_old_layout_give_the_same_settings():
    new, old = validate(_copy(NEW)), validate(_copy(OLD))
    assert new["parameters"] == old["parameters"] and new["segmentation"] == old["segmentation"]
    assert new["parameters"]["organism"] == "E.coli" and new["parameters"]["min_size"] == 15
    assert new["parameters"]["use_gpu"] is False and new["parameters"]["patch_size"] == [200, 200]   # default filled in
    assert new["segmentation"] == {"cell_model": "cyto2_omni", "nucleus_model": "nuclei", "tile": True, "use_preset": False,
                                   "cells": {"diameter": 30}, "nuclei": {}}
    assert new["parameters"]["decomposition"] == {"alpha": 0.5, "beta": 2, "gamma": 1}      # default filled in
    assert new["spots"]["scale"] == [200, 65, 65] and new["hardware"]["gpu_spot_detection"] is True


def test_validate_is_idempotent_and_blocks_win_over_parameters():
    cfg = validate(_copy(NEW))
    assert validate(_copy(cfg)) == cfg
    mixed = _copy(OLD); mixed["spots"] = {"scale": [300, 65, 65]}
    assert validate(mixed)["parameters"]["scale"] == [300, 65, 65]


def test_comments_are_ignored_everywhere():
    cfg = _copy(NEW)
    cfg["_about"] = ["a comment"]; cfg["spots"]["_about"] = "x"; cfg["channels"]["_about"] = ["y"]
    cfg["segmentation"]["cells"]["_note"] = "z"
    checked = validate(cfg)
    assert "_about" not in checked and "_about" not in checked["spots"] and list(checked["channels"]) == ["SUN4", "DAPI"]
    assert checked["segmentation"]["cells"] == {"diameter": 30}


def test_unknown_keys_are_reported(caplog):
    cfg = _copy(NEW); cfg["spots"]["sigmaa"] = [1, 1, 1]
    validate(cfg)
    assert "sigmaa" in caplog.text
    cfg = _copy(NEW); cfg["segmentation"]["cells"]["diamter"] = 40
    with pytest.raises(ValueError, match="diamter"):
        validate(cfg)
    cfg = _copy(NEW); del cfg["spots"]["sigma"]
    with pytest.raises(ValueError, match="sigma"):
        validate(cfg)


def test_write_config_is_readable_and_round_trips(tmp_path):
    path = write_config(validate(_copy(OLD)), tmp_path / "c.json")
    text = path.read_text(encoding="utf-8")
    assert '"scale": [200, 65, 65]' in text and '"parameters"' not in text and "\n\n" in text
    assert list(json.loads(text))[:4] == ["inputdir", "outputdir", "if_already_processed", "channels"]
    again = load_config(path)
    assert again["parameters"] == validate(_copy(OLD))["parameters"]


def test_relative_folders_are_relative_to_the_config_file(tmp_path):
    (tmp_path / "data").mkdir()
    cfg = _copy(NEW); cfg["inputdir"] = "data"; cfg["outputdir"] = "data/output"
    write_config(cfg, tmp_path / "sub" / "c.json")
    loaded = load_config(tmp_path / "sub" / "c.json")
    assert Path(loaded["inputdir"]) == (tmp_path / "sub" / "data").resolve()


def test_example_file_is_in_step_with_config_py():
    """config_files/config_example.json is written by write_config(example_config()); regenerate it when this fails."""
    path = ROOT / "config_files" / "config_example.json"
    assert path.read_text(encoding="utf-8") == to_json(file_layout(example_config())) + "\n"
    cfg = load_config(path)
    assert cfg["parameters"]["scale"] == [200, 65, 65]
    # the example spells out the yeast preset with use_preset true: no overrides, cell_flags not in the file
    assert cfg["segmentation"]["use_preset"] is True and cfg["segmentation"]["cells"]["diameter"] == 50
    assert omnipose_overrides(cfg["parameters"], cfg["segmentation"]) == ({}, {})
    text = path.read_text(encoding="utf-8")
    assert '"cell_flags"' not in text and '"_values"' in text
    assert list(json.loads(text))[5:8] == ["channels", "spots", "segmentation"]   # after _about, _values and the folders


def test_use_preset_and_decomposition(caplog, tmp_path):
    # use_preset false (or absent): cells / nuclei override the preset, values equal to it are no override
    cfg = _copy(NEW); cfg["segmentation"]["cells"] = {"diameter": 30, "mask_threshold": OMNIPOSE_PRESETS["E.coli"]["mask_threshold"]}
    checked = validate(cfg)
    assert omnipose_overrides(checked["parameters"], checked["segmentation"]) == ({"diameter": 30}, {})
    # use_preset true: ignored, with a warning naming the ignored keys
    cfg = _copy(NEW); cfg["segmentation"]["use_preset"] = True
    checked = validate(cfg)
    assert omnipose_overrides(checked["parameters"], checked["segmentation"]) == ({}, {})
    assert "use_preset is true" in caplog.text and "diameter" in caplog.text
    cfg = _copy(NEW); cfg["segmentation"]["use_preset"] = None
    with pytest.raises(ValueError, match="use_preset"):
        validate(cfg)
    # decomposition: part of "spots", the same for every channel; unknown keys reported, written back in "spots"
    cfg = _copy(NEW); cfg["spots"]["decomposition"] = {"alpha": 0.7, "gamma": 0}
    checked = validate(cfg)
    assert checked["parameters"]["decomposition"] == {"alpha": 0.7, "beta": 2, "gamma": 0}
    assert file_layout(checked)["spots"]["decomposition"] == {"alpha": 0.7, "beta": 2, "gamma": 0}
    cfg = _copy(NEW); cfg["spots"]["decomposition"] = {"alfa": 0.7}
    validate(cfg)
    assert "alfa" in caplog.text
    # cell_flags: written only when it differs from the defaults
    cfg = validate(_copy(NEW))
    assert "cell_flags" not in file_layout(cfg)
    cfg["cell_flags"] = {**CELL_FLAG_DEFAULTS, "fold": 6}
    assert file_layout(cfg)["cell_flags"]["fold"] == 6 and load_config(write_config(cfg, tmp_path / "c.json"))["cell_flags"]["fold"] == 6
