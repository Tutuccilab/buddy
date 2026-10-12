"""
The config file: defaults, checks and loading. Everything a run depends on is here, so a typo in a setting is
reported instead of silently ignored.

    from buddy.config import load_config
    cfg = load_config("config.json")      # dict with every block filled in with its defaults

Blocks of the config file (all settings explained in INSTALLATION_TUTORIAL.md):
    inputdir, outputdir        where the images are / where the results go
    parameters                 spot detection, segmentation, GPU use (PARAMETER_DEFAULTS)
    channels                   {name: {filter, mrna, threshold, colormap, tiff_channel?}}
    bud_classification         the bud model (BUD_CLASSIFICATION_DEFAULTS)
    cell_flags                 napari flags for high-ASH1/CLB2 cells (CELL_FLAG_DEFAULTS)
    cell_cycle                 cell-cycle phase per cell (CELL_CYCLE_DEFAULTS)
    if_already_processed       "ask" | "skip" | "overwrite" (what to do with images processed with other settings)
    experiments                list of {strain, condition} (used by the data-wrangling notebook)
"""
import json
import logging
from pathlib import Path

log = logging.getLogger("buddy")

PARAMETER_DEFAULTS = {
    "scale": None,                 # voxel size (z, y, x) in nm, e.g. [200, 65, 65]           - required
    "spot_radius": None,           # expected spot radius (z, y, x) in nm, e.g. [880, 120, 120] - required
    "sigma": None,                 # background filter Gaussian (z, y, x) in px, e.g. [0.75, 2.3, 2.3] - required
    "patch_size": [200, 200],      # patch with the most cells, used to find the in-focus slices
    "organism": "S.cerevisiae",    # Omnipose settings: "S.cerevisiae" or "E.coli"
    "min_size": 10,                # Omnipose: smallest cell (px)
    "nuclei_min_size": 10,         # Omnipose: smallest nucleus (px)
    "min_cell_area": 20,           # clean-up: cells smaller than this (px) are dropped
    "expand_px": 0,                # clean-up: grow every cell by this many px
    "remove_edge_cells": False,    # clean-up: drop cells touching the image border
    "use_gpu": True,               # Omnipose + bud model on the GPU when PyTorch sees one
    "gpu_spot_detection": True,    # GPU versions of the background filter, spot detection and decomposition
    "dic_channel": "DIC",          # TIFF input: channel name (or index) of the DIC image inside the stack
    "tiff_axes": None,             # TIFF input: axis order (e.g. "ZCYX") for files without that information
}
REQUIRED_PARAMETERS = ("scale", "spot_radius", "sigma")

CHANNEL_DEFAULTS = {
    "filter": None,                # filter name in the microscope file (the scene name of a .vsi) - required
    "mrna": None,                  # name of the mRNA, used for the output files; default: the channel name
    "threshold": None,             # spot detection threshold; None = Big-FISH's automatic threshold
    "colormap": "gray",            # napari colour
    "tiff_channel": None,          # TIFF input: channel index for files without channel names
}

# automatic mother/bud assignment: used when the config file has no "bud_classification" block
BUD_CLASSIFICATION_DEFAULTS = {
    "enabled": False,                       # true -> classify_buds() writes the bud masks automatically
    "model": "DIC_DAPI_only",               # or "with_ASH1CLB2" (needs the ASH1/CLB2 channel)
    "ash_channel": "ASH1CLB2",              # channel name (key in "channels") used by the with_ASH1CLB2 model
    "overwrite_manual_corrections": False,  # never replace bud masks you corrected by hand, unless true
    "weights_dir": None,                    # folder with the model weights; None = buddy/models (auto-download)
    "snap": True,                           # a claimed bud lying mostly inside one Omnipose object takes that whole object
}

# napari flags for unbudded cells with high ASH1/CLB2 (late M / early S, see buddy/cell_flags.py)
CELL_FLAG_DEFAULTS = {
    "enabled": True,              # false -> no cell_flags.csv
    "ash_channel": None,          # None = the "ash_channel" of "bud_classification"
    "fold": 8,                    # flag cells with >= fold x the ASH1/CLB2 of a typical unbudded cell
    "nucleus_area": 0.5,          # "own full nucleus": area >= 0.5 x ...
    "nucleus_intensity": 0.7,     # ... and mean DAPI >= 0.7 x those of a typical unbudded cell
    "max_gap_px": 10,             # neighbours: at most this many pixels apart
}

# cell-cycle phase per cell -> cell_cycle.csv and the napari "cell cycle" layers (see buddy/quantification/cell_cycle.py)
CELL_CYCLE_DEFAULTS = {
    "enabled": True,              # false -> no cell_cycle.csv
    "ash_channel": None,          # None = the "ash_channel" of "bud_classification"; without it no G1 call
    "g1_max_ash": 5,              # G1: no bud and at most this many ASH1/CLB2 mRNAs
    "s_g2_bud_um2": 2.0,          # bud area (µm²) that separates S from G2 (and the smallest bud for M)
    "max_bud_um2": 15.0,          # G2: buds up to this area (µm²)
    "min_length_um": 2.0,         # cells shorter or longer than this (long axis, µm) are not assessed
    "max_length_um": 20.0,
}

POLICIES = ("ask", "skip", "overwrite")
TOP_LEVEL = {"inputdir", "outputdir", "parameters", "channels", "bud_classification", "cell_flags", "cell_cycle",
             "if_already_processed", "experiments", "cfg_file"}


def _fill(block, defaults, where):
    """defaults + the user's values; unknown keys are reported (a typo would otherwise be ignored)"""
    block = dict(block or {})
    unknown = sorted(set(block) - set(defaults))
    if unknown:
        log.warning(f'config "{where}": unknown setting(s) {unknown} are ignored (known: {sorted(defaults)})')
    return {**defaults, **{k: v for k, v in block.items() if k in defaults}}


def validate(cfg):
    """Fill in the defaults and check the config dict in place. Raises ValueError for what cannot be guessed."""
    for key in ("inputdir", "outputdir", "channels"):
        if key not in cfg:
            raise ValueError(f'config: "{key}" is missing')
    unknown = sorted(set(cfg) - TOP_LEVEL)
    if unknown:
        log.warning(f"config: unknown top-level block(s) {unknown} are ignored")
    cfg["parameters"] = _fill(cfg.get("parameters"), PARAMETER_DEFAULTS, "parameters")
    missing = [k for k in REQUIRED_PARAMETERS if cfg["parameters"].get(k) is None]
    if missing:
        raise ValueError(f'config "parameters": {missing} must be set (voxel size, spot radius and filter sigma in z, y, x)')
    for name, ch in cfg["channels"].items():
        ch = _fill(ch, CHANNEL_DEFAULTS, f"channels/{name}")
        if ch["filter"] is None:
            raise ValueError(f'config "channels/{name}": "filter" (the channel name in the microscope file) is missing')
        if ch["mrna"] is None:
            ch["mrna"] = name
        cfg["channels"][name] = ch
    cfg["bud_classification"] = _fill(cfg.get("bud_classification"), BUD_CLASSIFICATION_DEFAULTS, "bud_classification")
    cfg["cell_flags"] = _fill(cfg.get("cell_flags"), CELL_FLAG_DEFAULTS, "cell_flags")
    cfg["cell_cycle"] = _fill(cfg.get("cell_cycle"), CELL_CYCLE_DEFAULTS, "cell_cycle")
    policy = cfg.setdefault("if_already_processed", "ask")
    if policy not in POLICIES:
        raise ValueError(f'config: "if_already_processed" must be one of {POLICIES}, not {policy!r}')
    cfg.setdefault("experiments", [])
    return cfg


def load_config(cfg_file):
    """Read a config file and fill in the defaults (see validate)."""
    cfg_file = Path(cfg_file)
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["cfg_file"] = str(cfg_file)
    return validate(cfg)
