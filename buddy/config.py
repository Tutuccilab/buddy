"""
The config file: defaults, checks, loading and writing. Everything a run depends on is here, so a typo in a setting
is reported instead of silently ignored.

    from buddy.config import load_config, write_config, example_config
    cfg = load_config("config.json")      # dict with every block filled in with its defaults
    write_config(cfg, "config.json")      # the file layout below, readably formatted

The file has one block per pipeline step (all settings explained in TUTORIAL.md, section 2.1 and 5):
    inputdir, outputdir        where the images are / where the results go
    if_already_processed       "ask" | "skip" | "overwrite" (what to do with images processed with other settings)
    channels                   step 1: {name: {filter, mrna, threshold, colormap, tiff_channel?}}
    spots                      step 3: scale, spot_radius, sigma, patch_size, decomposition (SPOT_DEFAULTS)
    segmentation               step 2: Omnipose (organism preset, models, tile, min sizes, use_preset, cells/nuclei)
                               and the mask clean-up (SEGMENTATION_FILE_DEFAULTS, OMNIPOSE_PRESETS)
    bud_classification         step 3b: the bud model (BUD_CLASSIFICATION_DEFAULTS)
    cell_cycle                 cell-cycle phase per cell (CELL_CYCLE_DEFAULTS)
    hardware                   use_gpu, gpu_spot_detection (HARDWARE_DEFAULTS)
    tiff_input                 step 1, TIFF stacks only: dic_channel, tiff_axes (TIFF_INPUT_DEFAULTS)
    experiments                list of {strain, condition} (used by the data-wrangling notebook)
    cell_flags                 napari flags for high-ASH1/CLB2 cells (CELL_FLAG_DEFAULTS); not in the example and
                               only written when changed - its defaults are not meant to be tuned
Keys that start with "_" are comments and ignored ("_about" in config_example.json). true / false switch a setting
on or off; null means "not set, buddy decides" (threshold null = automatic, dic_rotation null = measured per image).

In memory (Experiment, Image, img.json) the settings of steps 1-3 stay in the flat "parameters" dict of buddy <= 0.5
(PARAMETER_DEFAULTS) plus the Omnipose part of "segmentation" (SEGMENTATION_DEFAULTS): validate() builds those from
the blocks, and a file in the old layout (a "parameters" block) is still read.
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
    "decomposition": {             # Big-FISH decompose_dense, the same for every channel (buddy/spots/detection.py)
        "alpha": 0.5,              # intensity of a single molecule as a fraction of the reference spot: more spots per region when lower
        "beta": 2,                 # a dense region must be at least beta x the reference spot: more regions when lower
        "gamma": 1,                # width of the denoising filter before the regions are found (x spot radius)
    },
    "organism": "S.cerevisiae",    # Omnipose preset for the cells: "S.cerevisiae" or "E.coli" (OMNIPOSE_PRESETS)
    "min_size": 10,                # Omnipose: smallest cell (px); more Omnipose settings: the "segmentation" block
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

# the file layout (buddy >= 0.6): which block of the file holds which of the flat PARAMETER_DEFAULTS keys
SPOT_KEYS = ("scale", "spot_radius", "sigma", "patch_size", "decomposition")
DECOMPOSITION_DEFAULTS = PARAMETER_DEFAULTS["decomposition"]
SEGMENTATION_PARAMETER_KEYS = ("organism", "min_size", "nuclei_min_size", "min_cell_area", "expand_px", "remove_edge_cells")
HARDWARE_KEYS = ("use_gpu", "gpu_spot_detection")
TIFF_INPUT_KEYS = ("dic_channel", "tiff_axes")
SPOT_DEFAULTS = {k: PARAMETER_DEFAULTS[k] for k in SPOT_KEYS}
HARDWARE_DEFAULTS = {k: PARAMETER_DEFAULTS[k] for k in HARDWARE_KEYS}
TIFF_INPUT_DEFAULTS = {k: PARAMETER_DEFAULTS[k] for k in TIFF_INPUT_KEYS}

CHANNEL_DEFAULTS = {
    "filter": None,                # filter name in the microscope file (the scene name of a .vsi) - required
    "mrna": None,                  # name of the mRNA, used for the output files; default: the channel name
    "threshold": None,             # spot detection threshold; None = Big-FISH's automatic threshold
    "colormap": "gray",            # napari colour
    "tiff_channel": None,          # TIFF input: channel index for files without channel names
}

# step 2, Omnipose (buddy/segmentation/masks.py). The settings handed to Omnipose's model.eval() are
# OMNIPOSE_PRESETS[organism] (cells) / OMNIPOSE_NUCLEUS_PRESET (nuclei), then "min_size" / "nuclei_min_size" from
# "parameters", then "cells" / "nuclei" of this block, then "tile". Every key of the presets may be overridden.
SEGMENTATION_DEFAULTS = {
    "cell_model": "cyto2_omni",    # Omnipose model for the cells (on the DIC + DAPI composite)
    "nucleus_model": "nuclei",     # Omnipose model for the nuclei (on the DAPI maximum projection)
    "tile": False,                 # run the networks on tiles: slower, needs much less GPU memory
                                   # -> set true on "CUDA out of memory" (the result can differ slightly at tile borders)
    "use_preset": False,           # true: Omnipose runs with the presets (organism / nucleus) and "cells" / "nuclei"
                                   # are ignored (a warning when they differ); false or absent: they override the preset
    "cells": {},                   # Omnipose settings for the cells that override the preset, e.g. {"diameter": 40}
    "nuclei": {},                  # idem for the nuclei
}
_OMNIPOSE_COMMON = {
    "rescale": None,               # upscale or downscale the image, None = no rescaling
    "invert": False,
    "transparency": True,          # transparency in flow output
    "omni": True,                  # Omnipose mask reconstruction
    "cluster": True,               # DBSCAN clustering
    "resample": True,              # run dynamics on the rescaled grid or the original grid
    "verbose": False,
    "niter": None,                 # None lets Omnipose choose the number of Euler iterations
    "augment": False,              # average the outputs of flipped images (slower, usually not needed)
    "affinity_seg": False,
}
OMNIPOSE_PRESETS = {               # cells, per "organism" in "parameters"
    "S.cerevisiae": {**_OMNIPOSE_COMMON, "channels": [1, 2], "mask_threshold": -3, "flow_threshold": 1, "min_size": 10, "diameter": 50},
    "E.coli": {**_OMNIPOSE_COMMON, "channels": [1, 2], "mask_threshold": 0, "flow_threshold": 0, "min_size": 20, "diameter": 0},
}
OMNIPOSE_NUCLEUS_PRESET = {**_OMNIPOSE_COMMON, "channels": [0, 0], "mask_threshold": 0.0, "flow_threshold": 0.0, "min_size": 10, "diameter": 0.0}
OMNIPOSE_KEYS = sorted(set(OMNIPOSE_NUCLEUS_PRESET) | {"tile"})
EXAMPLE_OMNIPOSE_KEYS = ("diameter", "mask_threshold", "flow_threshold")   # the ones shown in config_example.json


def omnipose_overrides(parameters, segmentation):
    """
    What the "cells" / "nuclei" of a (validated) "segmentation" block change compared to the presets: nothing when
    "use_preset" is true, otherwise the keys whose value differs from the preset. masks.omnipose_settings() applies
    these; which_images.py records them (so a config that merely spells out the preset counts as the preset).
    """
    if segmentation.get("use_preset", False):
        return {}, {}
    presets = (OMNIPOSE_PRESETS[parameters["organism"]], OMNIPOSE_NUCLEUS_PRESET)
    return tuple({k: v for k, v in (segmentation.get(which) or {}).items() if v != preset.get(k)}
                 for which, preset in zip(("cells", "nuclei"), presets))
# the "segmentation" block of the file: Omnipose (organism, models, tile, min sizes, overrides) + the clean-up
SEGMENTATION_FILE_DEFAULTS = {"organism": PARAMETER_DEFAULTS["organism"], **{k: SEGMENTATION_DEFAULTS[k] for k in ("cell_model", "nucleus_model", "tile")},
                              "min_size": PARAMETER_DEFAULTS["min_size"], "nuclei_min_size": PARAMETER_DEFAULTS["nuclei_min_size"],
                              "use_preset": False, "cells": {}, "nuclei": {},
                              **{k: PARAMETER_DEFAULTS[k] for k in ("min_cell_area", "expand_px", "remove_edge_cells")}}

# automatic mother/bud assignment: used when the config file has no "bud_classification" block
BUD_CLASSIFICATION_DEFAULTS = {
    "enabled": False,                       # true -> classify_buds() writes the bud masks automatically
    "model": "DIC_DAPI_only",               # or "with_ASH1CLB2" (needs the ASH1/CLB2 channel)
    "ash_channel": "ASH1CLB2",              # channel name (key in "channels") used by the with_ASH1CLB2 model
    "overwrite_manual_corrections": False,  # never replace bud masks you corrected by hand, unless true
    "weights_dir": None,                    # folder with the model weights; None = buddy/models (auto-download)
    "snap": True,                           # a claimed bud lying mostly inside one Omnipose object takes that whole object
    "dic_rotation": None,                   # None = measured per image; 0-3 = turn every image by this many x 90 degrees
                                            # (use when the DIC lighting cannot be measured reliably, e.g. a blurry imaging day)
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
TOP_LEVEL = {"inputdir", "outputdir", "if_already_processed", "channels", "tiff_input", "segmentation", "spots", "hardware",
             "bud_classification", "cell_flags", "cell_cycle", "experiments", "cfg_file",
             "parameters"}                  # "parameters": the layout of buddy <= 0.5, still read (and kept in memory)
BLOCK_ORDER = ("inputdir", "outputdir", "if_already_processed", "channels", "spots", "segmentation", "bud_classification",
               "cell_cycle", "hardware", "tiff_input", "experiments", "cell_flags")


def _is_comment(key):
    return isinstance(key, str) and key.startswith("_")


def _fill(block, defaults, where):
    """defaults + the user's values; unknown keys are reported (a typo would otherwise be ignored); "_..." = comments"""
    block = {k: v for k, v in dict(block or {}).items() if not _is_comment(k)}
    unknown = sorted(set(block) - set(defaults))
    if unknown:
        log.warning(f'config "{where}": unknown setting(s) {unknown} are ignored (known: {sorted(defaults)})')
    return {**defaults, **{k: v for k, v in block.items() if k in defaults}}


def _flat_parameters(cfg):
    """
    The flat "parameters" dict (PARAMETER_DEFAULTS keys) from the file's blocks: "spots", "hardware", "tiff_input"
    and the scalar keys of "segmentation"; a "parameters" block (buddy <= 0.5) underneath, the blocks win.
    """
    flat = {k: v for k, v in dict(cfg.get("parameters") or {}).items() if not _is_comment(k)}
    for block, keys in (("spots", SPOT_KEYS), ("hardware", HARDWARE_KEYS), ("tiff_input", TIFF_INPUT_KEYS),
                        ("segmentation", SEGMENTATION_PARAMETER_KEYS)):
        values = cfg.get(block) or {}
        flat.update({k: values[k] for k in keys if k in values})
    return flat


def validate(cfg):
    """
    Fill in the defaults and check the config dict in place (idempotent). Raises ValueError for what cannot be
    guessed. Afterwards cfg has both the file blocks ("spots", "hardware", "tiff_input", "segmentation" with its
    Omnipose + clean-up keys) and the flat "parameters" + Omnipose-only "segmentation" the pipeline works with.
    """
    for key in list(cfg):
        if _is_comment(key):
            del cfg[key]
    for key in ("inputdir", "outputdir", "channels"):
        if key not in cfg:
            raise ValueError(f'config: "{key}" is missing')
    unknown = sorted(set(cfg) - TOP_LEVEL)
    if unknown:
        log.warning(f"config: unknown top-level block(s) {unknown} are ignored")
    if "parameters" in cfg and not any(b in cfg for b in ("spots", "hardware", "tiff_input")):
        log.info('config in the layout of buddy <= 0.5 ("parameters" block): read as before; write_config() saves the new layout')
    # the flat parameters, checked against PARAMETER_DEFAULTS; the blocks are checked for unknown keys too
    for block, defaults in (("spots", SPOT_DEFAULTS), ("hardware", HARDWARE_DEFAULTS), ("tiff_input", TIFF_INPUT_DEFAULTS)):
        if block in cfg:
            _fill(cfg[block], defaults, block)
    flat = _fill(_flat_parameters(cfg), PARAMETER_DEFAULTS, "parameters")
    flat["decomposition"] = _fill(flat["decomposition"], DECOMPOSITION_DEFAULTS, "spots/decomposition")
    missing = [k for k in REQUIRED_PARAMETERS if flat.get(k) is None]
    if missing:
        raise ValueError(f'config "spots": {missing} must be set (voxel size, spot radius and filter sigma in z, y, x)')
    if flat["organism"] not in OMNIPOSE_PRESETS:
        raise ValueError(f'config "segmentation": "organism" must be one of {list(OMNIPOSE_PRESETS)}, not {flat["organism"]!r}')
    cfg["parameters"] = flat
    cfg["spots"] = {k: flat[k] for k in SPOT_KEYS}
    cfg["hardware"] = {k: flat[k] for k in HARDWARE_KEYS}
    cfg["tiff_input"] = {k: flat[k] for k in TIFF_INPUT_KEYS}
    # Omnipose: the file block holds organism, min sizes and the clean-up as well; in memory "segmentation" keeps only
    # the Omnipose-specific keys (SEGMENTATION_DEFAULTS), the rest lives in "parameters" as in buddy <= 0.5
    seg_file = _fill(cfg.get("segmentation"), SEGMENTATION_FILE_DEFAULTS, "segmentation")
    omni = {k: seg_file[k] for k in SEGMENTATION_DEFAULTS}
    for which in ("cells", "nuclei"):
        overrides = {k: v for k, v in (omni[which] or {}).items() if not _is_comment(k)}
        unknown = sorted(set(overrides) - set(OMNIPOSE_KEYS))
        if unknown:
            raise ValueError(f'config "segmentation/{which}": unknown Omnipose setting(s) {unknown} (known: {OMNIPOSE_KEYS})')
        omni[which] = overrides
    if omni["use_preset"] not in (True, False):
        raise ValueError(f'config "segmentation": "use_preset" must be true or false, not {omni["use_preset"]!r}')
    if omni["use_preset"]:
        presets = {"cells": OMNIPOSE_PRESETS[flat["organism"]], "nuclei": OMNIPOSE_NUCLEUS_PRESET}
        for which in ("cells", "nuclei"):
            differ = sorted(k for k, v in omni[which].items() if v != presets[which].get(k))
            if differ:
                log.warning(f'config "segmentation": use_preset is true, so the "{which}" values {differ} are ignored '
                            f'(the {flat["organism"] if which == "cells" else "nucleus"} preset is used); set use_preset false to apply them')
    cfg["segmentation"] = omni
    cfg["channels"] = {name: ch for name, ch in cfg["channels"].items() if not _is_comment(name)}
    for name, ch in cfg["channels"].items():
        ch = _fill(ch, CHANNEL_DEFAULTS, f"channels/{name}")
        if ch["filter"] is None:
            raise ValueError(f'config "channels/{name}": "filter" (the channel name in the microscope file) is missing')
        if ch["mrna"] is None:
            ch["mrna"] = name
        cfg["channels"][name] = ch
    cfg["bud_classification"] = _fill(cfg.get("bud_classification"), BUD_CLASSIFICATION_DEFAULTS, "bud_classification")
    turn = cfg["bud_classification"]["dic_rotation"]
    if turn is not None and turn not in (0, 1, 2, 3):
        raise ValueError(f'config "bud_classification": "dic_rotation" must be null or 0, 1, 2 or 3 (x 90 degrees), not {turn!r}')
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
    for key in ("inputdir", "outputdir"):           # relative folders are relative to the config file, not to the cwd
        if isinstance(cfg.get(key), str) and cfg[key] and not Path(cfg[key]).is_absolute():
            cfg[key] = str((cfg_file.resolve().parent / cfg[key]).resolve())
    return validate(cfg)


# ---- writing ------------------------------------------------------------------------------------------------
def file_layout(cfg):
    """
    A validated (or new-layout) config dict in the layout of the file, blocks in BLOCK_ORDER: the flat "parameters"
    distributed over "spots", "hardware", "tiff_input" and "segmentation". Comments ("_..." keys) are kept.
    """
    comments = {None: {k: v for k, v in cfg.items() if _is_comment(k)}}          # kept: the "_about" lines of the example
    for block, value in cfg.items():
        if isinstance(value, dict) and not _is_comment(block):
            comments[block] = {k: v for k, v in value.items() if _is_comment(k)}
    cfg = validate(json.loads(json.dumps(cfg, default=str))) if "parameters" not in cfg or "spots" not in cfg else cfg
    p = cfg["parameters"]
    seg = {"organism": p["organism"], **{k: cfg["segmentation"][k] for k in ("cell_model", "nucleus_model", "tile")},
           "min_size": p["min_size"], "nuclei_min_size": p["nuclei_min_size"],
           **{k: cfg["segmentation"][k] for k in ("use_preset", "cells", "nuclei")},
           **{k: p[k] for k in ("min_cell_area", "expand_px", "remove_edge_cells")}}
    blocks = {"channels": cfg["channels"], "tiff_input": {k: p[k] for k in TIFF_INPUT_KEYS}, "segmentation": seg,
              "spots": {k: p[k] for k in SPOT_KEYS}, "hardware": {k: p[k] for k in HARDWARE_KEYS}}
    out = dict(comments[None])
    for key in BLOCK_ORDER:
        if key in blocks:
            out[key] = blocks[key]
        elif key == "cell_flags" and cfg.get(key, CELL_FLAG_DEFAULTS) == CELL_FLAG_DEFAULTS:
            continue                                 # not meant to be tuned: written only when changed
        elif key in cfg:
            out[key] = cfg[key]
        if isinstance(out.get(key), dict) and comments.get(key):
            out[key] = {**comments[key], **{k: v for k, v in out[key].items() if not _is_comment(k)}}
    return out


def to_json(obj, level=0, width=110):
    """
    JSON for people: 2-space indent, lists of numbers / strings and short dicts on one line, a blank line between
    the top-level blocks. (json.dumps puts every list element on its own line.)
    """
    pad = "  " * level
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = [f"{pad}  {json.dumps(k, ensure_ascii=False)}: {to_json(v, level + 1, width)}" for k, v in obj.items()]
        inline = "{" + ", ".join(i.strip() for i in items) + "}"
        if level > 0 and "\n" not in inline and len(inline) + len(pad) <= width:
            return inline
        sep = ",\n\n" if level == 0 else ",\n"
        if level == 0:                       # scalars at the top (folders, policy) stay together, blocks get a blank line
            lines, prev_block = [], False
            for key, item in zip(obj, items):
                block = isinstance(obj[key], (dict, list))
                lines.append(("\n" if lines and (block or prev_block) else "") + item)
                prev_block = block
            return "{\n" + ",\n".join(lines) + "\n}"
        return "{\n" + sep.join(items) + "\n" + pad + "}"
    if isinstance(obj, list):
        if all(not isinstance(x, (dict, list)) for x in obj):
            inline = "[" + ", ".join(json.dumps(x, ensure_ascii=False) for x in obj) + "]"
            if len(inline) + len(pad) <= width:
                return inline
        return "[\n" + ",\n".join(f"{pad}  {to_json(x, level + 1, width)}" for x in obj) + "\n" + pad + "]"
    return json.dumps(obj, ensure_ascii=False)


def write_config(cfg, path):
    """Write a config dict (validated, or in the file layout) to path in the file layout, readably formatted."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(to_json(file_layout(cfg)) + "\n")
    return path


def _shown(preset):
    """The Omnipose settings of a preset shown in the example (whole numbers as ints: 0 instead of 0.0)."""
    return {k: int(v) if float(v).is_integer() else v for k, v in ((k, preset[k]) for k in EXAMPLE_OMNIPOSE_KEYS)}


def example_config():
    """
    config_files/config_example.json: every setting with its default (the values of the paper's runs), one block per
    step, each with a one-line "_about" comment (the full explanation is TUTORIAL.md, section 5). Also the config the
    notebooks start from (the example set's channels).
    """
    return {
        "_about": "buddy config: one block per step, every setting at its default (explained in TUTORIAL.md, section 5)",
        "_values": "true / false switch a setting on or off; null = not set, buddy decides (threshold null = automatic, "
                   "dic_rotation null = measured per image, ash_channel null in cell_cycle = the ash_channel of bud_classification)",
        "inputdir": "../../test yourself/small example set/YET572 BR4",         # relative to this file's folder
        "outputdir": "../../test yourself/small example set/YET572 BR4/output",
        "if_already_processed": "ask",
        "channels": {
            "_about": "step 1 (read): filter = channel name in the microscope file, threshold null = automatic, tiff_channel for TIFF stacks",
            "ASH1CLB2": {"filter": "CY5", "mrna": "ASH1CLB2", "threshold": 50, "colormap": "magenta"},
            "SRL1": {"filter": "CY3.5 NAR", "mrna": "SRL1", "threshold": 50, "colormap": "gold"},
            "SUN4": {"filter": "CY3", "mrna": "SUN4", "threshold": 50, "colormap": "lime"},
            "DAPI": {"filter": "DAPI", "mrna": "DAPI", "colormap": "blue"},
        },
        "spots": {
            "_about": "step 3 (spots): scale and spot_radius (z, y, x) in nm, sigma (z, y, x) in px - required; patch_size for the "
                      "in-focus planes; decomposition of dense regions (Big-FISH alpha, beta, gamma), the same for every channel",
            "scale": [200, 65, 65], "spot_radius": [880, 120, 120], "sigma": [0.75, 2.3, 2.3],
            "patch_size": PARAMETER_DEFAULTS["patch_size"], "decomposition": dict(DECOMPOSITION_DEFAULTS),
        },
        "segmentation": {
            "_about": "step 2 (segment): Omnipose and the mask clean-up. use_preset true = the organism's Omnipose settings "
                      "(cells / nuclei show them); to tune, set use_preset false and edit cells / nuclei",
            **SEGMENTATION_FILE_DEFAULTS, "use_preset": True,
            "cells": _shown(OMNIPOSE_PRESETS[PARAMETER_DEFAULTS["organism"]]),
            "nuclei": _shown(OMNIPOSE_NUCLEUS_PRESET),
        },
        "bud_classification": {
            "_about": "step 3, bud model: DIC_DAPI_only (any strain) or with_ASH1CLB2 (not for she2/she3); enabled: false = annotate buds by hand",
            **BUD_CLASSIFICATION_DEFAULTS, "enabled": True,
        },
        "cell_cycle": {
            "_about": "cell-cycle phase per cell: G1 (unbudded, <= g1_max_ash ASH1/CLB2), S / G2 / M by bud area (um2)",
            **CELL_CYCLE_DEFAULTS,
        },
        "hardware": {
            "_about": "use_gpu: Omnipose + bud model; gpu_spot_detection: the spot steps; without a GPU all runs on the CPU",
            **HARDWARE_DEFAULTS,
        },
        "tiff_input": {
            "_about": "step 1, TIFF stacks only: the DIC channel inside the stack and the axis order when the file does not say",
            **TIFF_INPUT_DEFAULTS,
        },
        "experiments": [
            {"strain": "YET918", "condition": "ΔSUN4"}, {"strain": "YET563", "condition": "ΔSHE2"},
            {"strain": "YET572", "condition": "ΔSHE3"}, {"strain": "YET915", "condition": "WT"},
            {"strain": "YET916", "condition": "ΔSSD1"}, {"strain": "YET919", "condition": "ΔMPT5"},
            {"strain": "YET921", "condition": "ΔSEC72"},
        ],
    }
