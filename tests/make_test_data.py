"""
Make the test field (tests/data/field_A) from one processed image folder: a 512 x 512 px crop of every layer the
pipeline needs from step 2 on, the hand-corrected masks (manual/) and a config file. Run once by the maintainer, then zip the folder and attach it
to the GitHub release `test-data-v1` (see tests/conftest.py); tests/make_reference.py adds the reference outputs.

    python tests/make_test_data.py "<image folder>" <y0> <x0> [--size 512]

The folder is an output folder of the bulk-processing notebook (with ASH1CLB2.tif ..., DIC.tif, img.json,
miscellaneous/ or the masks next to img.json). The crop is taken from the fluorescence frame; the DIC image and
the Omnipose masks are cropped at the same place (they are aligned later by the pipeline itself).
"""
import argparse
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import tifffile

OUT = Path(__file__).resolve().parent / "data" / "field_A"
CHANNELS = {"ASH1CLB2": "CY5", "SRL1": "CY3.5 NAR", "SUN4": "CY3"}        # channel -> filter name, as in the config
CONFIG = {
    "inputdir": "", "outputdir": "",
    "parameters": {"scale": [200, 65, 65], "spot_radius": [880, 120, 120], "sigma": [0.75, 2.3, 2.3],
                   "patch_size": [200, 200], "organism": "S.cerevisiae", "min_size": 10, "nuclei_min_size": 10,
                   "min_cell_area": 20, "expand_px": 0, "remove_edge_cells": True, "use_gpu": False,
                   "gpu_spot_detection": False},
    "if_already_processed": "overwrite",
    "bud_classification": {"enabled": False, "model": "DIC_DAPI_only", "ash_channel": "ASH1CLB2",
                           "overwrite_manual_corrections": False},
    "cell_flags": {"enabled": True},
    "experiments": [{"strain": "YET916", "condition": "ΔSSD1"}],
    "channels": {**{ch: {"filter": f, "mrna": ch, "threshold": 50, "colormap": "magenta"} for ch, f in CHANNELS.items()},
                 "DAPI": {"filter": "DAPI", "mrna": "DAPI", "threshold": None, "colormap": "blue"}},
}


def find(folder, name):
    for p in (folder / name, folder / "miscellaneous" / name):
        if p.is_file():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder"); ap.add_argument("y0", type=int); ap.add_argument("x0", type=int)
    ap.add_argument("--size", type=int, default=512)
    a = ap.parse_args()
    src, S = Path(a.folder), a.size
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "manual").mkdir(parents=True)          # the hand-corrected masks of that image (used by the step-5 test)
    crop = lambda arr: arr[..., a.y0:a.y0 + S, a.x0:a.x0 + S]
    for ch in list(CHANNELS) + ["DAPI"]:
        tifffile.imwrite(OUT / f"{ch}.tif", crop(tifffile.imread(src / f"{ch}.tif")), compression="zlib")
    tifffile.imwrite(OUT / "DIC.tif", crop(tifffile.imread(src / "DIC.tif")), compression="zlib")
    for name in ("DIC_masks.tif", "DIC_masks_pp.tif", "DAPI_masks.tif"):
        p = find(src, name)
        if p is None:
            raise FileNotFoundError(f"{name} not found in {src} (or its miscellaneous/ folder)")
        tifffile.imwrite(OUT / name, crop(tifffile.imread(p)).astype(np.uint16), compression="zlib")
    for name in ("DIC_corrected_masks.tif", "DAPI_corrected_masks.tif", "bud_corrected_masks.tif"):
        p = src / "corrected" / name
        if p.is_file():
            arr = crop(tifffile.imread(p))
            tifffile.imwrite(OUT / "manual" / name, arr.astype(np.uint8 if name.startswith("bud") else np.uint16), compression="zlib")
    meta = json.load(open(src / "img.json"))
    json.dump({"vsi_file": meta["vsi_file"], "cell_file": meta["cell_file"], "stem": "field_A", "savepath": str(OUT),
               "parameters": CONFIG["parameters"], "source": {"folder": str(src), "y0": a.y0, "x0": a.x0, "size": S},
               "results": {}, "time": {}}, open(OUT / "img.json", "w"), indent=1)
    json.dump(CONFIG, open(OUT / "config.json", "w"), indent=1, ensure_ascii=False)
    with zipfile.ZipFile(OUT.parent / "field_A.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob("*"):
            z.write(p, p.relative_to(OUT.parent))
    print(f"wrote {OUT} and {OUT.parent / 'field_A.zip'}")


if __name__ == "__main__":
    main()
