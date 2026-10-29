"""
Make tests/data/vsi/vsi_reference.json for test_vsi_file.py: what buddy reads from the one real .vsi there (series
names, shape, a checksum per channel and of the DIC image). Run once, in an environment where the result is known to
be right, then zip tests/data/vsi as vsi.zip for the test-data-vsi release:

    python tests/make_vsi_reference.py                       # buddy's own Bio-Formats reader (buddy >= 0.5.2)
    BUDDY_VSI_READER=bioio python tests/make_vsi_reference.py   # the bioio reader of buddy <= 0.5.1, when installed

tests/data/vsi holds: <image>_CY5, CY3.5 NAR, CY3, DAPI.vsi, its _<image>..._ folder and <image>_DIC.tif, copied from
the example set (test yourself/small example set/YET572 BR4).
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from skimage import io

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from buddy.config import example_config  # noqa: E402
from buddy.reading.readers import read_vsi  # noqa: E402

VSI_DIR = Path(__file__).resolve().parent / "data" / "vsi"


def md5(array):
    return hashlib.md5(np.ascontiguousarray(array).tobytes()).hexdigest()


def main():
    vsi = sorted(VSI_DIR.glob("*.vsi"))
    assert len(vsi) == 1, f"expected one .vsi in {VSI_DIR}, found {vsi}"
    vsi = vsi[0]
    channels = {name: ch["filter"] for name, ch in example_config()["channels"].items() if not name.startswith("_")}
    filter2mrna = {f: name for name, f in channels.items()}
    if os.environ.get("BUDDY_VSI_READER") == "bioio":          # the reader of buddy <= 0.5.1 (needs bioio installed)
        import bioio_bioformats
        from bioio import BioImage
        image = BioImage(vsi, reader=bioio_bioformats.Reader)
        names, shapes = list(image.scenes), []
        for scene in names:
            image.set_scene(scene)
            shapes.append([int(n) for n in image.shape])            # T, C, Z, Y, X
    else:
        from buddy.reading.vsi import BioFormatsFile
        with BioFormatsFile(vsi) as f:
            names = f.series_names
            shapes = [list(f.shape(i)) for i in range(len(names))]
    stacks = read_vsi(vsi, filter2mrna)
    dic = sorted(VSI_DIR.glob("*_DIC.tif"))[0]
    ref = {"vsi": vsi.name, "dic": dic.name, "series_names": names, "shapes": shapes,
           "channels": {ch: {"shape": list(a.shape), "dtype": str(a.dtype), "md5": md5(a), "mean": float(a.mean())}
                        for ch, a in stacks.items()},
           "dic_md5": md5(io.imread(dic))}
    out = VSI_DIR / "vsi_reference.json"
    json.dump(ref, open(out, "w", encoding="utf-8"), indent=1)
    print(f"wrote {out}:", names, shapes, {ch: v["md5"][:8] for ch, v in ref["channels"].items()})


if __name__ == "__main__":
    main()
