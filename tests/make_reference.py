"""
Make (or renew) the reference outputs of the regression test: tests/data/field_A/reference/.

    python tests/make_reference.py            steps 3 (spots, alignment, counting), 5 (counting with the manual masks)
    python tests/make_reference.py --slow     also step 2 (Omnipose) and the bud model -> reference/slow/

Renew the reference ON PURPOSE only: after a change that is meant to alter the results, once you have looked at
what changed (pytest tests/test_regression.py shows the differences). Then zip tests/data/field_A again for the
test-data release. Run it from the buddy environment; the GPU is not used (same numbers on every machine).
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import FIELD                                     # noqa: E402

from buddy import pipeline
from buddy.experiment import Experiment

REFERENCE = FIELD / "reference"
CHANNELS = ("ASH1CLB2", "SRL1", "SUN4")


def stage(name):
    print(f"\n==== {name} ====", flush=True)


def fresh_copy(root):
    """copy of the test field without reference/ and without corrected/, config pointing at it"""
    out = root / "output" / FIELD.name
    shutil.copytree(FIELD, out, ignore=shutil.ignore_patterns("reference", "corrected"))
    cfg = json.load(open(FIELD / "config.json"))
    cfg["inputdir"], cfg["outputdir"] = str(root / "input"), str(root / "output")
    cfg_file = root / "config.json"
    json.dump(cfg, open(cfg_file, "w"), indent=1)
    meta = json.load(open(out / "img.json")); meta["savepath"] = str(out)
    json.dump(meta, open(out / "img.json", "w"), indent=1)
    return cfg_file, out


def run_spots_and_counts(cfg_file, folder):
    """step 3 (spots + alignment + counting) on the copy; returns the Experiment"""
    exp = Experiment.from_cfg_file(cfg_file)
    exp.parameters["use_gpu"] = False; exp.parameters["gpu_spot_detection"] = False; exp.use_gpu = False
    exp.bud_classification["enabled"] = False
    json_file = folder / "img.json"
    pipeline.spots_image(json_file, exp)
    pipeline.buds_and_counts(json_file, exp)
    return exp


def run_reassign_with_manual_masks(exp, folder):
    """step 5 with the hand-corrected masks (manual/ -> corrected/)"""
    (folder / "corrected").mkdir(exist_ok=True)
    for p in (folder / "manual").glob("*.tif"):
        shutil.copy(p, folder / "corrected" / p.name)
    pipeline.reassign_image(folder / "img.json", exp)


def collect(folder, dest, names):
    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = folder / name
        if src.is_file():
            (dest / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dest / name)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--slow", action="store_true")
    a = ap.parse_args()
    import tempfile
    root = Path(tempfile.mkdtemp(prefix="buddy_ref_"))
    cfg_file, folder = fresh_copy(root)
    if REFERENCE.exists():
        shutil.rmtree(REFERENCE)

    stage("step 3: spots, alignment, counting")
    exp = run_spots_and_counts(cfg_file, folder)
    collect(folder, REFERENCE / "step3", [f"{ch}_spots.npy" for ch in CHANNELS] + [f"{ch}_dense_regions.npy" for ch in CHANNELS]
            + [f"{ch}.csv" for ch in CHANNELS] + ["aligned_DIC_masks.tif", "corrected/DIC_corrected_masks.tif",
                                                   "corrected/DAPI_corrected_masks.tif", "cell_flags.csv", "img.json"])

    if a.slow:
        # bud model on the step-3 result - BEFORE the manual masks are copied into corrected/, because classify_buds
        # never overwrites a hand-made bud_corrected_masks.tif (it would silently skip and write no CSVs)
        stage("bud model (DIC_DAPI_only)")
        exp.bud_classification.update(enabled=True, model="DIC_DAPI_only")
        exp.init_bud_model()
        pipeline.buds_and_counts(folder / "img.json", exp)
        bud_files = ["bud_pairs.csv", "bud_flags.csv", "bud_model_cells.csv",
                     "corrected/bud_corrected_masks.tif", "corrected/DIC_corrected_masks.tif"]
        missing = [n for n in bud_files if not (folder / n).is_file()]
        assert not missing, f"the bud model did not write {missing}"
        collect(folder, REFERENCE / "slow" / "bud_model", bud_files)

    stage("step 5: counting with the manual masks")
    run_reassign_with_manual_masks(exp, folder)
    collect(folder, REFERENCE / "step5", [f"{ch}.csv" for ch in CHANNELS] + ["cell_flags.csv"])

    if a.slow:
        stage("step 2: Omnipose (CPU)")
        cfg2, folder2 = fresh_copy(root / "slow")
        exp2 = Experiment.from_cfg_file(cfg2); exp2.use_gpu = False
        exp2.init_omnipose(use_GPU=False)
        pipeline.segment_image(folder2 / "img.json", exp2)
        collect(folder2, REFERENCE / "slow" / "omnipose", ["DIC_masks.tif", "DIC_masks_pp.tif", "DAPI_masks.tif"])
    json.dump({"made_with": "tests/make_reference.py", "slow": a.slow}, open(REFERENCE / "README.json", "w"), indent=1)
    shutil.rmtree(root, ignore_errors=True)
    print(f"reference written to {REFERENCE}")


if __name__ == "__main__":
    main()
