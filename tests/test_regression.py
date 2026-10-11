"""
Regression test: does the pipeline still give the same results on the test field as the saved reference?
(tests/make_reference.py renews the reference - on purpose only.)

Fast tests start from the saved Omnipose masks (steps 3 and 5). The slow ones (Omnipose, bud model) run with
`pytest -m slow` or without a marker filter.
"""
import json
import shutil

import numpy as np
import pandas as pd
import pytest
import tifffile
from conftest import FIELD

CHANNELS = ("ASH1CLB2", "SRL1", "SUN4")
ID = "image_cell_id"                 # the cell label column of the {channel}.csv files
COUNT_COLUMNS = [ID, "area", "spots", "dense_regions", "decomposed_RNAs", "total_RNAs", "nuclei", "tx", "nascent_RNAs"]
BUD_COLUMNS = ["area_bud", "spots_bud", "total_RNAs_bud", "spots_mother", "total_RNAs_mother"]


def _reference(part):
    ref = FIELD / "reference" / part
    if not ref.is_dir():
        pytest.skip(f"no reference for {part}: run tests/make_reference.py")
    return ref


def _same_counts(folder, ref, with_bud):
    for ch in CHANNELS:
        got, want = pd.read_csv(folder / f"{ch}.csv"), pd.read_csv(ref / f"{ch}.csv")
        cols = COUNT_COLUMNS + (BUD_COLUMNS if with_bud else [])
        cols = [c for c in cols if c in want.columns]
        got, want = got.sort_values(ID).reset_index(drop=True), want.sort_values(ID).reset_index(drop=True)
        assert len(got) == len(want), f"{ch}: {len(got)} cells, reference {len(want)}"
        diff = (got[cols].round(6) != want[cols].round(6)).any(axis=1)
        assert not diff.any(), f"{ch}: counts differ for cells {got[ID][diff].tolist()[:10]}\n{got[cols][diff].head()}\nreference:\n{want[cols][diff].head()}"


def _same_spots(folder, ref):
    for ch in CHANNELS:
        got, want = np.load(folder / f"{ch}_spots.npy"), np.load(ref / f"{ch}_spots.npy")
        assert got.shape == want.shape, f"{ch}: {len(got)} spots, reference {len(want)}"
        got, want = got[np.lexsort(got.T[::-1])], want[np.lexsort(want.T[::-1])]
        assert np.array_equal(got, want), f"{ch}: spot coordinates differ"
        got, want = np.load(folder / f"{ch}_dense_regions.npy"), np.load(ref / f"{ch}_dense_regions.npy")
        assert got.shape == want.shape, f"{ch}: {len(got)} dense regions, reference {len(want)}"
        assert np.allclose(got[:, :4], want[:, :4]), f"{ch}: dense regions differ"


def _run_step3(work_copy):
    from buddy import pipeline
    from buddy.experiment import Experiment
    cfg_file, folder = work_copy
    shutil.rmtree(folder / "corrected", ignore_errors=True)           # a clean run builds corrected/ itself
    exp = Experiment.from_cfg_file(cfg_file)
    exp.use_gpu = False; exp.parameters["use_gpu"] = False; exp.parameters["gpu_spot_detection"] = False
    exp.bud_classification["enabled"] = False
    pipeline.spots_image(folder / "img.json", exp)
    pipeline.buds_and_counts(folder / "img.json", exp)
    return exp, folder


def test_spots_alignment_and_counts(work_copy):
    """step 3 from the saved Omnipose masks: spots, dense regions, alignment shift and per-cell counts"""
    ref = _reference("step3")
    exp, folder = _run_step3(work_copy)
    _same_spots(folder, ref)
    got = json.load(open(folder / "img.json"))["results"]["alignment"]
    want = json.load(open(ref / "img.json"))["results"]["alignment"]
    assert got["mask_shift"] == want["mask_shift"], f"alignment shift {got['mask_shift']} != {want['mask_shift']}"
    assert np.array_equal(tifffile.imread(folder / "corrected" / "DIC_corrected_masks.tif"),
                          tifffile.imread(ref / "corrected" / "DIC_corrected_masks.tif"))
    _same_counts(folder, ref, with_bud=False)
    if (ref / "cell_flags.csv").is_file():
        got, want = pd.read_csv(folder / "cell_flags.csv"), pd.read_csv(ref / "cell_flags.csv")
        assert got[["label", "flag"]].to_dict("records") == want[["label", "flag"]].to_dict("records")


def test_counts_with_manual_masks(work_copy):
    """step 5: counting with the hand-corrected masks (mother / bud)"""
    ref = _reference("step5")
    from buddy import pipeline
    exp, folder = _run_step3(work_copy)
    for p in (folder / "manual").glob("*.tif"):
        shutil.copy(p, folder / "corrected" / p.name)
    pipeline.reassign_image(folder / "img.json", exp)
    _same_counts(folder, ref, with_bud=True)


@pytest.mark.slow
def test_omnipose_segmentation(work_copy):
    """step 2: Omnipose cell + nucleus masks (same cells; outlines may differ slightly between machines)"""
    ref = _reference("slow/omnipose")
    from buddy import pipeline
    from buddy.experiment import Experiment
    cfg_file, folder = work_copy
    exp = Experiment.from_cfg_file(cfg_file); exp.use_gpu = False
    exp.init_omnipose(use_GPU=False)
    pipeline.segment_image(folder / "img.json", exp)
    for name in ("DIC_masks_pp.tif", "DAPI_masks.tif"):
        got, want = tifffile.imread(folder / name).astype(np.int64), tifffile.imread(ref / name).astype(np.int64)
        n_got, n_want = len(np.unique(got)) - 1, len(np.unique(want)) - 1
        assert abs(n_got - n_want) <= max(1, 0.05 * n_want), f"{name}: {n_got} objects, reference {n_want}"
        ious = []
        for label in np.unique(want)[1:]:
            w = want == label
            match = np.bincount(got[w])[1:]
            if match.size and match.max():
                g = got == (match.argmax() + 1)
                ious.append((w & g).sum() / (w | g).sum())
            else:
                ious.append(0.0)
        assert np.mean(ious) >= 0.95, f"{name}: mean IoU with the reference {np.mean(ious):.3f}"


@pytest.mark.slow
def test_bud_model(work_copy):
    """the shipped DIC_DAPI_only model: same mother-bud pairs as the reference"""
    ref = _reference("slow/bud_model")
    from buddy import pipeline
    exp, folder = _run_step3(work_copy)
    exp.bud_classification.update(enabled=True, model="DIC_DAPI_only")
    exp.init_bud_model()
    pipeline.buds_and_counts(folder / "img.json", exp)
    got, want = pd.read_csv(folder / "bud_pairs.csv"), pd.read_csv(ref / "bud_pairs.csv")
    assert sorted(got.mother_label) == sorted(want.mother_label), f"pairs {sorted(got.mother_label)} != {sorted(want.mother_label)}"
    merged = got.merge(want, on="mother_label", suffixes=("", "_ref"))
    assert (abs(merged.bud_px - merged.bud_px_ref) <= 0.02 * merged.bud_px_ref + 5).all(), "bud sizes differ"
    assert np.allclose(merged.strength, merged.strength_ref, atol=0.02), "claim strengths differ"
