"""Spot counting per cell / mother / bud on a small made-up image: every number is known in advance."""
import numpy as np
import pytest

from buddy.quantification.spot_assignment import spot_assignment


@pytest.fixture
def scene():
    """two cells: cell 1 (mother 40x40 at (10,10) + bud 10x20 at (10,50)), cell 2 (30x30 at (60,60)); one nucleus each"""
    masks = np.zeros((100, 100), np.int64)
    masks[10:50, 10:50] = 1
    masks[10:20, 50:70] = 1                      # the bud, same label as its mother
    masks[60:90, 60:90] = 2
    bud = np.zeros_like(masks); bud[10:20, 50:70] = 1
    nuclei = np.zeros_like(masks); nuclei[25:35, 25:35] = 1; nuclei[70:80, 70:80] = 2
    dapi = np.zeros((100, 100), np.float32); dapi[nuclei > 0] = 100.0
    # spots: (z, y, x); 3 in the mother (one in the nucleus), 2 in the bud, 1 in cell 2, 1 in the background
    spots = np.array([[5, 12, 12], [5, 30, 30], [5, 45, 45], [5, 15, 55], [5, 15, 65], [5, 65, 65], [5, 2, 2]])
    # dense regions: (z, y, x, nb_rna, ...); one in the mother's nucleus with 4 RNAs, one in cell 2 outside the nucleus
    dense = np.array([[5, 28, 28, 4, 0], [5, 62, 62, 3, 0]])
    return masks, bud, nuclei, dapi, spots, dense


def test_counts_per_cell_mother_and_bud(scene):
    masks, bud, nuclei, dapi, spots, dense = scene
    t = spot_assignment(masks, bud, nuclei, dapi, spots, dense).set_index("label")
    # cell 1: 5 single spots + dense region of 4 -> total 5 + 4 - 1 = 8 (the dense region replaces one 'spot' count)
    assert t.loc[1, "spots"] == 5 and t.loc[1, "dense_regions"] == 1 and t.loc[1, "decomposed_RNAs"] == 4
    assert t.loc[1, "total_RNAs"] == 8
    assert t.loc[1, "tx"] == 1 and t.loc[1, "nascent_RNAs"] == 4          # dense region in the nucleus = transcription site
    assert t.loc[1, "spots_bud"] == 2 and t.loc[1, "total_RNAs_bud"] == 2
    assert t.loc[1, "spots_mother"] == 3 and t.loc[1, "total_RNAs_mother"] == 6
    assert t.loc[1, "area_bud"] == 200
    # cell 2: 1 spot + a dense region of 3 outside the nucleus
    assert t.loc[2, "spots"] == 1 and t.loc[2, "total_RNAs"] == 3 and t.loc[2, "tx"] == 0
    assert t.loc[2, "spots_bud"] == 0 and t.loc[2, "area_bud"] == 0
    # one nucleus each, the mother's nucleus measured without the bud
    assert t.loc[1, "nuclei"] == 1 and t.loc[2, "nuclei"] == 1
    assert t.loc[1, "mother_nuclei_area"] == 100 and t.loc[1, "mother_nuclei_mean_intensity"] == 100.0
    # the background spot is not counted anywhere
    assert t.spots.sum() == 6


def test_without_bud_mask(scene):
    masks, bud, nuclei, dapi, spots, dense = scene
    t = spot_assignment(masks, None, nuclei, dapi, spots, dense).set_index("label")
    assert "spots_bud" not in t.columns
    assert t.loc[1, "total_RNAs"] == 8 and t.loc[2, "total_RNAs"] == 3


def test_no_spots_at_all(scene):
    masks, bud, nuclei, dapi, spots, dense = scene
    t = spot_assignment(masks, bud, nuclei, dapi, np.zeros((0, 3)), np.zeros((0, 5))).set_index("label")
    assert list(t.index) == [1, 2] and (t.total_RNAs == 0).all()
