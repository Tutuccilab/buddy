"""The napari flags for unbudded cells with high ASH1/CLB2 (buddy/cell_flags.py) on a made-up table + mask."""
import numpy as np
import pandas as pd

from buddy.quantification.cell_flags import HIGH_ASH, LATE_M, compute_flags

SETTINGS = dict(fold=8, nucleus_area=0.5, nucleus_intensity=0.7, max_gap_px=10)


def scene():
    """
    12 unbudded cells in a 300 x 300 image, 20 x 20 px each, well apart; cells 1 and 2 are neighbours (gap 4 px).
    Typical ASH1/CLB2 = 4 molecules; every cell has a nucleus of area 100 and intensity 50 unless changed.
    """
    masks = np.zeros((300, 300), np.int64)
    rows = []
    positions = [(40, 40), (40, 64)] + [(40 + 60 * (i // 4), 100 + 50 * (i % 4)) for i in range(10)]
    for label, (y, x) in enumerate(positions, start=1):
        masks[y:y + 20, x:x + 20] = label
        rows.append({"image_cell_id": label, "bbox-0": y, "bbox-1": x, "bbox-2": y + 20, "bbox-3": x + 20, "area_bud": 0,
                     "total_RNAs": 4, "mother_nuclei_area": 100, "mother_nuclei_mean_intensity": 50})
    return masks, pd.DataFrame(rows)


def test_nothing_flagged_in_a_quiet_field():
    masks, t = scene()
    assert len(compute_flags(t, masks, SETTINGS)) == 0


def test_late_m_pair():
    masks, t = scene()
    t.loc[t.image_cell_id == 1, "total_RNAs"] = 60              # 15x the typical cell, next to cell 2
    f = compute_flags(t, masks, SETTINGS)
    assert len(f) == 1 and f.flag.iloc[0] == LATE_M and f.label.iloc[0] == 1
    assert "cells 1 and 2" in f.reason.iloc[0]
    assert (f.y0.iloc[0], f.x0.iloc[0], f.y1.iloc[0], f.x1.iloc[0]) == (40, 40, 60, 84)   # the box spans both cells


def test_high_single_when_the_neighbour_has_no_nucleus():
    masks, t = scene()
    t.loc[t.image_cell_id == 1, "total_RNAs"] = 60
    t.loc[t.image_cell_id == 2, "mother_nuclei_area"] = 10      # no full nucleus of its own -> not a late-M pair
    f = compute_flags(t, masks, SETTINGS)
    assert len(f) == 1 and f.flag.iloc[0] == HIGH_ASH and f.label.iloc[0] == 1
    assert "DAPI signal" in f.reason.iloc[0]


def test_high_single_without_neighbour():
    masks, t = scene()
    t.loc[t.image_cell_id == 7, "total_RNAs"] = 40              # 10x, no neighbour within 10 px
    f = compute_flags(t, masks, SETTINGS)
    assert f.flag.tolist() == [HIGH_ASH] and f.label.tolist() == [7]
    assert f.ash_rel.iloc[0] == 10.0


def test_budded_and_edge_cells_are_ignored():
    masks, t = scene()
    t.loc[t.image_cell_id == 7, ["total_RNAs", "area_bud"]] = [40, 50]      # budded: not a candidate
    t.loc[t.image_cell_id == 8, ["total_RNAs", "bbox-0"]] = [40, 0]          # touches the image edge
    assert len(compute_flags(t, masks, SETTINGS)) == 0


def test_each_cell_flagged_once():
    masks, t = scene()
    t.loc[t.image_cell_id.isin([1, 2]), "total_RNAs"] = 60       # both high and neighbours: one pair, not two flags
    f = compute_flags(t, masks, SETTINGS)
    assert len(f) == 1 and f.flag.iloc[0] == LATE_M
