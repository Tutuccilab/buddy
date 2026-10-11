"""Alignment of the DIC frame (cell masks) to the fluorescence frame (spots): a known shift must be recovered."""
import numpy as np
import pytest

from buddy.spots.alignment import build_distance_map, estimate_alignment_shift, score_shift


def make_cells(n=30, size=600, seed=0):
    """random round cells and spots inside them; the spots are moved by (dy, dx) by the caller"""
    rng = np.random.default_rng(seed)
    masks = np.zeros((size, size), np.int64)
    yy, xx = np.ogrid[:size, :size]
    centres = rng.integers(60, size - 60, (n, 2))
    for i, (cy, cx) in enumerate(centres, start=1):
        masks[(yy - cy) ** 2 + (xx - cx) ** 2 < 20 ** 2] = i
    spots = []
    for cy, cx in centres:
        for _ in range(12):                   # spots up to the cell edge: a wrong shift pushes some out of the cell
            a, r = rng.uniform(0, 2 * np.pi), rng.uniform(5, 19)
            spots.append([5, cy + r * np.sin(a), cx + r * np.cos(a)])
    return masks, np.array(spots)


@pytest.mark.parametrize("shift", [(0, 0), (-10, 10), (23, -17), (-60, -45)])
def test_known_shift_is_recovered(shift):
    masks, spots = make_cells()
    dy, dx = shift
    moved = spots.copy(); moved[:, 1] += dy; moved[:, 2] += dx          # spots (fluorescence frame) sit `shift` away from the masks
    mask_shift, spot_shift = estimate_alignment_shift(moved, masks, n_iter=2000, max_shift=100, rng=np.random.default_rng(0))
    # mask_shift is what moves the masks onto the spots; the cost is flat inside a cell, so allow 2 px
    assert abs(mask_shift[0] - dy) <= 2 and abs(mask_shift[1] - dx) <= 2, f"found {mask_shift}, expected {shift}"


def test_distance_map_is_zero_inside_cells_and_capped_outside():
    masks, _ = make_cells(n=5)
    dist = build_distance_map(masks, cutoff=30)
    assert dist[masks > 0].max() == 0
    assert dist.max() <= 30
    assert dist[masks == 0].mean() > 0


def test_score_prefers_the_true_shift():
    masks, spots = make_cells(n=10)
    dist = build_distance_map(masks)
    assert score_shift(spots, dist, 0, 0) > score_shift(spots, dist, 15, 15)
    assert score_shift(spots, dist, 100000, 100000) == -np.inf                 # every spot outside the field


def test_same_result_every_run():
    masks, spots = make_cells()
    spots[:, 1] += 7; spots[:, 2] -= 3
    a = estimate_alignment_shift(spots, masks, n_iter=1000, rng=np.random.default_rng(0))
    b = estimate_alignment_shift(spots, masks, n_iter=1000, rng=np.random.default_rng(0))
    assert a == b
