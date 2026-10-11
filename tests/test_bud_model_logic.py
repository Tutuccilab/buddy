"""
The rules that turn the bud model's per-cell answers into mother-bud pairs (bud_model.py), without the network:
claims, competition between cells, snap, the output masks and the flags. Pure numpy, fast.
"""
import numpy as np
import pandas as pd

from buddy.bud_model import BACKGROUND, BUD, MOTHER, BudModel, mean_on


def rect(masks, label, y0, x0, h, w):
    masks[y0:y0 + h, x0:x0 + w] = label


def cell(label, p_budded, bud_pixels=(), prob=0.9, shape=(200, 200), p_bud=0.0):
    """a per-cell model answer: class probabilities + a bud map that is `prob` on `bud_pixels` (flat indices)"""
    px = np.array(sorted(bud_pixels), np.int64)
    return dict(label=label, p_unbudded=1 - p_budded - p_bud, p_budded=p_budded, p_bud=p_bud,
                pixels=px, prob=np.full(len(px), prob, np.float32))


def flat(mask):
    return np.flatnonzero(mask.ravel())


def model(cut_off=0.5, snap=True):
    m = BudModel.__new__(BudModel)
    m.cut_off, m.snap, m.name = cut_off, snap, "DIC_DAPI_only"
    return m


def scene():
    """mother (1) with a separate small Omnipose object (2) next to it, an unbudded neighbour (3) and a far cell (4)"""
    masks = np.zeros((200, 200), np.int64)
    rect(masks, 1, 50, 50, 40, 40)
    rect(masks, 2, 50, 90, 20, 20)        # candidate bud, touching the mother
    rect(masks, 3, 100, 90, 40, 40)
    rect(masks, 4, 150, 10, 30, 30)
    return masks


def test_mother_claims_its_bud_and_masks_are_merged():
    masks = scene()
    cells = [cell(1, 0.9, flat(masks == 2)), cell(2, 0.05, p_bud=0.95), cell(3, 0.1), cell(4, 0.1)]
    m = model()
    accepted = m.assign(cells, masks)
    assert list(accepted) == [1] and np.array_equal(accepted[1]["claim"], flat(masks == 2))
    merged, bud_mask = m.build_masks(masks, accepted, m._pixels_of)
    assert (merged[masks == 2] == 1).all() and (merged[masks == 1] == 1).all()        # the bud takes the mother's label
    assert (bud_mask[masks == 2] == BUD).all() and (bud_mask[masks == 1] == MOTHER).all()
    assert (bud_mask[masks == 0] == BACKGROUND).all() and (bud_mask[masks == 3] == MOTHER).all()


def test_claim_below_cut_off_is_dropped_and_flagged():
    masks = scene()
    cells = [cell(1, 0.5, flat(masks == 2), prob=0.8), cell(2, 0.1), cell(3, 0.1), cell(4, 0.1)]   # strength 0.4
    m = model(cut_off=0.45)
    accepted = m.assign(cells, masks)
    assert accepted == {}
    flags = m.find_flags(cells, accepted, masks)
    assert flags.flag.tolist() == ["possible missed bud"] and flags.label.tolist() == [1]


def test_stronger_claim_wins_the_same_bud():
    masks = scene()
    bud = flat(masks == 2)
    cells = [cell(1, 0.6, bud, prob=0.9), cell(3, 0.9, bud, prob=0.95), cell(2, 0.1), cell(4, 0.1)]
    m = model()
    accepted = m.assign(cells, masks)
    assert list(accepted) == [3]                                                       # 0.855 beats 0.54
    flags = m.find_flags(cells, accepted, masks)
    assert "close call" not in flags.flag.tolist()
    # nearly equal claims -> close call flagged
    cells = [cell(1, 0.9, bud, prob=0.9), cell(3, 0.85, bud, prob=0.9), cell(2, 0.1), cell(4, 0.1)]
    accepted = m.assign(cells, masks)
    assert list(accepted) == [1]
    assert m.find_flags(cells, accepted, masks).flag.tolist() == ["close call"]


def test_a_cell_inside_an_accepted_bud_cannot_be_a_mother():
    masks = scene()
    cells = [cell(1, 0.95, flat(masks == 2)), cell(2, 0.9, flat(masks == 3)[:400]), cell(3, 0.1), cell(4, 0.1)]
    accepted = model().assign(cells, masks)
    assert list(accepted) == [1]                   # cell 2 is a bud of 1, so its own claim on cell 3 is dropped


def test_bud_without_mother_is_flagged():
    masks = scene()
    cells = [cell(1, 0.1), cell(2, 0.02, p_bud=0.95), cell(3, 0.1), cell(4, 0.1)]
    m = model()
    flags = m.find_flags(cells, m.assign(cells, masks), masks)
    assert flags.flag.tolist() == ["bud without mother"] and flags.label.tolist() == [2]


def test_snap_takes_the_whole_omnipose_object():
    masks = scene()
    part = flat(masks == 2)[:150]                  # the model outlines only part of the separate object 2
    cells = [cell(1, 0.9, part), cell(2, 0.1), cell(3, 0.1), cell(4, 0.1)]
    m = model(snap=True)
    accepted = m.assign(cells, masks)
    assert len(accepted[1]["claim"]) == 150
    m.snap_claims(accepted, masks, m._pixels_of)
    assert np.array_equal(accepted[1]["claim"], flat(masks == 2)) and accepted[1]["snapped"]
    # not snapped when the claim lies mostly on background / the mother, or when the object is an accepted mother
    cells = [cell(1, 0.9, np.concatenate([flat(masks == 2)[:50], flat(masks == 1)[:200]])), cell(2, 0.1), cell(3, 0.1), cell(4, 0.1)]
    accepted = m.assign(cells, masks); m.snap_claims(accepted, masks, m._pixels_of)
    assert not accepted[1].get("snapped", False)


def test_classify_output_tables(monkeypatch):
    """classify(): the whole chain with a fake network answer"""
    masks = scene()
    m = model()
    m.uses_ash = False
    m.last_rotation = 0
    answers = [cell(1, 0.9, flat(masks == 2)), cell(2, 0.05, p_bud=0.95), cell(3, 0.1), cell(4, 0.1)]
    monkeypatch.setattr(m, "predict_cells", lambda *a, **k: answers)
    result = m.classify(np.zeros((200, 200)), np.zeros((200, 200)), masks)
    assert result["pairs"].to_dict("records") == [dict(mother_label=1, bud_px=400, strength=0.81, close_call=False, snapped=False)]
    assert sorted(result["cells"].label) == [1, 2, 3, 4]
    assert isinstance(result["flags"], pd.DataFrame) and len(result["flags"]) == 0
    assert (result["cell_masks"][masks == 2] == 1).all()


def test_mean_on():
    c = cell(1, 0.5, [10, 11, 12], prob=0.8)
    assert abs(mean_on(c, np.array([10, 11, 12])) - 0.8) < 1e-6
    assert abs(mean_on(c, np.array([10, 11, 12, 13])) - 0.6) < 1e-6          # a pixel outside the map counts as 0
    assert mean_on(c, np.array([], np.int64)) == 0.0
