"""bud_training.training_rows: which cells are trained on, per model."""
import pandas as pd

from buddy.buds import bud_training as bt


def table():
    names = ["240909__YET563BR3__output__YET563_01", "220601__yET572BR1__output__yET572-01",
             "240909__YET915BR5__output__YET915_01", "250820__YET5630BR1__output__x"]
    rows = [dict(image=n, kind=k, edge=e, late_m=False) for n in names
            for k, e in [("budded", False), ("unbudded", False), ("fragment", False), ("bud", True)]]
    return pd.DataFrame(rows)


def test_default_strains_per_model():
    t = table()
    dic = bt.training_rows(t, "DIC_DAPI_only")
    ash = bt.training_rows(t, "with_ASH1CLB2")
    assert set(dic.kind) == {"budded", "unbudded"} and len(dic) == 8           # no fragments, no edge cells
    assert sorted(ash.image.unique()) == ["240909__YET915BR5__output__YET915_01", "250820__YET5630BR1__output__x"]
    assert len(bt.training_rows(t, "with_ASH1CLB2", exclude_strains=[])) == 8   # explicit list overrides the default
