"""
Flags for napari: cells worth a look because they have the signature of the end or the start of a division,
the two stages that are hard to see in a DIC image. Computed from buddy's own measurements, after the spots are
counted per cell (needs the ASH1/CLB2 channel); written to cell_flags.csv and shown in napari's
'check these buds' layer (hover a box for the reason). Nothing in the masks or counts is changed.

Every unbudded cell with >= fold x the ASH1/CLB2 of a typical unbudded cell in the image is flagged once:
  "late-M pair?"            it has its own full nucleus and an unbudded neighbour (gap <= max_gap_px) that also has
                            one: mother + daughter in late M (not paired), or two cells that have just divided
  "high ASH1/CLB2, no bud"  otherwise: a newborn daughter, or a cell in early S whose small bud is not marked

"Own full nucleus": nucleus area >= nucleus_area x and mean DAPI >= nucleus_intensity x those of a typical
unbudded cell (the late-M rule used to curate the bud-model training data). Because the amount of ASH1/CLB2 is used,
not where it is, this also works for she2/she3 mutants.
The reason text also gives a rough DNA-content hint: the DAPI signal of the nucleus (area x mean intensity of the
DAPI max projection) relative to a typical unbudded cell; ~1 = G1-like, higher = replicating (S) or G2. It is only a
hint (a max projection is not a DNA measurement).

Settings: "cell_flags" block in the config file (defaults in experiment.py, CELL_FLAG_DEFAULTS).
"""
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

COLUMNS = ["label", "flag", "reason", "y0", "x0", "y1", "x1", "ash_rel", "dna_rel"]
LATE_M, HIGH_ASH = "late-M pair?", "high ASH1/CLB2, no bud"


def _col(df, name, default=0.0):
    return df[name].astype(float) if name in df else pd.Series(default, index=df.index, dtype=float)


def compute_flags(cells, masks, settings):
    """
    cells    : the ASH1/CLB2 channel table written by assign_spots ({channel}.csv)
    masks    : corrected cell masks (mother + bud share one label), same frame as the table
    settings : fold, nucleus_area, nucleus_intensity, max_gap_px
    Returns a DataFrame with COLUMNS.
    """
    H, W = masks.shape
    t = cells.set_index("image_cell_id")
    t = t[t.index > 0]
    bud_area = _col(t, "area_bud")
    unbudded = bud_area <= 0
    edge = (t["bbox-0"] <= 0) | (t["bbox-1"] <= 0) | (t["bbox-2"] >= H) | (t["bbox-3"] >= W)
    ref = t[unbudded & ~edge]
    ash = t["total_RNAs"].astype(float)
    ash_ref = max(float(ref["total_RNAs"].median()) if len(ref) else 1.0, 1.0)
    n_area, n_int = _col(t, "mother_nuclei_area"), _col(t, "mother_nuclei_mean_intensity")
    has_nuc = (n_area > 30)[unbudded & ~edge]
    ref_area = float(n_area[has_nuc[has_nuc].index].median()) if has_nuc.any() else np.nan
    ref_int = float(n_int[has_nuc[has_nuc].index].median()) if has_nuc.any() else np.nan
    ref_dna = float((n_area * n_int)[has_nuc[has_nuc].index].median()) if has_nuc.any() else np.nan
    own_nuc = (n_area >= settings["nucleus_area"] * ref_area) & (n_int >= settings["nucleus_intensity"] * ref_int)
    ash_rel = ash / ash_ref
    dna_rel = (n_area * n_int) / ref_dna if ref_dna and ref_dna > 0 else pd.Series(np.nan, index=t.index)
    fold, gap = float(settings["fold"]), int(settings["max_gap_px"])

    def box(ids):
        r = t.loc[list(ids)]
        return int(r["bbox-0"].min()), int(r["bbox-1"].min()), int(r["bbox-2"].max()), int(r["bbox-3"].max())

    def dna_text(i):
        v = dna_rel.get(i, np.nan)
        return "" if not np.isfinite(v) else f" DAPI signal {v:.1f}x a typical unbudded cell (rough DNA-content hint: ~1 = G1-like, higher = S/G2-like)."

    rows, used = [], set()
    boxes = ndi.find_objects(masks.astype(np.int64))

    def neighbours(g):
        if g - 1 >= len(boxes) or boxes[g - 1] is None:
            return []
        sl = tuple(slice(max(s.start - gap - 1, 0), s.stop + gap + 1) for s in boxes[g - 1])
        m = masks[sl] == g
        near = ndi.binary_dilation(m, iterations=gap) & ~m
        return [int(h) for h in np.unique(masks[sl][near]) if h not in (0, g) and h in t.index]

    # every unbudded cell with high ASH1/CLB2 is flagged once, highest first
    high = [i for i in t.index if unbudded[i] and not edge[i] and ash_rel[i] >= fold]
    for g in sorted(high, key=lambda i: -ash[i]):
        if g in used:
            continue
        # late-M pair: the high cell and an unbudded neighbour each have their own nucleus
        partners = [h for h in neighbours(g) if h not in used and unbudded[h] and not edge[h] and own_nuc[h]] if own_nuc[g] else []
        if partners:
            h = max(partners, key=lambda j: ash[j])
            together = ash[g] + ash[h]
            y0, x0, y1, x1 = box([g, h])
            rows.append(dict(label=int(g), flag=LATE_M, y0=y0, x0=x0, y1=y1, x1=x1, ash_rel=round(together / ash_ref, 1),
                             dna_rel=round(float(dna_rel.get(g, np.nan)), 2),
                             reason=(f"cells {g} and {h}: each with its own nucleus, ASH1/CLB2 {ash[g]:.0f} + {ash[h]:.0f} "
                                     f"molecules (typical unbudded cell {ash_ref:.0f}) - mother + daughter in late M, "
                                     f"or just divided? Pair them if they belong together.")))
            used |= {g, h}
        else:
            y0, x0, y1, x1 = box([g])
            rows.append(dict(label=int(g), flag=HIGH_ASH, y0=y0, x0=x0, y1=y1, x1=x1, ash_rel=round(float(ash_rel[g]), 1),
                             dna_rel=round(float(dna_rel.get(g, np.nan)), 2),
                             reason=(f"ASH1/CLB2 {ash_rel[g]:.0f}x a typical unbudded cell ({ash[g]:.0f} molecules), no bud: "
                                     f"a newborn daughter, or a small bud that is not marked?" + dna_text(g))))
            used.add(g)
    return pd.DataFrame(rows, columns=COLUMNS)
