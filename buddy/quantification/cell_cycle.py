"""
Cell-cycle phase of every cell (steps 3b and 5), from what buddy already measures: bud size, nuclei in the mother
and in the bud, and the ASH1/CLB2 mRNAs of the cell. Written to cell_cycle.csv and shown in napari as the
'cell cycle' layer (S, G2 and M written on the cells; unbudded cells get no label).

    from buddy.quantification.cell_cycle import assign_phases
    phases = assign_phases(table, settings, pixel_um, ash_rnas)

The rules (the same as the cell-cycle assignment that was validated against four annotators,
with the bud-size boundary used the same way for G2 and M):

    not assessed  cell length outside min_length_um - max_length_um, or a mother without exactly one nucleus
    G1            no bud, and at most g1_max_ash ASH1/CLB2 mRNAs in the cell
    unbudded      no bud, and no ASH1/CLB2 channel to tell G1 apart (strains without the probe)
    S             a bud smaller than s_g2_bud_um2, without a nucleus
    G2            a bud of s_g2_bud_um2 up to max_bud_um2, without a nucleus
    M             a bud of at least s_g2_bud_um2 with one nucleus in it
    NA            anything else: no bud but high ASH1/CLB2 (a newborn daughter, or a small bud that is not marked),
                  a nucleus in a very small bud, a bud larger than max_bud_um2 without a nucleus, ...
                  The reason is in the 'why' column.

Settings: "cell_cycle" block in the config file (defaults: CELL_CYCLE_DEFAULTS in buddy/config.py).
The S / G2 split rests on the bud size, so the phases are only as good as the bud masks: check the buds first (step 4).
"""
import numpy as np
import pandas as pd

PHASES = ("G1", "S", "G2", "M")
UNBUDDED, NA, NOT_ASSESSED = "unbudded", "NA", "not assessed"
COLUMNS = ["image_cell_id", "cell_cycle_phase", "why", "bud_area_um2", "cell_length_um",
           "mother_nuclei_count", "nuclei_bud_count", "ash_RNAs"]


def _col(df, name):
    return df[name].astype(float) if name in df else pd.Series(0.0, index=df.index)


def assign_phases(table, settings, pixel_um, ash_rnas=None):
    """
    table     one channel table written by assign_spots ({channel}.csv: morphology, bud and nucleus columns)
    settings  the "cell_cycle" block of the config (g1_max_ash, s_g2_bud_um2, max_bud_um2, min/max_length_um)
    pixel_um  pixel size in µm (y/x)
    ash_rnas  ASH1/CLB2 mRNAs per cell (Series indexed by image_cell_id), or None without that channel
    Returns a DataFrame with COLUMNS, one row per cell.
    """
    t = table.set_index("image_cell_id")
    t = t[t.index > 0]
    bud = _col(t, "area_bud") * pixel_um ** 2
    length = _col(t, "axis_major_length") * pixel_um
    n_mother, n_bud = _col(t, "mother_nuclei_count"), _col(t, "nuclei_bud_count")
    ash = ash_rnas.reindex(t.index).astype(float) if ash_rnas is not None else pd.Series(np.nan, index=t.index)
    s_g2, max_bud = float(settings["s_g2_bud_um2"]), float(settings["max_bud_um2"])

    phase, why = [], []
    for i in t.index:
        b, nb = bud[i], n_bud[i]
        if not settings["min_length_um"] < length[i] < settings["max_length_um"]:
            p, w = NOT_ASSESSED, f"cell length {length[i]:.1f} µm"
        elif n_mother[i] != 1:
            p, w = NOT_ASSESSED, f"{n_mother[i]:.0f} nuclei in the mother"
        elif b <= 0:
            if ash_rnas is None:
                p, w = UNBUDDED, "no bud (no ASH1/CLB2 channel to tell G1 apart)"
            elif ash[i] <= settings["g1_max_ash"]:
                p, w = "G1", f"no bud, {ash[i]:.0f} ASH1/CLB2 mRNAs"
            else:
                p, w = NA, f"no bud but {ash[i]:.0f} ASH1/CLB2 mRNAs (newborn daughter, or an unmarked bud?)"
        elif nb == 0:
            if b < s_g2:
                p, w = "S", f"bud {b:.1f} µm², no nucleus"
            elif b < max_bud:
                p, w = "G2", f"bud {b:.1f} µm², no nucleus"
            else:
                p, w = NA, f"bud {b:.1f} µm² without a nucleus (larger than {max_bud:g} µm²)"
        elif nb == 1:
            if b >= s_g2:
                p, w = "M", f"bud {b:.1f} µm² with a nucleus"
            else:
                p, w = NA, f"nucleus in a small bud ({b:.1f} µm²)"
        else:
            p, w = NA, f"{nb:.0f} nuclei in the bud"
        phase.append(p); why.append(w)

    out = pd.DataFrame({"image_cell_id": t.index.astype(int), "cell_cycle_phase": phase, "why": why,
                        "bud_area_um2": bud.round(2).to_numpy(), "cell_length_um": length.round(2).to_numpy(),
                        "mother_nuclei_count": n_mother.astype(int).to_numpy(), "nuclei_bud_count": n_bud.astype(int).to_numpy(),
                        "ash_RNAs": ash.to_numpy()})
    return out[COLUMNS]


def phase_counts(phases):
    """{phase: number of cells} in the order G1, S, G2, M, unbudded, NA, not assessed (for img.json / the log)."""
    order = list(PHASES) + [UNBUDDED, NA, NOT_ASSESSED]
    counts = phases["cell_cycle_phase"].value_counts()
    return {p: int(counts.get(p, 0)) for p in order if counts.get(p, 0)}
