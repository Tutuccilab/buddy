"""The cell-cycle phase rules (buddy/quantification/cell_cycle.py) on a made-up table, and on the test field."""
import pandas as pd
from buddy.config import CELL_CYCLE_DEFAULTS
from buddy.quantification.cell_cycle import NA, NOT_ASSESSED, UNBUDDED, assign_phases, phase_counts

PX = 0.065                                   # µm per pixel
UM2 = 1 / PX ** 2                            # pixels per µm²
SETTINGS = dict(CELL_CYCLE_DEFAULTS)


def cell(i, bud_um2=0.0, nuclei=1, bud_nuclei=0, length_um=5.0):
    return {"image_cell_id": i, "area_bud": bud_um2 * UM2, "axis_major_length": length_um / PX,
            "mother_nuclei_count": nuclei, "nuclei_bud_count": bud_nuclei}


def phases_of(rows, ash):
    table = pd.DataFrame(rows)
    ash = None if ash is None else pd.Series(ash, index=table.image_cell_id)
    return assign_phases(table, SETTINGS, PX, ash).set_index("image_cell_id").cell_cycle_phase.to_dict()


def test_the_four_phases():
    rows = [cell(1), cell(2, bud_um2=1.0), cell(3, bud_um2=4.0), cell(4, bud_um2=6.0, bud_nuclei=1)]
    assert phases_of(rows, [2, 30, 30, 30]) == {1: "G1", 2: "S", 3: "G2", 4: "M"}


def test_boundaries():
    rows = [cell(1, bud_um2=1.99), cell(2, bud_um2=2.0), cell(3, bud_um2=2.0, bud_nuclei=1),   # 2 µm²: G2 / M
            cell(4, bud_um2=15.0), cell(5, bud_um2=20.0, bud_nuclei=1)]                       # no upper limit for M
    assert phases_of(rows, [0] * 5) == {1: "S", 2: "G2", 3: "M", 4: NA, 5: "M"}


def test_g1_needs_low_ash1_clb2():
    rows = [cell(1), cell(2)]
    assert phases_of(rows, [5, 6]) == {1: "G1", 2: NA}                  # <= g1_max_ash (5) mRNAs


def test_without_ash1_clb2_channel_no_g1():
    rows = [cell(1), cell(2, bud_um2=1.0)]
    assert phases_of(rows, None) == {1: UNBUDDED, 2: "S"}


def test_not_assessed_and_na():
    rows = [cell(1, length_um=1.5), cell(2, length_um=25), cell(3, nuclei=0), cell(4, nuclei=2),
            cell(5, bud_um2=1.0, bud_nuclei=1), cell(6, bud_um2=5.0, bud_nuclei=2)]
    got = phases_of(rows, [0] * 6)
    assert [got[i] for i in (1, 2, 3, 4)] == [NOT_ASSESSED] * 4
    assert got[5] == NA and got[6] == NA                                 # nucleus in a tiny bud; two in one bud


def test_background_row_is_ignored_and_reasons_are_given():
    rows = [cell(0), cell(1, bud_um2=1.0)]
    out = assign_phases(pd.DataFrame(rows), SETTINGS, PX, pd.Series([0, 0], index=[0, 1]))
    assert out.image_cell_id.tolist() == [1]
    assert "bud 1.0 µm²" in out.why.iloc[0]
    assert phase_counts(out) == {"S": 1}


def test_on_the_test_field(field_a):
    """the reference counts of the test field get a phase for every cell; most are assessed"""
    ref = field_a / "reference" / "step3"
    table = pd.read_csv(ref / "ASH1CLB2.csv")
    out = assign_phases(table, SETTINGS, PX, table.set_index("image_cell_id")["total_RNAs"])
    assert len(out) == (table.image_cell_id > 0).sum()
    assert set(out.cell_cycle_phase) <= {"G1", "S", "G2", "M", NA, NOT_ASSESSED}
    assert (out.cell_cycle_phase != NOT_ASSESSED).mean() > 0.5
