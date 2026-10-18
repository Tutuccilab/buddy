"""
Helpers for the manual mask-correction / bud-annotation step of the FISH pipeline.

Pure array functions (no napari dependency) so they're easy to test and reuse.
The notebook builds the napari layers from what these return.

Bud-mask class convention (matches bud_corrected_masks.tif, and assign_spots
reads bud == 1):
    1 = bud,  2 = background,  3 = mother
"""
import numpy as np
from scipy.ndimage import binary_dilation
from skimage.measure import label as cc_label
from skimage.measure import regionprops

# 3-class bud-mask values, the one definition used everywhere in buddy (bud model, spot assignment, napari-buddy)
BUD, BACKGROUND, MOTHER = 1, 2, 3
BG = BACKGROUND             # old name, kept for existing notebooks


def _bbox_rectangle(bbox):
    """(min_row, min_col, max_row, max_col) -> (4, 2) corner array for a napari rectangle."""
    r0, c0, r1, c1 = bbox
    return np.array([[r0, c0], [r0, c1], [r1, c1], [r1, c0]])


def classify_cells(cell_masks, bud_area_fraction=0.5,
                   min_solidity=0.93, max_round_eccentricity=0.85):
    """
    Classify segmented cells into candidate buds and non-round ("potentially
    budded") cells.

    Parameters
    ----------
    cell_masks : label image (the DIC cell segmentation)
    bud_area_fraction : cells smaller than this * median cell area are candidate buds
    min_solidity : cells below this solidity are "non-round" (notched)
    max_round_eccentricity : cells above this eccentricity are "non-round" (elongated)

    Returns
    -------
    dict with:
        'props'          : list of regionprops
        'bud_labels'     : set of labels flagged as candidate buds (small)
        'budded_labels'  : set of labels flagged as non-round (excludes bud_labels)
        'median_area'    : median cell area (px)
        'bud_threshold'  : area threshold used (px)
    """
    props = regionprops(np.asarray(cell_masks))
    areas = np.array([p.area for p in props], dtype=float)
    median_area = float(np.median(areas)) if areas.size else 0.0
    threshold = bud_area_fraction * median_area

    bud_labels = {p.label for p in props if p.area < threshold}
    budded_labels = {p.label for p in props
                     if p.label not in bud_labels
                     and (p.solidity < min_solidity
                          or p.eccentricity > max_round_eccentricity)}

    return {'props': props, 'bud_labels': bud_labels, 'budded_labels': budded_labels,
            'median_area': median_area, 'bud_threshold': threshold}


def build_bud_seed(cell_masks, bud_labels, keep_bud_mask=None):
    """
    3-class seed: background (2) everywhere, every cell -> mother (3), the given
    bud_labels -> bud (1). If keep_bud_mask is given (an existing bud mask), any
    pixel already marked as bud there stays a bud, so buds keep their status even
    after their DIC cell has been merged into a mother. Returns a uint8 array.
    """
    cell_masks = np.asarray(cell_masks)
    seed = np.full(cell_masks.shape, BACKGROUND, dtype=np.uint8)
    seed[cell_masks > 0] = MOTHER
    if bud_labels:
        seed[np.isin(cell_masks, list(bud_labels))] = BUD
    if keep_bud_mask is not None:
        seed[np.asarray(keep_bud_mask) == BUD] = BUD    # preserve existing buds
    return seed


def candidate_rectangles(props, labels):
    """List of (4, 2) rectangle corner arrays, one per prop whose label is in `labels`."""
    return [_bbox_rectangle(p.bbox) for p in props if p.label in labels]


def cells_inside(dic_masks, region):
    """
    Labels of the segmented cells that lie mostly (> 50 %) inside `region`, most overlapping first.
    Same rule as the bud model (bud_model.build_masks): such a cell is part of the bud.
    """
    dic_masks = np.asarray(dic_masks)
    labels, inside = np.unique(dic_masks[region], return_counts=True)
    keep = labels > 0
    labels, inside = labels[keep], inside[keep]
    if not labels.size:
        return []
    area = np.bincount(dic_masks.ravel(), minlength=int(labels.max()) + 1)[labels]
    mostly = inside / area > 0.5
    order = np.argsort(-inside[mostly], kind="stable")
    return [int(v) for v in labels[mostly][order]]


def assign_buds_to_mothers(dic_masks, bud_masks, bud_value=BUD, border=1):
    """
    Assign each connected bud (bud_masks == bud_value) to its mother cell.

    Same rules as the bud model (bud_model.py) where the two overlap:
    - bud pixels are joined only through shared edges (4-connectivity), like the model's claims;
    - a segmented cell that lies mostly (> 50 %) inside the bud is part of the bud.

    Buds made by the bud model are already merged into their mother (the bud carries the mother's
    label) and keep the model's mother. For buds painted by hand (or when the model was not used),
    the mother is the cell that the bud - its cells, else the painted pixels - touches most along a
    1-px ring. A bud made of segmented cells is only linked to a mother that is at least as large,
    so a normal cell is never merged into a smaller neighbour.

    Returns
    -------
    bud_labeled : label image; each bud's pixels carry its assigned mother label.
    assignments : dict {bud_component_index: (mother_label, own_label) or (None, own_label)};
                  own_label is the cell that is (mostly) the bud, None if no cell lies mostly inside it.
    unassigned  : list of bud_component_index with no suitable mother
    bud_cc      : connected-component labelling of the buds.
    """
    dic_masks = np.asarray(dic_masks)
    is_bud = np.asarray(bud_masks) == bud_value
    # connected bud pixels per DIC cell, so two touching buds of different cells stay separate buds
    bud_cc = cc_label(np.where(is_bud, dic_masks.astype(np.int64) + 1, 0), background=0, connectivity=1)
    bud_labeled = np.zeros_like(dic_masks)
    assignments, unassigned = {}, []

    for comp in range(1, int(bud_cc.max()) + 1):
        comp_mask = bud_cc == comp

        # already linked: the bud carries the label of a cell that also has non-bud (mother) pixels
        # (e.g. after automatic bud classification, which merges each bud into its mother)
        under = dic_masks[comp_mask]
        under = under[under > 0]
        dominant = int(np.bincount(under).argmax()) if under.size else None
        if dominant is not None and ((dic_masks == dominant) & ~is_bud).sum() > comp_mask.sum() // 2:
            assignments[comp] = (dominant, dominant)
            bud_labeled[comp_mask] = dominant
            continue

        # the bud: the cells lying mostly inside it, else the painted pixels themselves
        own = cells_inside(dic_masks, comp_mask)
        own_label = own[0] if own else None
        region = np.isin(dic_masks, own) if own else comp_mask
        ring = binary_dilation(region, iterations=border) & ~region
        neigh = dic_masks[ring]
        neigh = neigh[(neigh > 0) & ~np.isin(neigh, own)]
        if neigh.size == 0:
            assignments[comp] = (None, own_label)
            unassigned.append(comp)
            continue
        vals, counts = np.unique(neigh, return_counts=True)
        mother = int(vals[counts.argmax()])

        # a bud must be no larger than its mother (guards against merging a real
        # cell into a smaller neighbour)
        if own and region.sum() > (dic_masks == mother).sum():
            assignments[comp] = (None, own_label)
            unassigned.append(comp)
            continue

        assignments[comp] = (mother, own_label)
        bud_labeled[comp_mask] = mother

    return bud_labeled, assignments, unassigned, bud_cc


def merge_buds_into_mothers(dic_masks, assignments, bud_cc, bud_value=BUD):
    """
    Relabel the cells of each linked bud (those lying mostly inside it, as in the bud model) to its
    mother's label, so a mother and its bud share one label in the returned DIC mask. Painted bud
    pixels on background join the mother too; pixels on other cells are left alone. Unassigned buds
    are untouched. Returns a copy of dic_masks with the merges applied.
    """
    original = np.asarray(dic_masks)
    dic = original.copy()
    for comp, (mother, own_label) in assignments.items():
        if mother is None:
            continue
        comp_mask = bud_cc == comp
        for label in cells_inside(original, comp_mask):
            if label != mother:
                dic[original == label] = mother          # merge the whole bud cell
        dic[comp_mask & (original == 0)] = mother        # add only background pixels of the annotation
    return dic


def mother_bud_pair_rectangles(dic_merged, assignments, bud_cc):
    """
    Rectangles around each mother-bud pair, computed from the *merged* DIC mask so
    each rectangle frames the whole (mother + bud) cell. One rectangle per mother
    that received a bud; unassigned buds get their own rectangle.

    Returns (pair_rects, orphan_rects).
    """
    dic_merged = np.asarray(dic_merged)
    pair_rects, orphan_rects, seen = [], [], set()
    for comp, (mother, own_label) in assignments.items():
        if mother is None:
            ys, xs = np.where(bud_cc == comp)
            orphan_rects.append(_bbox_rectangle((ys.min(), xs.min(), ys.max() + 1, xs.max() + 1)))
        elif mother not in seen:
            seen.add(mother)
            ys, xs = np.where(dic_merged == mother)
            pair_rects.append(_bbox_rectangle((ys.min(), xs.min(), ys.max() + 1, xs.max() + 1)))
    return pair_rects, orphan_rects
