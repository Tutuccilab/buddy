"""
Helpers for the manual mask-correction / bud-annotation step of the FISH pipeline.

Pure array functions (no napari dependency) so they're easy to test and reuse.
The notebook builds the napari layers from what these return.

Bud-mask class convention (matches bud_corrected_masks.tif, and assign_spots
reads bud == 1):
    1 = bud,  2 = background,  3 = mother
"""
import numpy as np
from skimage.measure import regionprops, label as cc_label
from scipy.ndimage import binary_dilation

# 3-class bud-mask labels
BG, BUD, MOTHER = 2, 1, 3


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
    seed = np.full(cell_masks.shape, BG, dtype=np.uint8)
    seed[cell_masks > 0] = MOTHER
    if bud_labels:
        seed[np.isin(cell_masks, list(bud_labels))] = BUD
    if keep_bud_mask is not None:
        seed[np.asarray(keep_bud_mask) == BUD] = BUD    # preserve existing buds
    return seed


def candidate_rectangles(props, labels):
    """List of (4, 2) rectangle corner arrays, one per prop whose label is in `labels`."""
    return [_bbox_rectangle(p.bbox) for p in props if p.label in labels]


def assign_buds_to_mothers(dic_masks, bud_masks, bud_value=BUD, border=1):
    """
    Assign each connected bud (bud_masks == bud_value) to its mother cell.

    The annotated DIC cell (the dominant label under the bud annotation) is taken
    to be the bud. Its mother is the cell that the *whole bud cell* touches most.
    A bud is only linked to a mother that is at least as large as the bud, so a
    normal cell is never merged into a smaller neighbour.

    Returns
    -------
    bud_labeled : label image; each bud's pixels carry its assigned mother label.
    assignments : dict {bud_component_index: (mother_label, own_label) or (None, own_label)}
    unassigned  : list of bud_component_index with no suitable mother
    bud_cc      : connected-component labelling of the buds.
    """
    dic_masks = np.asarray(dic_masks)
    bud_cc = cc_label(np.asarray(bud_masks) == bud_value, connectivity=2)
    bud_labeled = np.zeros_like(dic_masks)
    assignments, unassigned = {}, []

    for comp in range(1, int(bud_cc.max()) + 1):
        comp_mask = bud_cc == comp

        own = dic_masks[comp_mask]
        own = own[own > 0]
        own_label = int(np.bincount(own).argmax()) if own.size else None

        # region whose neighbours we search: the whole bud cell, else the annotation
        region = (dic_masks == own_label) if own_label is not None else comp_mask
        ring = binary_dilation(region, iterations=border) & ~region
        neigh = dic_masks[ring]
        neigh = neigh[(neigh > 0) & (neigh != own_label)]
        if neigh.size == 0:
            assignments[comp] = (None, own_label)
            unassigned.append(comp)
            continue
        vals, counts = np.unique(neigh, return_counts=True)
        mother = int(vals[counts.argmax()])

        # a bud must be no larger than its mother (guards against merging a real
        # cell into a smaller neighbour)
        if own_label is not None and (dic_masks == own_label).sum() > (dic_masks == mother).sum():
            assignments[comp] = (None, own_label)
            unassigned.append(comp)
            continue

        assignments[comp] = (mother, own_label)
        bud_labeled[comp_mask] = mother

    return bud_labeled, assignments, unassigned, bud_cc


def merge_buds_into_mothers(dic_masks, assignments, bud_cc, bud_value=BUD):
    """
    Relabel each linked bud's DIC cell to its mother's label, so a mother and its
    bud share one label in the returned DIC mask. Unassigned buds are untouched.
    Returns a copy of dic_masks with the merges applied.
    """
    dic = np.asarray(dic_masks).copy()
    for comp, (mother, own_label) in assignments.items():
        if mother is None:
            continue
        if own_label is not None and own_label != mother:
            dic[dic == own_label] = mother          # merge whole bud cell
        # add only background pixels of the annotation (never clobber other cells)
        comp_mask = bud_cc == comp
        dic[comp_mask & (dic == 0)] = mother
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
