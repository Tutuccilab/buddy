"""
Step 2, clean-up of the Omnipose masks: cells without a nucleus, cells smaller than min_cell_area and (optionally)
cells touching the image edge are dropped, cells can be grown by expand_px, nuclei without a cell are dropped.

    from buddy.segmentation.cleanup import clean_up_masks
"""
import logging
from copy import copy

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from skimage.segmentation import expand_labels

log = logging.getLogger("buddy")


def clean_up_masks(cell_masks, nucleus_masks, parameters):
    """
    The clean-up of step 2: cells without a nucleus and cells smaller than min_cell_area are dropped, optionally
    cells touching the image border (remove_edge_cells), then every cell is grown by expand_px; nuclei without a
    cell are dropped. Returns (cell masks, nucleus masks, settings used).
    """
    min_cell_area = parameters.get("min_cell_area", 20)
    expand_px = parameters.get("expand_px", 0)
    remove_edge = bool(parameters.get("remove_edge_cells", False))
    log.info(f"postprocessing masks (min_cell_area={min_cell_area}, expand_px={expand_px}, remove_edge_cells={remove_edge})")
    log.info("..removing cell masks with no DAPI")
    cells = remove_cell_masks_with_no_dapi(cell_masks, nucleus_masks)[0]
    log.info(f"..discarding masks of size < {min_cell_area}")
    cells = filter_cells_by_area(cells, get_regionprops(cell_masks), min_cell_area=min_cell_area)[0]
    n_edge = 0
    if remove_edge:                 # cells cut off by the image border are incomplete (too few RNAs, bud possibly outside)
        from skimage.segmentation import clear_border
        before = set(np.unique(cells)) - {0}
        cells = clear_border(cells)
        n_edge = len(before - set(np.unique(cells)))
        log.info(f"..removed {n_edge} cell(s) touching the image edge")
    log.info("..removing DAPI masks with no cell")
    nuclei, _ = remove_dapi_masks_with_no_cell(nucleus_masks, cells)
    if expand_px:
        log.info(f"..expanding cell masks by {expand_px} px")
        cells = expand_masks(cells, expand_px)
    settings = {"min_cell_area": min_cell_area, "expand_px": expand_px, "remove_edge_cells": remove_edge, "edge_cells_removed": n_edge}
    return cells, nuclei, settings

# -------------------------------------------------------------------------------------------------
# mask clean-up helpers (used by clean_up_masks)
# -------------------------------------------------------------------------------------------------


def expand_masks(masks, nr_of_pixels):
    return expand_labels(masks, distance=nr_of_pixels)


def get_regionprops(masks):
    properties = ['label', 'bbox', 'area', 'eccentricity', 'centroid', 'axis_major_length', 'axis_minor_length',
                  'orientation']
    return pd.DataFrame(regionprops_table(masks, properties=properties))


def remove_cell_masks_with_no_dapi(cell_masks, dapi_masks):
    cells_to_discard = []
    for label in np.unique(cell_masks):
        dapi_found = False
        cell_coordinates = np.where(cell_masks == label)
        for y, x in zip(cell_coordinates[0], cell_coordinates[1]):
            if dapi_masks[y, x] != 0:
                dapi_found = True
                break
        if not dapi_found:
            cells_to_discard.append(label)
    log.info(f'....{len(cells_to_discard)} cell(s) without DAPI signal removed')
    cell_masks_dapi = copy(cell_masks)
    cell_masks_dapi[np.isin(cell_masks, cells_to_discard)] = 0
    return cell_masks_dapi, cells_to_discard


def remove_dapi_masks_with_no_cell(dapi_masks, cell_masks):
    # keep only nuclei (DAPI labels) that overlap at least one cell pixel
    labels_with_cell = set(np.unique(dapi_masks[cell_masks > 0])) - {0}
    all_labels = set(np.unique(dapi_masks)) - {0}
    discarded = list(all_labels - labels_with_cell)
    log.info(f'....{len(discarded)} DAPI mask(s) without a cell removed')
    dapi_clean = copy(dapi_masks)
    dapi_clean[np.isin(dapi_masks, discarded)] = 0
    return dapi_clean, discarded


def filter_cells_by_area(masks, props, min_cell_area=2000):
    discarded = props.query('area < @min_cell_area')['label']
    selected = set(props['label']) - set(discarded)
    masks_selected = copy(masks)
    masks_selected[np.isin(masks, list(discarded))] = 0
    masks_discarded = copy(masks)
    masks_discarded[np.isin(masks, list(selected))] = 0

    return masks_selected, masks_discarded, len(np.unique(masks_selected)), len(np.unique(masks_discarded)), len(np.unique(masks))


def filter_cells_by_shape(masks, props, min_clump_area=1000, max_clump_eccentricity=0.8):
    discarded = props.query('area > @min_clump_area').query('eccentricity < @max_clump_eccentricity')['label']
    selected = set(props['label']) - set(discarded)
    masks_selected = copy(masks)
    masks_selected[np.isin(masks, list(discarded))] = 0
    masks_discarded = copy(masks)
    masks_discarded[np.isin(masks, list(selected))] = 0

    return masks_selected, masks_discarded, len(np.unique(masks_selected)), len(np.unique(masks_discarded)), len(np.unique(masks))
