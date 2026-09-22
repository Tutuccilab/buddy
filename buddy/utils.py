import numpy as np
from copy import copy
from skimage.segmentation import expand_labels
from skimage.measure import regionprops_table
from skimage.measure import regionprops
import pandas as pd
from typing import Tuple
from copy import copy

from skimage.measure import regionprops


# with eternal thanks to the interwebs:
def translate_image(img, tx, ty):
    N, M = img.shape
    img_translated = np.zeros_like(img)

    img_translated[max(tx, 0):M + min(tx, 0), max(ty, 0):N + min(ty, 0)] = img[-min(tx, 0):M - max(tx, 0),
                                                                             -min(ty, 0):N - max(ty, 0)]

    img_translated_cropped = img[-min(tx, 0):M - max(tx, 0), -min(ty, 0):N - max(ty, 0)]
    return img_translated, img_translated_cropped

def expand_masks(masks, nr_of_pixels):
    return expand_labels(masks, distance=nr_of_pixels)


def get_regionprops(masks):
    properties = ['label', 'bbox', 'area', 'eccentricity', 'centroid', 'axis_major_length', 'axis_minor_length',
                  'orientation']
    return pd.DataFrame(regionprops_table(masks, properties=properties))


def find_high_density_patch(mask: np.ndarray, patch_size: Tuple = (200, 200)):
    """
    Finds the densest patch in the mask deterministically.

    :param mask: segmentation image expected to have dimension (h, w)
    :param patch_size: height and width of the patch
    :return: coordinates of top-left corner of densest patch
    """

    h_patch, w_patch = patch_size

    binary = (mask > 0).astype(np.int32)

    # integral image
    integral = binary.cumsum(axis=0).cumsum(axis=1)

    h, w = mask.shape

    best_sum = -1
    best_coord = (0, 0)

    for r in range(h - h_patch):
        r2 = r + h_patch
        for c in range(w - w_patch):
            c2 = c + w_patch

            s = integral[r2-1, c2-1]
            if r > 0:
                s -= integral[r-1, c2-1]
            if c > 0:
                s -= integral[r2-1, c-1]
            if r > 0 and c > 0:
                s += integral[r-1, c-1]

            if s > best_sum:
                best_sum = s
                best_coord = (r, c)

    return best_coord


def find_in_focus_indices(focus: np.ndarray, adjustment_bottom: int = 5, adjustment_top: int = 10):
    """

    find the in-focus indices of calculated focus scores

    :param focus: series of values representing max intensity along z-axis
    :param adjustment_bottom: controls by how much the resulting range should be padded (bottom)
    :param adjustment_top: controls by how much the resulting range should be padded (top)

    :return: low and high z-level between which the spots are in focus
    :rtype: Tuple[int,int]
    """

    # find the inflection points of the smoothed curve
    ifx_1 = min([np.diff(focus).argmax(), np.diff(focus).argmin()])
    ifx_2 = max([np.diff(focus).argmax(), np.diff(focus).argmin()])

    # add a little cushion to one side.
    ifx_1 -= adjustment_bottom
    ifx_2 += adjustment_top

    return ifx_1, ifx_2


def preprocess_spot_data(spot_data, dense_data):
    # spot_data has the form:
    # z, y, x

    # dense data has the form
    # z, y, x, mRNA counts, -- other information --

    # let's introduce mRNA counts of 1 for the spots:
    spot_data_padded = np.pad(spot_data, ((0,0),(0,1)), mode='constant', constant_values=1)

    # discard other information and merge
    spot_data_combined = np.concatenate([spot_data_padded, dense_data[:,:4]], axis=0)
    return spot_data_combined


def count_spots(mask, nuclear_mask, spot_data, cells):
    for z, y, x, number in spot_data:
        y, x = int(y), int(x)
        cell_id = mask[y, x]
        nucleus = nuclear_mask[y, x]

        if number == 1:
            cells[cell_id]['spots'] += number
        else:
            cells[cell_id]['dense_regions'] += 1
            cells[cell_id]['decomposed_RNAs'] += number

            # if the spot sits in the nucleus,
            # also increase nascent RNAs and transcription sites
            if nucleus > 0:
                cells[cell_id]['tx'] += 1
                cells[cell_id]['nascent_RNAs'] += number
    return cells


def count_nuclei(mask, nuclear_mask, cells):
    # count nuclei per cell - hyphae may have multiple ones!
    for nucleus in regionprops(nuclear_mask):
        y, x = nucleus.centroid
        cell_id = mask[int(y), int(x)]
        cells[cell_id]['nuclei'] += 1
    return cells


def spot_assignment(mask, bud_mask, nuclear_mask, zprojDAPI, spot_data, dense_data):
    # bud_mask=None -> run without a bud compartment (no *_bud columns).
    include_bud = bud_mask is not None

    # Initialize cell dictionary (whole cell / mother)
    cells = {}
    for cell_id in np.unique(mask):
        cells[cell_id] = {'nuclei':0,'spots':0,'dense_regions':0,'decomposed_RNAs':0,'tx':0,'nascent_RNAs':0}

    # Preprocess spot data
    spot_data_combined = preprocess_spot_data(spot_data, dense_data)

    # Count spots/nascent RNAs and nuclei for the whole cell
    cells = count_spots(mask, nuclear_mask, spot_data_combined, cells)
    cells = count_nuclei(mask, nuclear_mask, cells)
    del cells[0]

    df_cells = pd.DataFrame(cells).T.reset_index().rename(columns={'index':'label'})
    df_cells['total_RNAs'] = df_cells['spots'] + df_cells['decomposed_RNAs'] - df_cells['dense_regions']

    # Morphology (whole cell)
    props = pd.DataFrame(regionprops_table(
        mask,
        properties=['label','bbox','area','eccentricity','axis_minor_length','axis_major_length',
                    'orientation','perimeter','solidity']
    ))

    # Nuclear properties: mother = cell minus bud, or the whole cell when there is no bud
    mother_mask_only = np.where(bud_mask == 0, mask, 0) if include_bud else mask
    nuclei_props = pd.DataFrame(regionprops_table(
        np.where(nuclear_mask>0,mother_mask_only,0),
        zprojDAPI,
        properties=['label','area','mean_intensity']
    ))
    nuclei_props_agg = nuclei_props.groupby('label').agg(
        mother_nuclei_count=('area','count'),
        mother_nuclei_area=('area','sum'),
        mother_nuclei_mean_intensity=('mean_intensity','mean')
    ).reset_index()

    if include_bud:
        # ---- bud compartment ----
        cells_bud = {}
        for cell_id in np.unique(mask):
            cells_bud[cell_id] = {'nuclei':0,'spots':0,'dense_regions':0,'decomposed_RNAs':0,'tx':0,'nascent_RNAs':0}
        cells_bud = count_spots(bud_mask, nuclear_mask, spot_data_combined, cells_bud)
        cells_bud = count_nuclei(bud_mask, nuclear_mask, cells_bud)
        del cells_bud[0]

        df_bud = pd.DataFrame(cells_bud).T.reset_index().rename(columns={'index':'label'})
        df_bud['total_RNAs_bud'] = df_bud['spots'] + df_bud['decomposed_RNAs'] - df_bud['dense_regions']
        bud_cols = ['spots','dense_regions','decomposed_RNAs','tx','nascent_RNAs','nuclei']
        df_bud = df_bud[['label'] + bud_cols + ['total_RNAs_bud']]
        df_bud = df_bud.rename(columns={c:f"{c}_bud" for c in bud_cols})

        props_bud = pd.DataFrame(regionprops_table(
            bud_mask,
            properties=['label','bbox','area','eccentricity','axis_minor_length','axis_major_length',
                        'orientation','perimeter','solidity']
        ))
        props_bud = props_bud.rename(columns={c: f"{c}_bud" for c in props_bud.columns if c != 'label'})

        nuclei_props_bud = pd.DataFrame(regionprops_table(
            np.where(nuclear_mask>0,bud_mask,0),
            zprojDAPI,
            properties=['label','area','mean_intensity']
        ))
        nuclei_props_bud_agg = nuclei_props_bud.groupby('label').agg(
            nuclei_bud_count=('area','count'),
            nuclei_bud_area=('area','sum'),
            nuclei_bud_mean_intensity=('mean_intensity','mean')
        ).reset_index()

        # Merge everything (same order/columns as the original bud-enabled version)
        df_all = props.merge(props_bud, on='label', how='left')
        df_all = df_all.merge(df_cells, on='label', how='left')
        df_all = df_all.merge(df_bud, on='label', how='left')
        df_all = df_all.merge(nuclei_props_agg, on='label', how='left')
        df_all = df_all.merge(nuclei_props_bud_agg, on='label', how='left')
        df_all = df_all.fillna(0)

        df_all['spots_mother'] = df_all['spots'] - df_all['spots_bud']
        df_all['total_RNAs_mother'] = df_all['total_RNAs'] - df_all['total_RNAs_bud']
    else:
        # ---- no bud compartment: whole-cell output only ----
        df_all = props.merge(df_cells, on='label', how='left')
        df_all = df_all.merge(nuclei_props_agg, on='label', how='left')
        df_all = df_all.fillna(0)

    return df_all


def remove_cell_masks_with_no_dapi(cell_masks, dapi_masks):
    cells_to_discard = []
    for label in np.unique(cell_masks):
        dapi_found = False
        cell_coordinates = np.where(cell_masks == label)
        for y, x in zip(cell_coordinates[0], cell_coordinates[1]):
            if dapi_masks[y, x] != 0:
                dapi_found = True
                break
        if dapi_found == False:
            cells_to_discard.append(label)
    print(f'removing {len(cells_to_discard)} cells with no dapi signal')
    cell_masks_dapi = copy(cell_masks)
    cell_masks_dapi[np.isin(cell_masks, cells_to_discard)] = 0
    return cell_masks_dapi, cells_to_discard


def remove_dapi_masks_with_no_cell(dapi_masks, cell_masks):
    # keep only nuclei (DAPI labels) that overlap at least one cell pixel
    labels_with_cell = set(np.unique(dapi_masks[cell_masks > 0])) - {0}
    all_labels = set(np.unique(dapi_masks)) - {0}
    discarded = list(all_labels - labels_with_cell)
    print(f'removing {len(discarded)} DAPI masks with no cell')
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