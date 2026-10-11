"""
Spots per cell, mother and bud (step 3b / step 5): the per-cell table that buddy writes as <channel>.csv.

    from buddy.quantification.spot_assignment import spot_assignment
    table = spot_assignment(cell_masks, bud_labels, nucleus_labels, dapi_projection, spots, dense_regions)

cell_masks     : every cell one label, mother + bud share the mother's label
bud_labels     : the bud pixels, labelled with their mother's label (None = no bud compartment)
nucleus_labels : nucleus pixels, labelled with their cell's label
spots          : (z, y, x, ...) single spots;  dense_regions: (z, y, x, nb_rna, ...) decomposed dense regions
"""
import numpy as np
import pandas as pd
from skimage.measure import regionprops, regionprops_table


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
