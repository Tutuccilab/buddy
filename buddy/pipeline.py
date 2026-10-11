"""
The processing steps for one image, used by 01_fish_bulk_processing.ipynb and by the napari "buddy pipeline"
panel (napari-buddy), so both always do exactly the same.

    read_image(params, exp)          step 1   read the .vsi + DIC, write {channel}.tif and DIC.tif
    segment_image(json_file, exp)    step 2   Omnipose cell + nucleus masks       (first: exp.init_omnipose())
    spots_image(json_file, exp)      step 3a  focus, background filter, spot detection, decomposition, alignment
    buds_and_counts(json_file, exp)  step 3b  bud model, count spots per cell, cell flags (first: exp.init_bud_model())
    reassign_image(json_file, exp)   step 5   count spots again with the corrected masks of step 4

Each function records its step times and settings in the image's img.json.
STEP_TIMES lists which img.json times belong to each step (used for time estimates).
"""
import shutil
import time
from pathlib import Path

from skimage import io

from buddy.image import Image

STEP_TIMES = {
    'read': ['01-configure'],
    'segment': ['02-segment-cells', '02-segment-dapi', '02-segment-pp'],
    'spots': ['03-detect-spots', '04-decompose-spots', 'align', 'bud-classification', '05-assign-spots'],
    'reassign': ['05-assign-spots'],
}


# What is happening right now, for a GUI (napari-buddy sets this): progress_callback(image stem, text). Optional.
progress_callback = None

STEP_LABELS = {
    'read': 'reading the microscope files', 'grgb': 'saving channels + projections',
    '02-segment-cells': 'segmenting cells (Omnipose)', '02-segment-dapi': 'segmenting nuclei (Omnipose)',
    '02-segment-pp': 'cleaning up the masks', 'load': 'loading stacks + masks', 'focus': 'finding the in-focus slices',
    'filter': 'background filter', 'detect': 'detecting spots', '04-decompose-spots': 'decomposing dense spots',
    'align': 'aligning masks to the spots', 'bud-classification': 'bud model', '05-assign-spots': 'counting spots per cell',
}


def _report(my_image, key):
    if progress_callback is not None:
        try:
            progress_callback(getattr(my_image, 'stem', ''), STEP_LABELS.get(key, key))
        except Exception:                 # a GUI problem must not stop the processing
            pass


def _timed(my_image, key, function):
    _report(my_image, key)
    tic = time.time()
    function()
    my_image.time[key] = time.time() - tic


def read_image(params, exp, mode='serial'):
    """Step 1: read the microscope images of one field (params = an entry of exp.images). Returns its img.json."""
    my_image = Image.from_dict(params, exp)
    tic = time.time()
    _report(my_image, 'read')
    my_image.read_image()            # .vsi: fluorescence channels + DAPI
    my_image.read_cells()            # DIC
    _report(my_image, 'grgb')
    my_image.create_grgb()
    my_image.save_input_layers()     # {channel}.tif, DIC.tif and max projections
    my_image.time['01-configure'] = time.time() - tic
    my_image.time['processing_mode'] = mode
    my_image.record_settings('read')
    my_image.save_metadata('configure')
    return Path(my_image.savepath) / 'img.json'


def segment_image(json_file, exp):
    """Step 2: cell masks (DIC + DAPI composite) and nucleus masks (DAPI) with Omnipose, then clean them up."""
    my_image = Image.from_json(json_file, exp)
    my_image.load_data('dapi')                                     # only DAPI is needed here
    my_image.DIC = io.imread(Path(my_image.savepath) / 'DIC.tif')
    my_image.create_grgb()                                         # composite used by segment_cells()

    _timed(my_image, '02-segment-cells', my_image.segment_cells)
    _timed(my_image, '02-segment-dapi', my_image.segment_dapi)
    _timed(my_image, '02-segment-pp', my_image.postprocess_masks)  # -> DIC_masks_pp.tif (not yet aligned)
    my_image.record_settings('segment')
    my_image.save_metadata('segment')
    return json_file


def spots_image(json_file, exp, mode='serial'):
    """Step 3a: in-focus planes, background filter, spot detection + decomposition, align the masks to the spots."""
    my_image = Image.from_json(json_file, exp)
    _report(my_image, 'load')
    my_image.load_data('masks')
    my_image.load_data('mrna')

    tic = time.time()
    _report(my_image, 'focus')
    my_image.find_focus()
    _report(my_image, 'filter')
    my_image.filter()
    _report(my_image, 'detect')
    my_image.detect_spots()
    my_image.time['03-detect-spots'] = time.time() - tic
    _timed(my_image, '04-decompose-spots', my_image.decompose_spots)
    _timed(my_image, 'align', lambda: my_image.align(overwrite_corrected=True))   # -> corrected/ masks

    corrected = Path(my_image.savepath) / 'corrected'
    corrected.mkdir(parents=True, exist_ok=True)
    dapi_masks = Path(my_image.savepath) / 'DAPI_masks.tif'
    if not (corrected / 'DAPI_corrected_masks.tif').is_file() and dapi_masks.is_file():
        shutil.copy2(dapi_masks, corrected / 'DAPI_corrected_masks.tif')

    my_image.time['processing_mode'] = mode
    my_image.record_settings('spots')
    my_image.save_metadata('spots')
    return json_file


def _load_for_counting(json_file, exp):
    my_image = Image.from_json(json_file, exp)
    my_image.load_data('mrna')       # channel data incl. DAPI (bud model, nuclear z-projection)
    my_image.load_data('spots')      # detected spots
    my_image.load_data('dense')      # decomposed dense regions
    return my_image


def buds_and_counts(json_file, exp):
    """Step 3b: bud model (if enabled in the config) -> bud masks, then count spots per cell, mother and bud."""
    my_image = _load_for_counting(json_file, exp)
    if exp.bud_classification['enabled']:
        _timed(my_image, 'bud-classification', my_image.classify_buds)
    _timed(my_image, '05-assign-spots', my_image.assign_spots)     # -> {channel}.csv
    my_image.flag_cells()                                          # -> cell_flags.csv (napari)
    my_image.record_settings('spots')
    my_image.save_metadata('spots')
    return json_file


def reassign_image(json_file, exp):
    """Step 5: count spots again with the (hand-)corrected masks in corrected/."""
    my_image = _load_for_counting(json_file, exp)
    _timed(my_image, '05-assign-spots', my_image.assign_spots)
    my_image.flag_cells()                                          # -> cell_flags.csv (napari)
    my_image.save_metadata('spots')
    return json_file
