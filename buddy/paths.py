"""
The files of one image folder (the output folder of an image), in one place. buddy and napari-buddy both read
and write these; change a name here and nowhere else.

    from buddy.paths import ImageFiles
    f = ImageFiles(folder)
    f.channel_stack("ASH1CLB2")      -> <folder>/ASH1CLB2.tif
    f.corrected_cells                -> <folder>/corrected/DIC_corrected_masks.tif
"""
from pathlib import Path

# --- step 1: input layers
DIC = "DIC.tif"                              # 2D DIC image (DIC frame)
IMG_JSON = "img.json"                        # metadata, settings and results of every step
PROJECTIONS = "projections"                  # <stack>_max.tif: 2D maximum projections for fast viewing in napari

# --- step 2: Omnipose
CELL_MASKS_RAW = "DIC_masks.tif"             # cells as Omnipose made them (DIC frame)
CELL_MASKS = "DIC_masks_pp.tif"              # cleaned-up cells (DIC frame)
NUCLEUS_MASKS = "DAPI_masks.tif"             # nuclei (fluorescence frame)

# --- step 3: alignment of the DIC frame to the fluorescence frame
ALIGNED_DIC = "aligned_DIC.tif"
ALIGNED_CELL_MASKS = "aligned_DIC_masks.tif"

# --- corrected/: what spot counting uses (step 3 writes them, step 4 lets you correct them by hand)
CORRECTED = "corrected"
CORRECTED_CELLS = "DIC_corrected_masks.tif"  # cells, mother + bud share one label
CORRECTED_NUCLEI = "DAPI_corrected_masks.tif"
CORRECTED_BUDS = "bud_corrected_masks.tif"   # 1 = bud, 2 = background, 3 = mother

# --- bud model and flags
BUD_FLAGS = "bud_flags.csv"                  # cells the bud model was unsure about
BUD_PAIRS = "bud_pairs.csv"
BUD_MODEL_CELLS = "bud_model_cells.csv"
CELL_FLAGS = "cell_flags.csv"                # unbudded cells with high ASH1/CLB2 (late M / early S)


class ImageFiles:
    """Paths of the files of one image folder."""

    def __init__(self, folder):
        self.folder = Path(folder)

    def __truediv__(self, name):
        return self.folder / name

    # input layers
    @property
    def dic(self): return self.folder / DIC

    @property
    def img_json(self): return self.folder / IMG_JSON

    def channel_stack(self, channel): return self.folder / f"{channel}.tif"

    def projection(self, stack_name): return self.folder / PROJECTIONS / f"{Path(stack_name).stem}_max.tif"

    # per-channel spot results
    def filtered(self, channel): return self.folder / f"{channel}_filtered.tif"

    def spots(self, channel): return self.folder / f"{channel}_spots.npy"

    def spots_csv(self, channel): return self.folder / f"{channel}_spots.csv"

    def decomposed_spots(self, channel): return self.folder / f"{channel}_decomposed_spots.npy"

    def dense_regions(self, channel): return self.folder / f"{channel}_dense_regions.npy"

    def reference_spot(self, channel): return self.folder / f"{channel}_reference_spot.tif"

    def counts(self, channel): return self.folder / f"{channel}.csv"

    # masks
    @property
    def cell_masks_raw(self): return self.folder / CELL_MASKS_RAW

    @property
    def cell_masks(self): return self.folder / CELL_MASKS

    @property
    def nucleus_masks(self): return self.folder / NUCLEUS_MASKS

    @property
    def aligned_dic(self): return self.folder / ALIGNED_DIC

    @property
    def aligned_cell_masks(self): return self.folder / ALIGNED_CELL_MASKS

    @property
    def corrected(self): return self.folder / CORRECTED

    @property
    def corrected_cells(self): return self.folder / CORRECTED / CORRECTED_CELLS

    @property
    def corrected_nuclei(self): return self.folder / CORRECTED / CORRECTED_NUCLEI

    @property
    def corrected_buds(self): return self.folder / CORRECTED / CORRECTED_BUDS

    # bud model and flags
    @property
    def bud_flags(self): return self.folder / BUD_FLAGS

    @property
    def bud_pairs(self): return self.folder / BUD_PAIRS

    @property
    def bud_model_cells(self): return self.folder / BUD_MODEL_CELLS

    @property
    def cell_flags(self): return self.folder / CELL_FLAGS
