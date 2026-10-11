import logging
from buddy.utils import keep_logging
from pathlib import Path
from skimage import io
import numpy as np
import json, jsonpickle
import pandas as pd
import re

from bigfish.detection import detect_spots, decompose_dense
from bigfish.stack import remove_background_gaussian, compute_focus
from scipy.signal import savgol_filter
from scipy.ndimage import distance_transform_edt, shift as ndi_shift

from buddy.utils import translate_image, get_regionprops, expand_masks
from buddy.utils import find_in_focus_indices, find_high_density_patch
from buddy.utils import spot_assignment
from buddy import gpu_filters

from .Masks_alignment import *
from .utils import *

jsonpickle.handlers.register('pandas')

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s ', datefmt='%m/%d/%Y %I:%M:%S%p',
                    level=logging.INFO)


def access():
    logging.info("OK")


def save_projection(stack_file, stack):
    """
    Save the maximum projection of a z-stack as projections/<name>_max.tif next to it. napari-buddy shows
    these 2D projections when an image is opened (much faster than reading the full stacks).
    """
    stack_file = Path(stack_file)
    if np.ndim(stack) != 3:
        return
    folder = stack_file.parent / 'projections'
    folder.mkdir(exist_ok=True)
    io.imsave(folder / f'{stack_file.stem}_max.tif', np.asarray(stack).max(axis=0), check_contrast=False)


def _md5(path):
    """Checksum of a file (used to recognise whether a mask file was edited after buddy wrote it)."""
    import hashlib
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


class Image:

    def __init__(self, exp, img):
        """
        Class for a FISH image.

        Parameters
        ----------
        - exp : an Experiment object containing at least
            - channel configuration
            - spot detection parameters
            - input and output directories

        - img : a dict containing at least two keys
            - vsi_file : the VSI file with the fluorescent channels
            - cell_file : the file with the cell outline data, e.g. a 2D DIC picture

            Optional keys:
            - segmentation: segmentation parameters
            - crop_by: cropping coordinates
            - translate_by: cell file translation vector


        Returns
        -------
        a FISH image object.
        """

        setattr(self, "experiment", exp)
        setattr(self, "parameters", exp.parameters)
        setattr(self, "time", {})
        setattr(self, "results", {})
        for key, value in img.items():
            setattr(self, key, value)


        if 'savepath' not in img:
            try:
                stem = re.sub(r'_CY[A-Z0-9\s,._]+DAPI', '', Path(img['vsi_file']).stem)
                stem = re.sub(r'\.ome$', '', stem)              # foo.ome.tif -> foo
                self.stem = stem
                self.savepath = str(Path(exp.outputdir) / stem)
                Path(self.savepath).mkdir(parents=True, exist_ok=True)
                logging.info(f"created output dir {self.savepath}")
            except:
                logging.warning(f"failed to create output dir {Path(exp.outputdir) / stem}")
        else:
            try:
                if Path(img['savepath']).is_dir():
                    logging.info(f"found output dir {img['savepath']}")
            except FileNotFoundError:
                logging.warning(f"could not find output dir {img['savepath']}")


    @classmethod
    def from_json(cls, filename, exp):
        img = json.load(open(filename))
        return cls(exp, img)


    @classmethod
    def from_dict(cls, params, exp):
        return cls(exp, params)

    def read_cells(self):
        if getattr(self, 'cell_file', None) is None and getattr(self, '_dic_from_stack', None) is not None:
            logging.info('cell image: the DIC channel of the TIFF stack')
            self.cells = self._dic_from_stack
        else:
            logging.info(f'reading cell image: {self.cell_file}')
            self.cells  = io.imread(Path(self.experiment.inputdir) / self.cell_file)
        self.DIC = self.cells   # create_grgb() / align() read self.DIC


    def read_image(self):
        """The fluorescence channels: from the .vsi file, or from a multi-page TIFF (see buddy/tiff_input.py)."""
        if str(self.vsi_file).lower().endswith(('.tif', '.tiff')):
            from buddy.tiff_input import read_stack
            logging.info(f'reading fluorescence image (TIFF stack): {self.vsi_file}')
            channels, self._dic_from_stack = read_stack(Path(self.experiment.inputdir) / self.vsi_file, self.experiment)
            self.mrna = {mrna: {"data": data} for mrna, data in channels.items()}
            for mrna in self.mrna:
                if mrna != 'DAPI':
                    self.results[mrna] = {}
            return
        # heavy import (starts the Bio-Formats Java bridge); load only when reading a VSI
        from bioio import BioImage
        import bioio_bioformats
        img = {}

        logging.info(f'reading fluorescence image: {self.vsi_file}')
        image = BioImage(Path(self.experiment.inputdir) / self.vsi_file, reader=bioio_bioformats.Reader)
        logging.info(f'..found scenes: {image.scenes}')
        for s in image.scenes:
            if s == 'macro image':
                logging.info(f"....ignoring scene '{s}'")
            else:
                image.set_scene(s)
                logging.info(f"....reading scene '{s}' (shape {image.shape})")
                logging.info(f"....found channels {image.channel_names}")
                self.mrna = {}
                # parse the channel/filter names from the scene name (strip stray whitespace)
                tokens = [t.strip() for t in s.replace(r'001 ', '').split(",")]
                logging.info(f"....channel tokens from scene name: {tokens}")
                logging.info(f"....configured filters: {list(self.experiment.filter2mRNA.keys())}")
                for c, token in enumerate(tokens):
                    if token in self.experiment.filter2mRNA:
                        mrna = self.experiment.filter2mRNA[token]
                        logging.info(f"......reading in channel: C={c} filter={token} mrna={mrna}")
                        img = image.get_image_data("ZYX", C=c)
                        self.mrna[mrna] = {"data": img}
                        if mrna != 'DAPI':
                            self.results[mrna] = {}
                    else:
                        logging.warning(f"......ignoring unknown channel token '{token}' "
                                        f"(not a 'filter' in the config)")


    def load_data(self, layer):
        if layer == 'masks':
            try:
                self.cell_masks = io.imread(Path(self.savepath) / "corrected" / "DIC_corrected_masks.tif")
            except FileNotFoundError:
                self.cell_masks = None
            except Exception as e:
                self.cell_masks = None
            self.cell_masks_uncorrected = io.imread(Path(self.savepath) / "DIC_masks_pp.tif")

            try:
                self.bud_masks = io.imread(Path(self.savepath) / "corrected" / "bud_corrected_masks.tif")
            except FileNotFoundError:
                self.bud_masks = None
            except Exception as e:
                self.bud_masks = None

            try:
                self.cell_masks_aligned = io.imread(Path(self.savepath,'aligned_DIC_masks.tif'))
            except FileNotFoundError:
                self.cell_masks_aligned = None
            except Exception as e:
                self.cell_masks_aligned = None


            try:
                self.dapi_masks = io.imread(Path(self.savepath) / "corrected" / "DAPI_corrected_masks.tif")
            except FileNotFoundError:
                self.dapi_masks = None
            except Exception as e:
                self.dapi_masks = None
            self.dapi_masks_uncorrected = io.imread(Path(self.savepath) / "DAPI_masks.tif")

        elif layer == 'mrna':
            setattr(self, 'mrna', {})
            for ch in self.experiment.channels.keys():
                self.mrna[ch] = {}
                self.mrna[ch]['data'] = io.imread(Path(self.savepath) / f'{ch}.tif')
        elif layer == 'dapi':
            # only the DAPI stack (all that segmentation needs; much faster than reading every channel)
            self.mrna = {'DAPI': {'data': io.imread(Path(self.savepath) / 'DAPI.tif')}}
        elif layer =='spots':
            for ch in self.experiment.channels.keys():
                if ch !='DAPI':
                    self.mrna[ch]['spots'] = np.load(Path(self.savepath) / f'{ch}_spots.npy')
        elif layer =='dense':
            for ch in self.experiment.channels.keys():
                if ch !='DAPI':
                    self.mrna[ch]['dense_regions'] = np.load(Path(self.savepath) / f'{ch}_dense_regions.npy')

        elif layer == 'DIC':
            self.DIC = io.imread(Path(self.savepath) / "DIC.tif")
            self.aligned_DIC = io.imread(Path(self.savepath) / "aligned_DIC.tif")

    # ------------------------------------------------------------------
    # helper: read a tif if it exists, else return None
    # ------------------------------------------------------------------
    def _imread_or_none(self, *path_parts):
        p = Path(self.savepath, *path_parts)
        try:
            return io.imread(p)
        except (FileNotFoundError, OSError):
            return None

    # ------------------------------------------------------------------
    # helper: pad a 2D array into the top-left of a (target_h, target_w) canvas
    # ------------------------------------------------------------------
    @staticmethod
    def _pad_to(arr, target_h, target_w):
        out = np.zeros((target_h, target_w), dtype=arr.dtype)
        out[:arr.shape[0], :arr.shape[1]] = arr
        return out

    def align(self, overwrite_corrected=False, n_iter=4000, max_shift=200,
              radius=20, cutoff=50):
        """
        Estimate and correct the translation between the DIC frame (DIC image +
        cell masks) and the fluorescence frame (in which the spots were
        detected), using the detected spots.

        The spots live in the fluorescence coordinate frame, the cell masks in
        the DIC frame; the two are offset by a small translation. We estimate
        the (dy, dx) that maps the spots onto the cell masks, then apply the
        inverse shift to the cell masks and the DIC image so they line up with
        the spots.

        Only the DIC image and the cell masks are shifted. The DAPI masks are
        already in the fluorescence frame (segmented from the DAPI channel), and
        the bud masks are handled separately, so neither is touched here.

        Parameters
        ----------
        overwrite_corrected : bool
            If True, also update corrected/DIC_corrected_masks.tif (what
            `assign_spots` reads). If it does not exist yet, it becomes the shifted
            Omnipose masks. If it exists and was aligned before (e.g. corrected by
            hand in napari), it is only moved by the change in shift since then
            (normally none), so running this step again never shifts it twice.
            If False, only aligned_*.tif copies are written.

        The shift is always estimated on the Omnipose masks (DIC_masks_pp.tif, DIC
        frame) with a fixed random seed, so it is the same every time it is run.
        n_iter, max_shift, radius, cutoff :
            Search parameters passed through to estimate_alignment_shift.
        """
        logging.info("aligning DIC / cell masks to detected spots")

        # --- 1. gather spots (prefer in-memory, else load from disk) ----------
        spots = []
        for ch in self.experiment.channels.keys():
            if ch == 'DAPI':
                continue
            if getattr(self, 'mrna', None) and ch in self.mrna and 'spots' in self.mrna[ch]:
                spots.append(np.asarray(self.mrna[ch]['spots'])[:, :3])
            else:
                p = Path(self.savepath) / f'{ch}_spots.npy'
                if p.is_file():
                    spots.append(np.load(p)[:, :3])
        if not spots:
            logging.warning("..no spots found, skipping alignment")
            return
        spots = np.concatenate(spots).astype(float)

        # --- 2. load the DIC-frame layers -------------------------------------
        # always the Omnipose masks (DIC frame): corrected/ masks may already be in the
        # fluorescence frame after an earlier run, and shifting those again would move them twice
        previous = self.results.get('alignment')
        cell_masks = self._imread_or_none('DIC_masks_pp.tif')
        cell_source = 'pp'
        if cell_masks is None:
            logging.warning("..no cell masks found, skipping alignment")
            return

        dic = self._imread_or_none('DIC.tif')

        # --- 3. common canvas so spot coords index validly into every layer ---
        shapes = [cell_masks.shape]
        if dic is not None:
            shapes.append(dic.shape)
        target_h = max([s[0] for s in shapes] + [int(spots[:, 1].max()) + 1])
        target_w = max([s[1] for s in shapes] + [int(spots[:, 2].max()) + 1])

        cell_masks = self._pad_to(cell_masks, target_h, target_w)
        if dic is not None:
            dic = self._pad_to(dic, target_h, target_w)

        # --- 4. estimate the shift from spots -> cell masks -------------------
        mask_shift, spot_shift = estimate_alignment_shift(
            spots, cell_masks,
            n_iter=n_iter, max_shift=max_shift, radius=radius, cutoff=cutoff,
            rng=np.random.default_rng(0)        # same shift every run
        )
        dy, dx = mask_shift
        logging.info("..spots->mask shift %s, applying (dy=%d, dx=%d) to DIC + cell masks "
                     "(cell mask source: %s)", spot_shift, dy, dx, cell_source)

        # record for reproducibility (saved by save_metadata via 'results')
        self.results['alignment'] = {
            'mask_shift': [int(dy), int(dx)],
            'spot_shift': [int(spot_shift[0]), int(spot_shift[1])],
            'cell_mask_source': cell_source,
            'target_shape': [int(target_h), int(target_w)],
            'n_spots': int(len(spots)),
        }

        # --- 5. apply the shift to the cell masks and the DIC image -----------
        self.cell_masks_aligned = ndi_shift(cell_masks, shift=(dy, dx), order=0,
                                            mode='constant', cval=0)
        io.imsave(Path(self.savepath, 'aligned_DIC_masks.tif'), self.cell_masks_aligned, check_contrast=False)

        if dic is not None:
            self.aligned_DIC = ndi_shift(dic, shift=(dy, dx), order=0,
                                         mode='constant', cval=0)
            io.imsave(Path(self.savepath, 'aligned_DIC.tif'), self.aligned_DIC, check_contrast=False)

        # --- 6. optionally update the corrected cell mask (assign_spots reads it)
        if overwrite_corrected:
            corrected_dir = Path(self.savepath) / 'corrected'
            corrected_dir.mkdir(parents=True, exist_ok=True)
            dst = corrected_dir / 'DIC_corrected_masks.tif'
            if not dst.is_file():
                io.imsave(dst, self.cell_masks_aligned, check_contrast=False)
                logging.info("..wrote aligned cell masks to corrected/")
            elif previous is not None and previous.get('cell_mask_source') == 'pp':
                # already in the fluorescence frame: move only by the change in shift (normally 0)
                ddy, ddx = dy - previous['mask_shift'][0], dx - previous['mask_shift'][1]
                if ddy or ddx:
                    corrected = self._pad_to(io.imread(dst), target_h, target_w)
                    io.imsave(dst, ndi_shift(corrected, shift=(ddy, ddx), order=0, mode='constant', cval=0), check_contrast=False)
                    logging.info(f"..moved corrected cell masks by the change in shift ({ddy}, {ddx})")
                else:
                    logging.info("..corrected cell masks already aligned, left as they are")
            elif previous is None:
                # corrected masks made before any alignment (DIC frame): shift them like the pp masks
                corrected = self._pad_to(io.imread(dst), target_h, target_w)
                io.imsave(dst, ndi_shift(corrected, shift=(dy, dx), order=0, mode='constant', cval=0), check_contrast=False)
                logging.info("..shifted existing corrected cell masks into the fluorescence frame")
            else:
                logging.warning("..corrected/DIC_corrected_masks.tif was aligned by an older buddy version "
                                "(shift relative to earlier corrected masks) - left as it is; delete it to rebuild "
                                "it from the Omnipose masks")


    def classify_buds(self):
        """
        Automatic mother/bud assignment with the trained bud model (see buddy/bud_model.py).
        Run after align(), before assign_spots(). Settings: "bud_classification" in the config file.

        Uses the Omnipose cells (DIC_masks_pp.tif) moved into the fluorescence frame with the shift
        found by align(), the aligned DIC, the DAPI max projection and - for the "with_ASH1CLB2"
        model - the ASH1/CLB2 max projection. Writes:
          corrected/DIC_corrected_masks.tif  cells, each bud merged into its mother (same label)
          corrected/bud_corrected_masks.tif  1 = bud, 2 = background, 3 = mother
          bud_flags.csv                      cells worth checking in napari (with the reason)
          bud_pairs.csv, bud_model_cells.csv the pairs found / the model output per cell
        Bud masks you corrected by hand are never overwritten (unless "overwrite_manual_corrections": true).
        """
        settings = self.experiment.bud_classification
        corrected = Path(self.savepath) / 'corrected'
        # a bud mask that is not exactly the file the model wrote last time was made or edited by hand
        bud_file = corrected / 'bud_corrected_masks.tif'
        written_by_model = self.results.get('bud_classification', {}).get('bud_mask_md5')
        if bud_file.is_file() and _md5(bud_file) != written_by_model and not settings['overwrite_manual_corrections']:
            logging.warning('..corrected/bud_corrected_masks.tif was made or edited by hand - not overwritten '
                            '(set "overwrite_manual_corrections": true to replace it)')
            return
        model = getattr(self.experiment, 'bud_model', None)
        if model is None:
            self.experiment.init_bud_model()
            model = self.experiment.bud_model
        logging.info(f'classifying buds (model: {model.name})')

        # Omnipose cells and DIC, in the fluorescence frame (same shift and canvas as align())
        alignment = self.results.get('alignment')
        if alignment is None:
            logging.warning('..no alignment found - run align() first; skipping bud classification')
            return
        h, w = alignment['target_shape']
        dy, dx = alignment['mask_shift']
        cells = self._pad_to(io.imread(Path(self.savepath) / 'DIC_masks_pp.tif'), h, w)
        cells = ndi_shift(cells, shift=(dy, dx), order=0, mode='constant', cval=0)
        dic = self._imread_or_none('aligned_DIC.tif')
        if dic is None:
            dic = ndi_shift(self._pad_to(io.imread(Path(self.savepath) / 'DIC.tif'), h, w), shift=(dy, dx), order=0)

        def max_projection(channel):
            if getattr(self, 'mrna', None) and channel in self.mrna and 'data' in self.mrna[channel]:
                return np.max(self.mrna[channel]['data'], axis=0)
            return np.max(io.imread(Path(self.savepath) / f'{channel}.tif'), axis=0)

        dapi = max_projection('DAPI')
        ash = max_projection(settings['ash_channel']) if model.uses_ash else None

        result = model.classify(dic, dapi, cells, ash)

        corrected.mkdir(parents=True, exist_ok=True)
        io.imsave(corrected / 'DIC_corrected_masks.tif', result['cell_masks'].astype(np.uint16), check_contrast=False)
        io.imsave(bud_file, result['bud_masks'], check_contrast=False)
        result['flags'].to_csv(Path(self.savepath) / 'bud_flags.csv', index=False)
        result['pairs'].to_csv(Path(self.savepath) / 'bud_pairs.csv', index=False)
        result['cells'].to_csv(Path(self.savepath) / 'bud_model_cells.csv', index=False)

        self.results['bud_classification'] = {
            'model': model.name,
            'cut_off': model.cut_off,
            'device': model.device,
            'dic_rotation_k': int(result['dic_rotation']),
            'n_cells': int(len(result['cells'])),
            'n_pairs': int(len(result['pairs'])),
            'n_flags': int(len(result['flags'])),
            'bud_mask_md5': _md5(bud_file),      # lets a later run recognise hand-made edits
        }
        logging.info(f"..{len(result['pairs'])} mother-bud pairs, {len(result['flags'])} cells flagged for checking")


    def flag_cells(self):
        """
        Flags for napari (buddy/cell_flags.py): unbudded cells with high ASH1/CLB2 - late-M pairs and possible
        newborn daughters / early-S cells with an unmarked bud. Run after assign_spots(). Writes cell_flags.csv
        (shown in napari's 'check these buds' layer). Changes nothing else. Settings: "cell_flags" in the config.
        """
        from buddy.cell_flags import compute_flags
        settings = dict(self.experiment.cell_flags)
        out = Path(self.savepath) / 'cell_flags.csv'
        if not settings['enabled']:
            return
        ch = settings['ash_channel'] or self.experiment.bud_classification['ash_channel']
        table = Path(self.savepath) / f'{ch}.csv'
        masks = Path(self.savepath) / 'corrected' / 'DIC_corrected_masks.tif'
        if not table.is_file() or not masks.is_file():
            logging.info(f'..no {ch} channel counts - no cell flags')
            if out.is_file():
                out.unlink()
            return
        flags = compute_flags(pd.read_csv(table), io.imread(masks), settings)
        flags.to_csv(out, index=False)
        self.results['cell_flags'] = {'ash_channel': ch, 'n_flags': int(len(flags)),
                                      **{k: settings[k] for k in ('fold', 'nucleus_area', 'nucleus_intensity', 'max_gap_px')}}
        logging.info(f'..{len(flags)} cells flagged for high {ch} (cell_flags.csv)')


    def create_grgb(self):
        logging.info(f'creating GRGB composite image')
        cells = self.DIC
        dapimaxproj = np.max(self.mrna['DAPI']['data'], axis=0)
        self.grgb = np.stack([np.zeros(cells.shape), cells, dapimaxproj, np.zeros(cells.shape)], axis=0)


    def save_input_layers(self):
        from bioio_ome_tiff.writers import OmeTiffWriter
        logging.info(f'saving layers to: {self.savepath}')
        # fluorescent channels
        for ch in self.mrna.keys():
            data = self.mrna[ch]['data']
            savepath = Path(self.savepath) / f'{ch}.tif'
            OmeTiffWriter.save(data, savepath, dim_order="ZYX", channel_names=[ch])
            save_projection(savepath, data)          # 2D projection for fast viewing in napari
            logging.info(f'..saving {ch} channel to {savepath}')

        # cell channel
        data = self.cells
        savepath = Path(self.savepath) / 'DIC.tif'
        io.imsave(savepath, data, check_contrast=False)
        logging.info(f'..saving DIC channel to {savepath}')


    def record_settings(self, step):
        """
        Record in img.json (results -> settings -> step) the config settings this step used ('read',
        'segment' or 'spots'), so a later run can tell whether the image is already processed with the same
        settings (see buddy/rerun_check.py). Also updates the 'parameters' saved in img.json to the current config.
        """
        from buddy.rerun_check import step_settings
        self.results.setdefault('settings', {})[step] = step_settings(self.experiment, step)
        self.parameters = dict(self.experiment.parameters)


    def save_metadata(self, stage):
        logging.info(f'saving metadata to file: {Path(self.savepath) / "img.json"}')

        keys = [
            'vsi_file',
            'cell_file',
            'stem',
            'savepath',
            'parameters',
            'segmentation',
            'results',
            'time'
        ]
        metadata = { i: getattr(self, i) for i in keys if hasattr(self, i)}
        metadata['last_run'] = stage

        with open(Path(self.savepath) / "img.json", "w") as f:
            json.dump(metadata, f, default=vars, indent=4)

        return metadata


    def save(self, suffix):
        logging.info(f'saving image to: {self.savepath}/img.{suffix}.pkl')

        with open(Path(self.savepath) / f'img.{suffix}.pkl', "w") as f:
            f.write(jsonpickle.encode(self))


    def unpickle(pickle_file):
        with open(pickle_file, "r+") as f:
            return jsonpickle.decode(f.read())


    # redo for list of images to avoid loading the models for each image
    def segment_cells(self):
        # heavy import (pulls in torch); load only when segmenting
        with keep_logging():
            from cellpose_omni import models
        logging.info(f'segmenting DIC image')

        # reuse the model loaded by Experiment.init_omnipose() if available,
        # otherwise load one here (slower: re-loaded for every image)
        use_GPU = getattr(self.experiment, 'use_GPU', False)
        model_type = "cyto2_omni"
        model = getattr(self.experiment, 'cell_model', None)
        if model is None:
            model = models.CellposeModel(gpu=use_GPU, model_type=model_type)

        if self.experiment.parameters["organism"] == 'E.coli': 

            params = {
                'channels': [1, 2],  # always define this with the model
                'rescale': None,  # upscale or downscale your images, None = no rescaling
                'mask_threshold': 0,  # erode or dilate masks with higher or lower values between -5 and 5
                'flow_threshold': 0,
                'min_size': 20,
                'diameter': 0,
                'invert': False,
                'transparency': True,  # transparency in flow output
                'omni': True,  # we can turn off Omnipose mask reconstruction, not advised
                'cluster': True,  # use DBSCAN clustering
                'resample': True,  # whether or not to run dynamics on rescaled grid or original grid
                'verbose': False,  # turn on if you want to see more output
                'tile': False,  # average the outputs from flipped (augmented) images; slower, usually not needed
                'niter': None,
                # default None lets Omnipose calculate # of Euler iterations (usually <20) but you can tune it for over/under segmentation
                'augment': False,  # Can optionally rotate the image and average network outputs, usually not needed
                'affinity_seg': False,  # new feature, stay tuned...
            }
        

        elif self.experiment.parameters["organism"] == 'S.cerevisiae': 

            params = {
                'channels': [1, 2],  # always define this with the model
                'rescale': None,  # upscale or downscale your images, None = no rescaling
                'mask_threshold': -3,  # erode or dilate masks with higher or lower values between -5 and 5
                'flow_threshold': 1,
                'min_size': 10,
                'diameter': 50,
                'invert': False,
                'transparency': True,  # transparency in flow output
                'omni': True,  # we can turn off Omnipose mask reconstruction, not advised
                'cluster': True,  # use DBSCAN clustering
                'resample': True,  # whether or not to run dynamics on rescaled grid or original grid
                'verbose': False,  # turn on if you want to see more output
                'tile': False,  # average the outputs from flipped (augmented) images; slower, usually not needed
                'niter': None,
                # default None lets Omnipose calculate # of Euler iterations (usually <20) but you can tune it for over/under segmentation
                'augment': False,  # Can optionally rotate the image and average network outputs, usually not needed
                'affinity_seg': False,  # new feature, stay tuned...
            }

        # allow the cell-segmentation minimum object size to be set from the config
        params['min_size'] = self.experiment.parameters.get('min_size', params['min_size'])

        mask, flow, style = model.eval(self.grgb, **params)
        logging.info(f'Found {np.max(mask)} cells')


        self.cell_masks = mask
        if not hasattr(self, 'segmentation'):
            self.segmentation = {'cells': {}, 'dapi': {}}
        self.segmentation['cells'] = params
        self.segmentation['cells']['model_type'] = model_type
        maskfile_latest = Path(self.savepath) / f'DIC_masks.model={model_type}_chan={str(params["channels"]).replace(" ", "")}_diameter={params["diameter"]}_minsize={params["min_size"]}_mask={params["mask_threshold"]}_flow={params["flow_threshold"]}.tif'
        io.imsave(maskfile_latest, mask, check_contrast=False)
        logging.info(f"writing cell mask to {maskfile_latest}")


    def segment_dapi(self):
        # heavy import (pulls in torch); load only when segmenting
        with keep_logging():
            from cellpose_omni import models
        logging.info(f'segmenting DAPI image')

        # check if the object contains the grgb data,
        # otherwise load it
        DAPIzproj = np.amax(self.mrna['DAPI']['data'],axis=0)

        # This checks to see if you have set up your GPU properly.
        # CPU performance is a lot slower, but not a problem if you
        # are only processing a few images.
        #use_GPU = use_gpu()
        use_GPU = getattr(self.experiment, 'use_GPU', False)
        # model for nuclei
        model_type = "nuclei"
        model = getattr(self.experiment, 'nuclei_model', None)
        if model is None:
            model = models.CellposeModel(gpu=use_GPU, model_type=model_type)

        # pass parameters to model
        params = {
            'channels': [0, 0], # always define this with the model
            'rescale': None, # upscale or downscale your images, None = no rescaling
            'mask_threshold': 0.0, # erode or dilate masks with higher or lower values between -5 and 5
            'flow_threshold': 0.0,
            'min_size': 10,
            'diameter': 0.0,
            'invert': False,
            'transparency': True, # transparency in flow output
            'omni': True, # we can turn off Omnipose mask reconstruction, not advised
            'cluster': True, # use DBSCAN clustering
            'resample': True, # whether or not to run dynamics on rescaled grid or original grid
            'verbose': False, # turn on if you want to see more output
            'tile': False, # average the outputs from flipped (augmented) images; slower, usually not needed
            'niter': None, # default None lets Omnipose calculate # of Euler iterations (usually <20) but you can tune it for over/under segmentation
            'augment': False, # Can optionally rotate the image and average network outputs, usually not needed
            'affinity_seg': False, # new feature, stay tuned...
        }

        # nuclei are segmented with the 'nuclei' model and the DAPI params above.
        # (previously these were overwritten with the *cell* params - a bug.)
        if not hasattr(self, 'segmentation'):
            self.segmentation = {'cells': {}, 'dapi': {}}
        logging.info(f"nuclei model_type: {model_type}, channels: {params['channels']}")

        
        # allow the nucleus-segmentation minimum object size to be set from the config
        params['min_size'] = self.experiment.parameters.get('nuclei_min_size', params['min_size'])

        mask, flow, style = model.eval(DAPIzproj, **params)
        logging.info(f'Found {np.max(mask)} cells')

        self.dapi_masks = mask
        self.segmentation['dapi'] = params
        self.segmentation['dapi']['model_type'] = model_type


        maskfile_latest = Path(self.savepath) / f'DAPI_masks_model={model_type}_chan={str(params["channels"]).replace(" ", "")}_diameter={params["diameter"]}_minsize={params["min_size"]}_mask={params["mask_threshold"]}_flow={params["flow_threshold"]}.tif'
        io.imsave(maskfile_latest, mask, check_contrast=False)
        cellmaskfile = Path(self.savepath) / 'DAPI_masks.tif'
        io.imsave(cellmaskfile, mask, check_contrast=False)
        logging.info(f"writing DAPI mask to {maskfile_latest}")


    def postprocess_masks(self):
        # min_cell_area and expand_px are read from the config 'parameters' block
        # (with defaults) so they can be tuned without editing this code; the
        # values actually used are recorded in img.json's segmentation block.
        min_cell_area = self.experiment.parameters.get('min_cell_area', 20)
        expand_px     = self.experiment.parameters.get('expand_px', 0)
        remove_edge   = bool(self.experiment.parameters.get('remove_edge_cells', False))
        logging.info(f'postprocessing masks (min_cell_area={min_cell_area}, expand_px={expand_px}, '
                     f'remove_edge_cells={remove_edge})')

        logging.info(f'..removing cell masks with no DAPI')
        self.cell_masks_with_dapi = remove_cell_masks_with_no_dapi(self.cell_masks, self.dapi_masks)[0]

        logging.info(f'..discarding masks of size < {min_cell_area}')
        self.regionprops = get_regionprops(self.cell_masks)
        self.cell_masks_by_area = filter_cells_by_area(self.cell_masks_with_dapi, self.regionprops, min_cell_area=min_cell_area)[0]

        n_edge = 0
        if remove_edge:
            # cells cut off by the image border are incomplete (too few RNAs, bud possibly outside the image)
            from skimage.segmentation import clear_border
            before = set(np.unique(self.cell_masks_by_area)) - {0}
            self.cell_masks_by_area = clear_border(self.cell_masks_by_area)
            n_edge = len(before - set(np.unique(self.cell_masks_by_area)))
            logging.info(f'..removed {n_edge} cell(s) touching the image edge')

        logging.info(f'..removing DAPI masks with no cell')
        self.dapi_masks, _ = remove_dapi_masks_with_no_cell(self.dapi_masks, self.cell_masks_by_area)
        io.imsave(Path(self.savepath) / 'DAPI_masks.tif', self.dapi_masks, check_contrast=False)

        #logging.info(f'..discarding large, round masks (min size 1000, min excentricity 0.8')
        #self.cell_masks_by_shape = filter_cells_by_shape(self.cell_masks_by_area, self.regionprops,
        #                                                 min_clump_area=200, max_clump_eccentricity=0.1)[0]

        # optionally grow each cell outward (0 = off); overwrites the same file
        if expand_px:
            logging.info(f'..expanding cell masks by {expand_px} px')
            self.cell_masks_by_area = expand_masks(self.cell_masks_by_area, expand_px)

        # record what was actually used so it lands in img.json
        if not hasattr(self, 'segmentation'):
            self.segmentation = {'cells': {}, 'dapi': {}}
        self.segmentation['postprocess'] = {'min_cell_area': min_cell_area, 'expand_px': expand_px,
                                            'remove_edge_cells': remove_edge, 'edge_cells_removed': n_edge}

        cellmaskfile = Path(self.savepath) / 'DIC_masks_pp.tif'
        io.imsave(cellmaskfile, self.cell_masks_by_area, check_contrast=False)


    def find_focus(self):
        logging.info(f'finding focus')
        self.selected_patch = find_high_density_patch(self.cell_masks_uncorrected, patch_size=self.experiment.parameters['patch_size'])

        for ch in self.mrna.keys():
            if ch != 'DAPI':
                self.find_focus_channel(ch)


    def find_focus_channel(self, ch):
        mrna_data = self.mrna.get(ch)['data']
        self.results[ch] = {
            'colormap': self.experiment.channels[ch]['colormap']
        }

        img_patch = mrna_data[:,
                    self.selected_patch[0]:self.selected_patch[0] + self.experiment.parameters['patch_size'][0],
                    self.selected_patch[1]:self.selected_patch[1] + self.experiment.parameters['patch_size'][1]
                    ]
        focus = compute_focus(img_patch)
        projected_focus = np.max(focus, axis=(1, 2))
        projected_focus_smoothed = savgol_filter(projected_focus, 16, 2, 0)
        ifx_1, ifx_2 = find_in_focus_indices(projected_focus_smoothed)

        if ifx_1 < 0 or ifx_2 > mrna_data.shape[0]:
            logging.warning(f'....focus detection: max focus is too close to highest or lowest slice')
        ifx_1 = int(max(ifx_1, 0))
        ifx_2 = int(min(ifx_2, mrna_data.shape[0]))

        self.results[ch].update({
            'z_max_focus': int(np.argmax(projected_focus_smoothed)),
            'ifx_1': ifx_1,
            'ifx_2': ifx_2
        })
        logging.info(f'..channel {ch}: [{ifx_1}, {ifx_2}], max focus slice {self.results[ch]["z_max_focus"]}')


    def filter(self):
        logging.info(f'filtering: remove_background_gaussian')
        for ch in self.mrna.keys():
            if ch != 'DAPI':
                self.filter_channel(ch)


    def filter_channel(self, ch):
        """
        Background removal (Big-FISH remove_background_gaussian), only on the in-focus slices found by
        find_focus() plus a margin as wide as the Gaussian's reach in z, so the result inside the in-focus
        range is identical to filtering the whole stack (spot detection only uses those slices).
        Slices outside the range stay 0 - exactly as they were saved before.
        """
        data = self.mrna.get(ch)
        sigma = self.experiment.parameters['sigma']
        n_z = data['data'].shape[0]
        ifx_1, ifx_2 = self.results.get(ch, {}).get('ifx_1', 0), self.results.get(ch, {}).get('ifx_2', n_z)
        margin = int(4 * sigma[0] + 0.5) + 1          # Gaussian kernel reach in z (truncate = 4 sigma) + 1
        lo, hi = max(ifx_1 - margin, 0), min(ifx_2 + margin, n_z)
        filtered = np.zeros_like(data['data'])
        filtered[ifx_1:ifx_2] = self._run_on_gpu_or_cpu(
            ch, 'filter',
            gpu=lambda device: gpu_filters.remove_background(data['data'][lo:hi], sigma, device=device),
            cpu=lambda: remove_background_gaussian(data['data'][lo:hi], sigma=sigma))[ifx_1 - lo:ifx_2 - lo]
        data['filtered'] = filtered
        logging.info(f'..channel {ch} (slices {ifx_1}-{ifx_2}, {self.results[ch]["filter_backend"]})')


    def _gpu_device(self):
        """'cuda' when the GPU versions of the background filter and spot detection should be used, else None.
        Controlled by "gpu_spot_detection" in the parameters block (default true) and "use_gpu"."""
        if not hasattr(self, '_filter_device'):
            from buddy.experiment import gpu_available
            wanted = self.experiment.parameters.get('gpu_spot_detection', True) and getattr(self.experiment, 'use_gpu', True)
            self._filter_device = 'cuda' if wanted and gpu_available(True) else None
        return self._filter_device


    def _run_on_gpu_or_cpu(self, ch, step, gpu, cpu):
        """Run gpu(device) when a GPU is available (falls back to Big-FISH on the CPU if the GPU runs out of
        memory), otherwise cpu(). Records which one was used in results[ch][f'{step}_backend']."""
        self.results.setdefault(ch, {})
        device = self._gpu_device() if gpu is not None else None
        if device is not None:
            try:
                result = gpu(device)
                self.results[ch][f'{step}_backend'] = 'gpu'
                return result
            except RuntimeError as e:                       # includes torch.cuda.OutOfMemoryError
                import torch
                torch.cuda.empty_cache()
                logging.warning(f'..channel {ch}: GPU {step} failed ({e.__class__.__name__}), using Big-FISH on the CPU '
                                f'(lower buddy.gpu_filters.BLOCK_ROWS if the GPU runs out of memory)')
        self.results[ch][f'{step}_backend'] = 'cpu (big-fish)'
        return cpu()


    def detect_spots(self):
        logging.info(f'detecting spots')
        for ch in self.mrna.keys():
            if ch != 'DAPI':
                self.detect_spots_channel(ch)


    def detect_spots_channel(self, ch):
        mrna_data = self.mrna.get(ch)['data']
        mrna_filtered = self.mrna.get(ch)['filtered']
        ifx_1, ifx_2 = self.results[ch]['ifx_1'], self.results[ch]['ifx_2']
        mrna_filtered_selected = mrna_filtered[ifx_1:ifx_2, ...]
        threshold = self.experiment.channels[ch]['threshold']
        scale, spot_radius = self.experiment.parameters['scale'], self.experiment.parameters['spot_radius']
        # threshold None = Big-FISH's automatic threshold (both versions)
        spots, threshold = self._run_on_gpu_or_cpu(
            ch, 'detection',
            gpu=lambda device: gpu_filters.detect_spots(mrna_filtered_selected, threshold, scale, spot_radius, device=device),
            cpu=lambda: detect_spots(mrna_filtered_selected, threshold=threshold, voxel_size=scale,
                                     spot_radius=spot_radius, return_threshold=True))
        self.results[ch]['threshold'] = threshold

        # always elegant:
        filtered_padded_intensities = np.concatenate((np.zeros([ifx_1, mrna_data.shape[1], mrna_data.shape[2]]).astype(
            'uint16'), mrna_filtered_selected, np.zeros(
            [mrna_data.shape[0] - ifx_2, mrna_data.shape[1], mrna_data.shape[2]]).astype('uint16')), axis=0)
        io.imsave(Path(self.savepath) / f'{ch}_filtered.tif', filtered_padded_intensities, check_contrast=False)
        save_projection(Path(self.savepath) / f'{ch}_filtered.tif', mrna_filtered_selected)   # for napari

        # restore z-level
        spots[:, 0] = spots[:, 0] + ifx_1

        # adjustable out-of focus filtering
        #  we remove the bottom two slices because their detected spots look like noise:
        spots = spots[spots[:, 0] > ifx_1 + 2]

        spot_intensities = mrna_data[spots[:,0], spots[:,1], spots[:,2]].reshape(-1,1)
        filtered_spot_intensities = mrna_filtered[spots[:,0], spots[:,1], spots[:,2]].reshape(-1,1)
        # combine spot coordinates with intensities
        spots_with_intensities = np.concatenate(
            (spots, spot_intensities, filtered_spot_intensities), axis=1
        )

        np.save(Path(self.savepath) / f'{ch}_spots.npy', spots_with_intensities)

        spots_df = pd.DataFrame(
            spots_with_intensities,
            columns=['z', 'y', 'x', 'intensity', 'filtered_intensity']
        )
        spots_df.to_csv(Path(self.savepath) / f'{ch}_spots.csv')

        self.mrna.get(ch)['spots'] = spots_with_intensities
        self.results[ch]['nr_spots'] = len(spots)

        logging.info(f'..channel {ch}: found {len(spots)} spots, threshold={threshold} ({self.results[ch]["detection_backend"]})')


    def decompose_spots(self):
        logging.info(f'decomposing spots')
        for ch in self.mrna.keys():
            if ch != 'DAPI':
                self.decompose_spots_channel(ch)


    def decompose_spots_channel(self, ch):
        mrna_data = self.mrna.get(ch)['data']
        #ifx_1, ifx_2 = self.results[ch]['ifx_1'], self.results[ch]['ifx_2']
        mrna_data_selected = mrna_data#[ifx_1:ifx_2, ...]

        # reset spot coordinates back to cropped coordinates 
        spots = self.mrna.get(ch)['spots'][:, 0:3].copy()
        #spots[:,0] -= ifx_1

        # we should probably only do spot decomposition on spots that are in cells
        scale, spot_radius = self.experiment.parameters['scale'], self.experiment.parameters['spot_radius']
        settings = dict(alpha=0.5,  # alpha impacts the number of spots per candidate region
                        beta=2,     # beta impacts the number of candidate regions to decompose
                        gamma=1)    # gamma the filtering step to denoise the image
        # the denoising filter (most of the time) runs on the GPU when available; the rest is Big-FISH
        decomp_spots, dense_regions, reference_spot = self._run_on_gpu_or_cpu(
            ch, 'decomposition',
            gpu=lambda device: gpu_filters.decompose_dense(mrna_data_selected, spots, scale, spot_radius,
                                                           device=device, **settings),
            cpu=lambda: decompose_dense(mrna_data_selected, spots, voxel_size=scale, spot_radius=spot_radius,
                                        **settings))

        self.mrna.get(ch)['decomposed_spots'] = decomp_spots
        self.mrna.get(ch)['dense_regions'] = dense_regions
        self.mrna.get(ch)['reference_spot'] = reference_spot

        np.save(Path(self.savepath) / f'{ch}_decomposed_spots.npy', decomp_spots)
        np.save(Path(self.savepath) / f'{ch}_dense_regions.npy', dense_regions)
        io.imsave(Path(self.savepath) / f'{ch}_reference_spot.tif', reference_spot, check_contrast=False)

        logging.info(f'..channel {ch}: found {len(dense_regions)} dense regions, {len(decomp_spots)} decomposed spots (from {len(spots)}')


    def assign_spots(self):
        logging.info(f'assign spots')
        for ch in self.mrna.keys():
            if ch != 'DAPI':
                self.assign_spots_channel(ch)


    def assign_spots_channel(self, ch):
        # reload these as changes could have been made to the cell and nuclear masks.
        self.cell_masks = io.imread(Path(self.savepath)/ f'corrected'/ f'DIC_corrected_masks.tif')
        self.dapi_masks = io.imread(Path(self.savepath)/ f'corrected'/ f'DAPI_corrected_masks.tif')

        # bud mask is optional: run without a bud compartment if it is absent
        bud_path = Path(self.savepath) / 'corrected' / 'bud_corrected_masks.tif'
        if bud_path.is_file():
            self.bud_masks = io.imread(bud_path)
            bud_labeled = np.where(self.bud_masks == 1, self.cell_masks, 0)
        else:
            self.bud_masks = None
            bud_labeled = None
            logging.info('..no bud_corrected_masks.tif; assigning without bud compartment')

        nuclear_labeled = np.where(self.dapi_masks > 0, self.cell_masks, 0)

        #load spot data 
        spot_data = self.mrna.get(ch)['spots'][:, 0:3]
        dense_data = self.mrna.get(ch)['dense_regions']

        # DAPI zproj
        zprojDAPI = np.max(self.mrna.get('DAPI')['data'],axis=0)

        df = spot_assignment(self.cell_masks, bud_labeled, nuclear_labeled, zprojDAPI,spot_data, dense_data)

        df.rename(columns={'label': 'image_cell_id'}, inplace=True)
        
        df.to_csv(Path(self.savepath) / f'{ch}.csv', index=False)
        logging.info(f'..channel {ch}: saving results to {self.savepath}/{ch}.csv')