import logging
from buddy.utils import keep_logging
import json
from pathlib import Path
import re
from buddy.image import Image


# automatic mother/bud assignment: used when the config file has no "bud_classification" block
BUD_CLASSIFICATION_DEFAULTS = {
    "enabled": False,                       # true -> classify_buds() writes the bud masks automatically
    "model": "DIC_DAPI_only",               # or "with_ASH1CLB2" (needs the ASH1/CLB2 channel)
    "ash_channel": "ASH1CLB2",              # channel name (key in "channels") used by the with_ASH1CLB2 model
    "overwrite_manual_corrections": False,  # never replace bud masks you corrected by hand, unless true
    "weights_dir": None,                    # folder with the model weights; None = buddy/models (auto-download)
    "snap": True,                           # a claimed bud lying mostly inside one Omnipose object takes that whole object
}

# napari flags for unbudded cells with high ASH1/CLB2 (late M / early S, see buddy/cell_flags.py):
# used when the config file has no "cell_flags" block
CELL_FLAG_DEFAULTS = {
    "enabled": True,              # false -> no cell_flags.csv
    "ash_channel": None,          # None = the "ash_channel" of "bud_classification"
    "fold": 8,                    # flag cells with >= fold x the ASH1/CLB2 of a typical unbudded cell
    "nucleus_area": 0.5,          # "own full nucleus": area >= 0.5 x ...
    "nucleus_intensity": 0.7,     # ... and mean DAPI >= 0.7 x those of a typical unbudded cell
    "max_gap_px": 10,             # neighbours: at most this many pixels apart
}


def gpu_available(wanted=True):
    """True when a GPU is wanted and PyTorch can use one."""
    if not wanted:
        return False
    import torch
    if torch.cuda.is_available():
        return True
    logging.warning("use_gpu is true, but no CUDA GPU is available to PyTorch - running on CPU "
                    "(see INSTALL.md to install the GPU version of PyTorch)")
    return False


class Experiment:
    """
    An experiment is a list of images in the same directory,
    that share processing parameters
    """
    def __init__(self, cfg):
        for key, value in cfg.items():
            setattr(self, key, value)
        self.filter2mRNA = { v['filter']: v['mrna'] for v in cfg['channels'].values() }
        # deal with threshold 'None'
        for key in cfg['channels'].keys():
            if 'threshold' not in cfg['channels'][key]:
                self.channels[key]['threshold'] = None
        # GPU use (Omnipose + bud model): "use_gpu" in the parameters block, default true
        self.use_gpu = self.parameters.get('use_gpu', True)
        # automatic bud classification settings, with defaults for anything not in the config
        self.bud_classification = {**BUD_CLASSIFICATION_DEFAULTS, **cfg.get('bud_classification', {})}
        # napari flags for unbudded cells with high ASH1/CLB2
        self.cell_flags = {**CELL_FLAG_DEFAULTS, **cfg.get('cell_flags', {})}


    @classmethod
    def from_cfg_file(cls, cfg_file):
        try:
            with open(cfg_file, 'r') as file:
                logging.info(f'found cfg file {cfg_file}')
                cfg = json.load(file)
                cfg['cfg_file'] = cfg_file
                return cls(cfg)
        except FileNotFoundError:
            logging.warning(f'failed to find cfg file {cfg_file}')


    def init_omnipose(self, use_GPU=None):
        """
        Load the Omnipose/Cellpose segmentation models once, so they are not
        re-instantiated for every image in segment_cells() / segment_dapi().
        Stores them on the Experiment; Image.segment_* pick them up via
        self.experiment.cell_model / .nuclei_model.
        use_GPU: None = follow "use_gpu" in the config file (default true); falls back to CPU without a GPU.
        """
        with keep_logging():
            from cellpose_omni import models   # torch import deferred to here
            self.use_GPU = gpu_available(self.use_gpu if use_GPU is None else use_GPU)
            logging.info(f'initialising omnipose models (cell: cyto2_omni, nuclei) on {"GPU" if self.use_GPU else "CPU"}')
            self.cell_model = models.CellposeModel(gpu=self.use_GPU, model_type='cyto2_omni')
            self.nuclei_model = models.CellposeModel(gpu=self.use_GPU, model_type='nuclei')


    def init_bud_model(self):
        """
        Load the bud model chosen in the config ("bud_classification" -> "model") once, so
        Image.classify_buds() can reuse it for every image. Does nothing when bud classification is off.
        """
        settings = self.bud_classification
        if not settings['enabled']:
            logging.info('automatic bud classification is off ("bud_classification" -> "enabled": false)')
            self.bud_model = None
            return
        from buddy.bud_model import BudModel     # torch import deferred to here
        self.bud_model = BudModel.load(settings['model'], use_gpu=gpu_available(self.use_gpu),
                                       weights_dir=settings['weights_dir'], snap=settings['snap'])


    def create_image_list(self):
        """The images in inputdir: .vsi files (+ their _DIC.tif), and multi-page TIFF stacks (see buddy/tiff_input.py)."""
        self.images = {}
        self._add_tiff_stacks()

        # get the vsi files:
        vsi_files = list(Path(self.inputdir).glob('*.vsi'))
        logging.info(f'getting .vsi files from {self.inputdir}')

        for f in vsi_files:
            logging.info(f'examining image {f}')
            stem = re.sub(r'_CY[A-Z0-9\s,._]+DAPI', '', f.stem)
            params = { 'vsi_file': f.parts[-1], 'cell_file': None, 'valid': True }

            # we need a VSI directory
            if (Path(self.inputdir) / f'_{f.stem}_').is_dir():
                logging.info(f'..found VSI directory _{f.stem}_')
            else:
                logging.warning(f'..failed to find VSI directory _{f.stem}_')
                params['valid'] = False

            # we need a DIC file
            if (Path(self.inputdir) / f'{stem}_DIC.tif').is_file():
                # sequence number precedes "DIC"
                params['cell_file'] = (Path(self.inputdir) / f'{stem}_DIC.tif').parts[-1]
                logging.info(f'..found DIC file {params["cell_file"]}')
            else:
                # sequence number follows "DIC":
                dicfilename = re.sub(r'(_\d\d)$', r'_DIC\1', stem)
                if (Path(self.inputdir) / f'{dicfilename}.tif').is_file():
                    params['cell_file'] = (Path(self.inputdir) / f'{dicfilename}.tif').parts[-1]
                    logging.info(f'..found DIC file {params["cell_file"]}')
                else:
                    logging.warning(f'..failed to find a DIC file for {Path(self.inputdir) / stem}')
                    params['valid'] = False

            if params['valid'] == True:
                logging.info(f'..all good, adding image to image list')
                self.images[stem] = params
                # self.images[stem] = Image.from_dict(self, params)


    def _add_tiff_stacks(self):
        from buddy.tiff_input import find_dic_file, is_stack
        for f in sorted(Path(self.inputdir).glob('*.tif*')):
            stem = re.sub(r'\.ome$', '', f.stem)
            if stem.endswith('_DIC') or re.search(r'_DIC_\d\d$', stem):
                continue                                     # a DIC image, not a stack
            info = is_stack(f)
            if info is None:
                continue
            logging.info(f'examining TIFF stack {f.name}: axes {info["axes"]}, shape {info["shape"]}, channels {info["channel_names"] or "unnamed"}')
            dic_file = find_dic_file(self.inputdir, stem)
            dic_name = str(self.parameters.get('dic_channel', 'DIC'))
            dic_in_stack = any(n.strip().upper() == dic_name.upper() for n in info['channel_names']) or dic_name.isdigit()
            if dic_file is None and not dic_in_stack:
                logging.warning(f'..no DIC image for {f.name} (expected {stem}_DIC.tif, or a channel named {dic_name!r}) - skipped')
                continue
            logging.info(f'..DIC: {dic_file or "channel of the stack"}; adding image to image list')
            self.images[stem] = {'vsi_file': f.name, 'cell_file': dic_file, 'valid': True}


    def read_image_list_from_jsons(self):
        # get the img.json files:
        self.json_files = sorted(Path(self.outputdir).glob('*/img.json'))

        for f in self.json_files:
            logging.info(f'reading json file {f}')