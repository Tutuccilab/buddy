"""
Reading the microscope files (step 1) and writing buddy's own files of an image folder.

    from buddy.reading.readers import read_vsi, read_tiff_stack, save_channel_stacks, save_projection, read_json, write_json
"""
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
from skimage import io

from buddy.paths import ImageFiles

log = logging.getLogger("buddy")


def read_vsi(path, filter2mRNA):
    """
    The fluorescence channels of an Olympus .vsi file: {mrna: ZYX array} for every channel whose filter name
    (from the scene name, e.g. "CY5, CY3.5 NAR, CY3, DAPI") is a "filter" of the config.
    """
    import bioio_bioformats
    from bioio import BioImage  # heavy import (starts the Bio-Formats Java bridge)
    log.info(f"reading fluorescence image: {path}")
    image = BioImage(Path(path), reader=bioio_bioformats.Reader)
    log.info(f"..found scenes: {image.scenes}")
    channels = {}
    for scene in image.scenes:
        if scene == "macro image":
            log.info(f"....ignoring scene '{scene}'")
            continue
        image.set_scene(scene)
        log.info(f"....reading scene '{scene}' (shape {image.shape})")
        log.info(f"....found channels {image.channel_names}")
        tokens = [t.strip() for t in scene.replace("001 ", "").split(",")]
        log.info(f"....channel tokens from scene name: {tokens}")
        log.info(f"....configured filters: {list(filter2mRNA)}")
        for c, token in enumerate(tokens):
            if token in filter2mRNA:
                mrna = filter2mRNA[token]
                log.info(f"......reading in channel: C={c} filter={token} mrna={mrna}")
                channels[mrna] = image.get_image_data("ZYX", C=c)
            else:
                log.warning(f"......ignoring unknown channel token '{token}' (not a 'filter' in the config)")
    return channels


def read_tiff_stack(path, exp):
    """The channels of a multi-page TIFF: ({mrna: ZYX array}, DIC image or None). See buddy/tiff_input.py."""
    from buddy.reading.tiff_input import read_stack
    log.info(f"reading fluorescence image (TIFF stack): {path}")
    return read_stack(Path(path), exp)


def read_channels(exp, image_file):
    """Dispatch on the file type: ({mrna: ZYX array}, DIC from the stack or None)."""
    path = Path(exp.inputdir) / image_file
    if path.suffix.lower() in (".tif", ".tiff"):
        return read_tiff_stack(path, exp)
    return read_vsi(path, exp.filter2mRNA), None


def read_dic(path):
    log.info(f"reading cell image: {path}")
    return io.imread(path)


def save_projection(stack_file, stack):
    """
    Save the maximum projection of a z-stack as projections/<name>_max.tif next to it. napari-buddy shows
    these 2D projections when an image is opened (much faster than reading the full stacks).
    """
    stack_file = Path(stack_file)
    if np.ndim(stack) != 3:
        return
    folder = stack_file.parent / "projections"
    folder.mkdir(exist_ok=True)
    io.imsave(folder / f"{stack_file.stem}_max.tif", np.asarray(stack).max(axis=0), check_contrast=False)


def save_channel_stacks(folder, channels, dic):
    """{mrna: ZYX array} -> <folder>/<mrna>.tif (OME-TIFF) + projections, the DIC image -> <folder>/DIC.tif."""
    from bioio_ome_tiff.writers import OmeTiffWriter
    files = ImageFiles(folder)
    log.info(f"saving layers to: {folder}")
    for mrna, data in channels.items():
        path = files.channel_stack(mrna)
        OmeTiffWriter.save(data, path, dim_order="ZYX", channel_names=[mrna])
        save_projection(path, data)
        log.info(f"..saving {mrna} channel to {path}")
    io.imsave(files.dic, dic, check_contrast=False)
    log.info(f"..saving DIC channel to {files.dic}")


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, default=vars, indent=4)


def md5(path):
    """Checksum of a file (used to recognise whether a mask file was edited after buddy wrote it)."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()
