"""
Multi-page TIFF as input for step 1 (instead of an Olympus .vsi): an OME-TIFF or an ImageJ/Fiji hyperstack with the
fluorescence channels (z-stacks), e.g. exported from Fiji or written by another microscope's software.

File naming, as for .vsi: <stem>.tif (or .ome.tif) with the channels, and <stem>_DIC.tif (or <stem with _DIC before
the sequence number>.tif) with the 2D DIC image next to it. The DIC image may also be a channel of the stack
(name "DIC", or "dic_channel" in the config's "tiff_input" block); then the middle z plane is used.

Which channel is which mRNA: the channel names in the file (OME: Channel Name; ImageJ: the "Labels" or "channel
names" the writer stored) are matched against the config's "channels": the "filter" name, the channel key or the
"mrna" name. Files without channel names need "tiff_channel": <index> in every "channels" entry of the config.
Axis order comes from the file's metadata; files without it need "tiff_axes" (e.g. "ZCYX") in "tiff_input".
Pixel size and z step are taken from the config ("scale"), as for .vsi files.

    from buddy.reading.tiff_input import is_stack, read_stack
    info = is_stack(path)                                   # None, or dict(axes, shape, channel_names)
    channels, dic = read_stack(path, exp)                   # {mrna: ZYX array}, 2D DIC or None
"""
import logging
import re
from pathlib import Path

import numpy as np

log = logging.getLogger("buddy")

STACK_SUFFIXES = (".tif", ".tiff", ".ome.tif", ".ome.tiff")


def _channel_names(tf, n_channels):
    """channel names stored in the file, or [] when there are none"""
    names = []
    if tf.ome_metadata:
        names = re.findall(r'<Channel\b[^>]*\bName="([^"]*)"', tf.ome_metadata)
    elif tf.imagej_metadata:
        meta = tf.imagej_metadata
        for key in ("channel_names", "Labels"):
            v = meta.get(key)
            if isinstance(v, (list, tuple)) and len(v) >= n_channels:
                names = [str(x) for x in v[:n_channels]]
                break
    return names if len(names) == n_channels else []


def is_stack(path):
    """
    dict(axes, shape, channel_names) when `path` is a multi-channel or multi-plane TIFF, else None.
    A plain 2D image (the DIC file) is not a stack.
    """
    import tifffile
    path = Path(path)
    if path.suffix.lower() not in (".tif", ".tiff"):
        return None
    try:
        with tifffile.TiffFile(path) as tf:
            s = tf.series[0]
            axes, shape = s.axes, s.shape
            if len(shape) < 3:
                return None
            n_channels = shape[axes.index("C")] if "C" in axes else 1
            return dict(axes=axes, shape=tuple(int(x) for x in shape), channel_names=_channel_names(tf, n_channels),
                        ome=bool(tf.ome_metadata), imagej=bool(tf.imagej_metadata))
    except Exception as e:
        log.warning(f"{path.name}: not readable as a TIFF stack ({e})")
        return None


def _to_czyx(data, axes):
    """array with the file's axes -> (C, Z, Y, X); T / S / Q axes of size 1 are dropped, others are an error"""
    axes = axes.upper()
    for ax in [a for a in axes if a not in "CZYX"]:
        i = axes.index(ax)
        if data.shape[i] != 1:
            raise ValueError(f"cannot handle axis {ax!r} of size {data.shape[i]} (axes {axes})")
        data = np.take(data, 0, axis=i); axes = axes[:i] + axes[i + 1:]
    for ax in "CZ":
        if ax not in axes:
            data = data[None]; axes = ax + axes
    order = [axes.index(a) for a in "CZYX"]
    return np.transpose(data, order)


def read_stack(path, exp, info=None):
    """
    Read the channels of a TIFF stack for the Experiment's config.
    Returns ({mrna: ZYX array}, dic) where dic is the 2D DIC image when it is a channel of the stack, else None.
    """
    import tifffile
    path = Path(path)
    info = info or is_stack(path)
    if info is None:
        raise ValueError(f"{path} is not a TIFF stack")
    params = getattr(exp, "parameters", {}) or {}
    axes = info["axes"]
    if params.get("tiff_axes"):
        axes = params["tiff_axes"]
    elif not any(a in axes for a in "CZ"):
        raise ValueError(f"{path.name}: no axis information in the file - set \"tiff_axes\" (e.g. \"ZCYX\") in the config's parameters")
    data = _to_czyx(tifffile.imread(path), axes)
    n_channels = data.shape[0]
    names = info["channel_names"]
    dic_name = str(params.get("dic_channel", "DIC"))

    # which channel index belongs to which config entry
    index_of = {}
    for key, cfg in exp.channels.items():
        idx = cfg.get("tiff_channel")
        if idx is None and names:
            wanted = {str(cfg.get("filter", "")).strip(), key, str(cfg.get("mrna", ""))} - {""}
            matches = [i for i, n in enumerate(names) if n.strip() in wanted]
            idx = matches[0] if matches else None
        if idx is None:
            raise ValueError(f"{path.name}: cannot tell which channel is {key!r}: the file's channel names are {names or 'missing'}; "
                             f"add \"tiff_channel\": <index> to that entry of \"channels\" in the config")
        if not 0 <= int(idx) < n_channels:
            raise ValueError(f"{path.name}: \"tiff_channel\" {idx} for {key!r} but the file has {n_channels} channels")
        index_of[key] = int(idx)
    channels = {}
    for key, cfg in exp.channels.items():
        mrna = cfg.get("mrna", key)
        channels[mrna] = data[index_of[key]]
        log.info(f"......channel {index_of[key]} ({names[index_of[key]] if names else 'no name'}) -> {mrna}, shape {channels[mrna].shape}")
    dic = None
    dic_idx = next((i for i, n in enumerate(names) if n.strip().upper() == dic_name.upper()), None)
    if dic_idx is None and str(params.get("dic_channel", "")).isdigit():
        dic_idx = int(params["dic_channel"])
    if dic_idx is not None and dic_idx not in index_of.values():
        z = data[dic_idx]
        dic = z[z.shape[0] // 2] if z.shape[0] > 1 else z[0]
        log.info(f"......channel {dic_idx} -> DIC (middle z plane)")
    unused = [names[i] if names else i for i in range(n_channels) if i not in index_of.values() and i != dic_idx]
    if unused:
        log.warning(f"......ignoring channel(s) {unused} (not a 'filter' in the config)")
    return channels, dic


def find_dic_file(inputdir, stem):
    """<stem>_DIC.tif, or <stem with _DIC before the sequence number>.tif, next to the stack; None when absent"""
    inputdir = Path(inputdir)
    for name in (f"{stem}_DIC.tif", f"{stem}_DIC.tiff", re.sub(r"(_\d\d)$", r"_DIC\1", stem) + ".tif"):
        if (inputdir / name).is_file():
            return name
    return None
