"""
Step 3a per channel: the in-focus slices, the background filter, spot detection and the decomposition of dense
regions (Big-FISH, with GPU versions in buddy.spots.gpu). Every function takes arrays and settings and returns
its result; Image writes the files.

    from buddy.spots.detection import find_focus, filter_background, detect_spots, decompose_spots
"""
import logging
from typing import Tuple

import numpy as np
from bigfish.detection import decompose_dense as _bf_decompose_dense
from bigfish.detection import detect_spots as _bf_detect_spots
from bigfish.stack import compute_focus, remove_background_gaussian
from scipy.signal import savgol_filter

from buddy.spots import gpu as gpu_versions

log = logging.getLogger("buddy")

DECOMPOSITION = dict(alpha=0.5,   # alpha impacts the number of spots per candidate region
                     beta=2,      # beta impacts the number of candidate regions to decompose
                     gamma=1)     # gamma the filtering step to denoise the image
DROP_BOTTOM_SLICES = 2            # spots in the two lowest in-focus slices look like noise and are dropped


# -------------------------------------------------------------------------------------------------
# GPU / CPU
# -------------------------------------------------------------------------------------------------
def run_on_gpu_or_cpu(gpu, cpu, device, what=""):
    """
    gpu(device) when a device is given (falls back to the CPU version when the GPU runs out of memory), else cpu().
    Returns (result, backend name) - the backend is recorded in img.json.
    """
    if device is not None:
        try:
            return gpu(device), "gpu"
        except RuntimeError as e:                       # includes torch.cuda.OutOfMemoryError
            import torch
            torch.cuda.empty_cache()
            log.warning(f"..{what}: GPU failed ({e.__class__.__name__}), using Big-FISH on the CPU "
                        "(lower buddy.spots.gpu.BLOCK_ROWS if the GPU runs out of memory)")
    return cpu(), "cpu (big-fish)"


# -------------------------------------------------------------------------------------------------
# focus
# -------------------------------------------------------------------------------------------------
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



def find_focus(stack, patch, patch_size):
    """
    The in-focus z range of a channel: the focus score of the densest patch (patch = its top-left corner) per
    slice, smoothed, and its steepest rise and fall. Returns dict(z_max_focus, ifx_1, ifx_2).
    """
    img_patch = stack[:, patch[0]:patch[0] + patch_size[0], patch[1]:patch[1] + patch_size[1]]
    focus = compute_focus(img_patch)
    projected = np.max(focus, axis=(1, 2))
    smoothed = savgol_filter(projected, 16, 2, 0)
    ifx_1, ifx_2 = find_in_focus_indices(smoothed)
    if ifx_1 < 0 or ifx_2 > stack.shape[0]:
        log.warning("....focus detection: max focus is too close to highest or lowest slice")
    return {"z_max_focus": int(np.argmax(smoothed)), "ifx_1": int(max(ifx_1, 0)), "ifx_2": int(min(ifx_2, stack.shape[0]))}


# -------------------------------------------------------------------------------------------------
# background filter
# -------------------------------------------------------------------------------------------------
def filter_background(stack, sigma, ifx_1, ifx_2, device=None, what=""):
    """
    Big-FISH remove_background_gaussian, only on the in-focus slices [ifx_1, ifx_2) plus a margin as wide as
    the Gaussian's reach in z, so the result inside the range is identical to filtering the whole stack.
    Slices outside the range stay 0. Returns (filtered stack, backend).
    """
    n_z = stack.shape[0]
    margin = int(4 * sigma[0] + 0.5) + 1          # Gaussian kernel reach in z (truncate = 4 sigma) + 1
    lo, hi = max(ifx_1 - margin, 0), min(ifx_2 + margin, n_z)
    part, backend = run_on_gpu_or_cpu(lambda d: gpu_versions.remove_background(stack[lo:hi], sigma, device=d),
                                      lambda: remove_background_gaussian(stack[lo:hi], sigma=sigma), device, what)
    filtered = np.zeros_like(stack)
    filtered[ifx_1:ifx_2] = part[ifx_1 - lo:ifx_2 - lo]
    return filtered, backend


# -------------------------------------------------------------------------------------------------
# spot detection
# -------------------------------------------------------------------------------------------------
def detect_spots(stack, filtered, ifx_1, ifx_2, threshold, scale, spot_radius, device=None, what=""):
    """
    Spots in the in-focus slices of the filtered stack (threshold None = Big-FISH's automatic threshold).
    Returns (spots (z, y, x, intensity, filtered_intensity) in full-stack coordinates, threshold used, backend).
    """
    selected = filtered[ifx_1:ifx_2]
    (spots, threshold), backend = run_on_gpu_or_cpu(
        lambda d: gpu_versions.detect_spots(selected, threshold, scale, spot_radius, device=d),
        lambda: _bf_detect_spots(selected, threshold=threshold, voxel_size=scale, spot_radius=spot_radius, return_threshold=True),
        device, what)
    spots[:, 0] += ifx_1                                                    # back to full-stack z
    spots = spots[spots[:, 0] > ifx_1 + DROP_BOTTOM_SLICES]
    intensity = stack[spots[:, 0], spots[:, 1], spots[:, 2]].reshape(-1, 1)
    filtered_intensity = filtered[spots[:, 0], spots[:, 1], spots[:, 2]].reshape(-1, 1)
    return np.concatenate((spots, intensity, filtered_intensity), axis=1), threshold, backend


def decompose_spots(stack, spots, scale, spot_radius, device=None, what=""):
    """Dense regions (several molecules) decomposed with Big-FISH. Returns (decomposed spots, dense regions, reference spot, backend)."""
    spots = np.asarray(spots)[:, :3].copy()
    (decomposed, dense, reference), backend = run_on_gpu_or_cpu(
        lambda d: gpu_versions.decompose_dense(stack, spots, scale, spot_radius, device=d, **DECOMPOSITION),
        lambda: _bf_decompose_dense(stack, spots, voxel_size=scale, spot_radius=spot_radius, **DECOMPOSITION), device, what)
    return decomposed, dense, reference, backend
