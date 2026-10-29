"""
Step 3a per channel: the in-focus slices, the background filter, spot detection and the decomposition of dense
regions (Big-FISH). Every function takes arrays and settings and returns its result; pipeline.py writes the files.
spots/gpu_detection.py has the same three Big-FISH operations (background filter, detection, decomposition) in
PyTorch for the GPU; run_on_gpu_or_cpu() picks one per call and falls back to Big-FISH when the GPU cannot.

    from buddy.spots.detection import find_focus, filter_background, detect_spots, decompose_spots
"""
import logging
from typing import Tuple

import numpy as np
from bigfish.detection import decompose_dense as _bf_decompose_dense
from bigfish.detection import detect_spots as _bf_detect_spots
from bigfish.stack import compute_focus, remove_background_gaussian
from scipy.signal import savgol_filter

from buddy.config import DECOMPOSITION_DEFAULTS
from buddy.spots import gpu_detection as gpu_versions

log = logging.getLogger("buddy")

DECOMPOSITION = DECOMPOSITION_DEFAULTS   # alpha, beta, gamma of Big-FISH's decompose_dense; the "spots" block of the config can change them
DECOMPOSITION_SEED = 0            # Big-FISH builds the reference spot from up to 2000 randomly chosen spots; a fixed seed makes reruns identical


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
                        "(lower buddy.spots.gpu_detection.BLOCK_ROWS if the GPU runs out of memory)")
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
def filter_background(stack, sigma, device=None, what=""):
    """
    Big-FISH remove_background_gaussian on the whole stack. Returns (filtered stack, backend).
    The whole stack, not only the in-focus slices: the LoG filter of the detection reaches 4 sigma = 16 slices
    in z, so a stack cut at the focus range would make the cut planes local maxima (a ridge of false spots on
    the lowest in-focus plane). The in-focus range is applied to the spots instead (detect_spots).
    """
    return run_on_gpu_or_cpu(lambda d: gpu_versions.remove_background(stack, sigma, device=d),
                             lambda: remove_background_gaussian(stack, sigma=sigma), device, what)


# -------------------------------------------------------------------------------------------------
# spot detection
# -------------------------------------------------------------------------------------------------
def detect_spots(stack, filtered, ifx_1, ifx_2, threshold, scale, spot_radius, device=None, what=""):
    """
    Spots of the filtered stack that lie in the in-focus slices [ifx_1, ifx_2) (threshold None = Big-FISH's
    automatic threshold). The detection runs on the whole stack, so that no plane is an edge; only the spots
    in the range are kept. Returns (spots (z, y, x, intensity, filtered_intensity), threshold used, backend).
    """
    (spots, threshold), backend = run_on_gpu_or_cpu(
        lambda d: gpu_versions.detect_spots(filtered, threshold, scale, spot_radius, device=d),
        lambda: _bf_detect_spots(filtered, threshold=threshold, voxel_size=scale, spot_radius=spot_radius, return_threshold=True),
        device, what)
    spots = spots[(spots[:, 0] >= ifx_1) & (spots[:, 0] < ifx_2)]
    intensity = stack[spots[:, 0], spots[:, 1], spots[:, 2]].reshape(-1, 1)
    filtered_intensity = filtered[spots[:, 0], spots[:, 1], spots[:, 2]].reshape(-1, 1)
    return np.concatenate((spots, intensity, filtered_intensity), axis=1), threshold, backend


def decompose_spots(stack, spots, scale, spot_radius, device=None, what="", decomposition=None):
    """
    Dense regions (several molecules) decomposed with Big-FISH. Returns (decomposed spots, dense regions, reference spot, backend).
    decomposition: {"alpha", "beta", "gamma"} of Big-FISH's decompose_dense ("spots" block of the config), None = the defaults.
    Big-FISH samples up to 2000 spots at random to build the reference spot; NumPy's random generator is seeded with
    DECOMPOSITION_SEED for this step (and restored afterwards), so reruns on the CPU or the GPU give the same result.
    """
    spots = np.asarray(spots)[:, :3].copy()
    settings = {**DECOMPOSITION, **(decomposition or {})}
    state = np.random.get_state()
    np.random.seed(DECOMPOSITION_SEED)
    try:
        (decomposed, dense, reference), backend = run_on_gpu_or_cpu(
            lambda d: gpu_versions.decompose_dense(stack, spots, scale, spot_radius, device=d, **settings),
            lambda: _bf_decompose_dense(stack, spots, voxel_size=scale, spot_radius=spot_radius, **settings), device, what)
    finally:
        np.random.set_state(state)                 # leave the rest of buddy's random state as it was
    return decomposed, dense, reference, backend
