"""
GPU versions of the two slow Big-FISH steps, giving the same result as Big-FISH.

    remove_background(image, sigma)                         = bigfish.stack.remove_background_gaussian
    detect_spots(image, threshold, voxel_size, spot_radius) = bigfish.detection.detect_spots (fixed or automatic threshold)
    decompose_dense(image, spots, voxel_size, spot_radius)  = bigfish.detection.decompose_dense

Why Big-FISH is slow here: its spot detector smooths with a Laplacian-of-Gaussian whose kernel reaches
+-18 z-slices (spot radius 880 nm / 200 nm slices), computed on the CPU in 64-bit precision, for every
pixel of every channel. These versions do the same separable filtering with PyTorch (on the GPU when
available; also works on the CPU) in 32-bit precision, in blocks of rows so a full image fits in GPU memory.
The final steps (threshold, removing duplicate detections) are Big-FISH's own functions.

Tested against Big-FISH on two images (YET915 BR5 _02, three channels; 220601 yET563 BR1 _02 SRL1 at three
fixed thresholds and the automatic threshold): identical spots; filtered values differ by at most 1 grey level in ~0.01% of pixels (rounding).
On a CPU, Big-FISH itself is faster, so buddy only uses these functions on a GPU (see Image._run_on_gpu_or_cpu).
Switch off with "gpu_spot_detection": false in the parameters block of the config file.
"""
import numpy as np
from bigfish.detection import automated_threshold_setting, spots_thresholding
from bigfish.detection.utils import get_object_radius_pixel

try:                                                       # the exact kernels SciPy (and so Big-FISH) uses
    from scipy.ndimage._filters import _gaussian_kernel1d
except ImportError:                                        # older SciPy
    from scipy.ndimage.filters import _gaussian_kernel1d

TRUNCATE = 4.0          # SciPy/scikit-image default: the kernel reaches 4 sigma
BLOCK_ROWS = 512        # rows (y) processed at once; lower this if the GPU runs out of memory


# ---------------------------------------------------------------------------------------------
# separable filtering with SciPy's border rules
# ---------------------------------------------------------------------------------------------
def _pad(x, r, dim, mode):
    """Pad r pixels on both sides of one dimension. 'reflect' = SciPy reflect (d c b a | a b c d), 'nearest' = edge value."""
    import torch
    while r > 0:                      # SciPy keeps reflecting when the kernel is longer than the image
        n = x.shape[dim]
        k = min(r, n)
        if mode == "nearest":
            before = x.narrow(dim, 0, 1).expand(*[k if d == dim else -1 for d in range(x.dim())])
            after = x.narrow(dim, n - 1, 1).expand(*[k if d == dim else -1 for d in range(x.dim())])
        else:
            before = x.narrow(dim, 0, k).flip(dim)
            after = x.narrow(dim, n - k, k).flip(dim)
        x = torch.cat([before, x, after], dim)
        r -= k
    return x


def _filter_1d(x, sigma, order, dim, mode):
    """1D Gaussian (order 0) or second-derivative-of-Gaussian (order 2) filter along one dimension."""
    import torch
    import torch.nn.functional as F
    radius = int(TRUNCATE * float(sigma) + 0.5)
    weights = _gaussian_kernel1d(float(sigma), order, radius)[::-1].copy()     # SciPy correlates with the reversed kernel
    w = torch.as_tensor(weights, dtype=x.dtype, device=x.device).view(1, 1, -1)
    x = _pad(x, radius, dim, mode).movedim(dim, -1)
    shape = x.shape
    y = F.conv1d(x.reshape(-1, 1, shape[-1]), w)
    return y.reshape(*shape[:-1], -1).movedim(-1, dim)


def _gaussian(x, sigma, mode):
    for dim, s in enumerate(sigma):
        x = _filter_1d(x, s, 0, dim, mode)
    return x


def _laplace_of_gaussian(x, sigma, mode="reflect"):
    total = None
    for axis in range(x.dim()):
        y = x
        for dim, s in enumerate(sigma):
            y = _filter_1d(y, s, 2 if dim == axis else 0, dim, mode)
        total = y if total is None else total + y
    return total


def _in_row_blocks(image, halo, function, device, out_dtype=None):
    """
    Apply function to blocks of rows (+ halo rows on each side, discarded afterwards); returns the stitched result(s).
    out_dtype: store the result(s) in this dtype right away (block by block), instead of a full-size copy afterwards.
    """
    import torch
    n_rows = image.shape[1]
    outputs = None
    for y0 in range(0, n_rows, BLOCK_ROWS):
        y1 = min(y0 + BLOCK_ROWS, n_rows)
        a, b = max(y0 - halo, 0), min(y1 + halo, n_rows)
        block = torch.as_tensor(np.ascontiguousarray(image[:, a:b]), device=device).float()
        results = function(block)
        results = results if isinstance(results, tuple) else (results,)
        if outputs is None:
            outputs = [np.zeros(image.shape, out_dtype or r.cpu().numpy().dtype) for r in results]
        for out, r in zip(outputs, results):
            out[:, y0:y1] = r[:, y0 - a:y0 - a + (y1 - y0)].cpu().numpy()      # numpy casts to out's dtype
        del block, results
    return outputs if len(outputs) > 1 else outputs[0]


# ---------------------------------------------------------------------------------------------
# the two Big-FISH steps
# ---------------------------------------------------------------------------------------------
def remove_background(image, sigma, device="cuda"):
    """Same as bigfish.stack.remove_background_gaussian for a uint16 (z, y, x) stack."""
    import torch
    halo = int(TRUNCATE * float(sigma[1]) + 0.5)

    def one_block(x):
        smooth = torch.clamp(torch.round(_gaussian(x, sigma, mode="nearest")), 0, 65535)   # Big-FISH: blurred image as uint16
        return torch.where(x > smooth, x - smooth, torch.zeros_like(x)).to(torch.int32)

    with torch.no_grad():
        return _in_row_blocks(image, halo, one_block, device, out_dtype=np.uint16)   # 0..65535: fits, no int32 copy


def log_and_local_maxima(image, voxel_size, spot_radius, device="cuda"):
    """
    The two filtering steps of Big-FISH spot detection, on the GPU: the Laplacian-of-Gaussian filtered image
    (as uint16, like bigfish.stack.log_filter) and the local-maximum mask (like
    bigfish.detection.local_maximum_detection). Thresholding these gives the spots.
    """
    import torch
    import torch.nn.functional as F
    sigma = get_object_radius_pixel(voxel_size_nm=voxel_size, object_radius_nm=spot_radius, ndim=3)
    min_distance = np.ceil(sigma).astype(int)                 # local-maximum neighbourhood, as in Big-FISH
    halo = int(TRUNCATE * float(sigma[1]) + 0.5) + int(min_distance[1])

    def one_block(x):
        log = torch.clamp(-_laplace_of_gaussian(x, sigma), min=0)
        log = torch.clamp(torch.round(log), 0, 65535)         # Big-FISH keeps the filtered image as uint16
        local_max = F.max_pool3d(log[None, None], kernel_size=tuple(2 * min_distance + 1), stride=1,
                                 padding=tuple(min_distance))[0, 0]
        return log.to(torch.int32), (log == local_max)

    with torch.no_grad():
        log, mask = _in_row_blocks(image, halo, one_block, device)
    return log.astype(np.uint16), mask


def detect_spots(image, threshold, voxel_size, spot_radius, device="cuda"):
    """
    Same as bigfish.detection.detect_spots(image, threshold, voxel_size=..., spot_radius=..., return_threshold=True)
    for a uint16 (z, y, x) stack. threshold=None: Big-FISH's automatic threshold (the kink in the spot-count
    curve), computed by Big-FISH from the filtered image. Returns (spots (n, 3: z, y, x), threshold).
    """
    log, mask = log_and_local_maxima(image, voxel_size, spot_radius, device=device)
    if threshold is None:
        threshold = automated_threshold_setting(log, mask)
    spots, _ = spots_thresholding(log, mask, threshold, remove_duplicate=True)
    return spots, threshold


def gpu_device():
    """'cuda' when PyTorch can use a GPU, otherwise None (callers then use Big-FISH on the CPU)."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else None
    except ImportError:
        return None


def decompose_dense(image, spots, voxel_size, spot_radius, alpha=0.5, beta=2, gamma=1, device="cuda"):
    """
    Same as bigfish.detection.decompose_dense(image, spots, voxel_size, spot_radius, alpha, beta, gamma).
    About 80% of Big-FISH's time is its first step, a Gaussian background filter over the whole stack
    (sigma = spot radius in pixels x gamma). That filter runs here on the GPU; the rest (reference spot,
    dense regions, Gaussian mixture) is Big-FISH's own code on the filtered image (gamma=0 skips its filter).
    Note: Big-FISH places the extra spots in a dense region partly at random, so two runs (also two Big-FISH
    runs) give the same number of spots and the same dense regions, but some positions inside regions differ.
    Returns (spots, dense_regions, reference_spot) as Big-FISH.
    """
    from bigfish.detection import decompose_dense as bigfish_decompose_dense
    if gamma > 0 and len(spots):
        kernel = tuple(get_object_radius_pixel(voxel_size_nm=voxel_size, object_radius_nm=spot_radius, ndim=3) * gamma)
        image = remove_background(image, kernel, device=device)
    return bigfish_decompose_dense(image, spots, voxel_size=voxel_size, spot_radius=spot_radius,
                                   alpha=alpha, beta=beta, gamma=0)
