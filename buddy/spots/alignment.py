"""
Alignment of the DIC frame (DIC image + Omnipose cell masks) to the fluorescence frame (where the spots were
detected). The two are offset by a small translation that differs per image. The shift is found by moving the
spots onto the cell masks: a Monte-Carlo search over shifts scored on a capped distance map, then a local
refinement (Sander van Otterdijk, 2026). `align_image` applies it to an image folder (step 3).

    from buddy.spots.alignment import estimate_alignment_shift, align_image
    mask_shift, spot_shift = estimate_alignment_shift(spots, cell_masks)   # (dy, dx) to move the masks by
"""
import logging

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import shift as ndi_shift

log = logging.getLogger("buddy")


# ----------------------------
# Function: score_shift
# ----------------------------
def score_shift(spots, dist, dy, dx):
    """
    Compute a score for a given shift of spot coordinates relative to a 2D cost field.

    Parameters
    ----------
    spots : np.ndarray
        Array of shape (N, >=3) representing detected spots. Columns:
        [z, y-coordinate, x-coordinate, ...]. Only columns 1 and 2 are used.
    dist : np.ndarray
        2D cost field to align to (lower = better; e.g. a capped distance map).
    dy, dx : int
        Shift in the y- and x-direction.

    Returns
    -------
    float
        Negative mean of the values in `dist` at the shifted spot coordinates
        (so higher is better). Returns -inf if no spot lands inside the field.
    """
    ys = spots[:, 1] + dy
    xs = spots[:, 2] + dx
    valid = (ys >= 0) & (ys < dist.shape[0]) & (xs >= 0) & (xs < dist.shape[1])

    # If the shift pushes every spot out of bounds, this shift is worthless.
    if not valid.any():
        return -np.inf

    ys = ys[valid].astype(int)
    xs = xs[valid].astype(int)
    return -np.mean(dist[ys, xs])


# ----------------------------
# Function: build_distance_map
# ----------------------------
def build_distance_map(masks, cutoff=50):
    """
    Build the 2D cost field used for alignment from a label/binary mask.

    For every pixel this is the Euclidean distance to the nearest masked pixel,
    clipped at `cutoff`. Pixels inside a mask score 0 (best); pixels far from
    any mask saturate at `cutoff` (worst, but bounded so a handful of stray
    spots can't dominate the mean).

    IMPORTANT
    ---------
    Do NOT set far-away pixels to 0 (e.g. ``np.where(dist > cutoff, 0, dist)``).
    That makes empty space score identically to the interior of a cell, which
    creates a degenerate optimum: the search can maximise the score by pushing
    every spot off into empty space instead of onto the cells. Clipping with
    ``np.minimum`` keeps empty space penalised.

    Parameters
    ----------
    masks : np.ndarray
        2D label or binary mask (anything > 0 is treated as foreground).
    cutoff : float
        Distance (in pixels) at which the cost saturates.

    Returns
    -------
    np.ndarray
        2D float cost field, same shape as `masks`.
    """
    binary = masks > 0
    dist = distance_transform_edt(~binary)
    return np.minimum(dist, cutoff)


# ----------------------------
# Function: monte_carlo_align
# ----------------------------
def monte_carlo_align(spots, dist, n_iter=3000, max_shift=200, rng=None):
    """
    Coarse global alignment via random search over integer shifts.

    Parameters
    ----------
    spots : np.ndarray
        Array of detected spot coordinates ([z, y, x, ...]).
    dist : np.ndarray
        2D cost field to align to (e.g. from `build_distance_map`).
    n_iter : int, optional
        Number of random shifts to try (default 3000).
    max_shift : int, optional
        Maximum absolute shift to consider in both directions (default 200).
    rng : np.random.Generator or None, optional
        Optional random generator for reproducibility.

    Returns
    -------
    tuple
        Best shift (dy, dx) found during the search.
    """
    if rng is None:
        rng = np.random.default_rng()

    best_shift = (0, 0)
    best_score = -np.inf
    for _ in range(n_iter):
        dy = int(rng.integers(-max_shift, max_shift + 1))
        dx = int(rng.integers(-max_shift, max_shift + 1))
        s = score_shift(spots, dist, dy, dx)
        if s > best_score:
            best_score = s
            best_shift = (dy, dx)

    return best_shift


# ----------------------------
# Function: refine_shift
# ----------------------------
def refine_shift(spots, dist, shift, radius=10):
    """
    Local exhaustive refinement of an initial shift.

    Parameters
    ----------
    spots : np.ndarray
        Array of detected spot coordinates.
    dist : np.ndarray
        2D cost field to align to.
    shift : tuple
        Initial guess for the shift (dy, dx).
    radius : int, optional
        Half-width of the local neighbourhood to explore (default 10).

    Returns
    -------
    tuple
        Refined shift (dy, dx) with the highest score in the local neighbourhood.
    """
    best_shift = shift
    best_score = score_shift(spots, dist, *shift)
    for dy in range(shift[0] - radius, shift[0] + radius + 1):
        for dx in range(shift[1] - radius, shift[1] + radius + 1):
            s = score_shift(spots, dist, dy, dx)
            if s > best_score:
                best_score = s
                best_shift = (dy, dx)

    return best_shift


# ----------------------------
# Function: estimate_alignment_shift
# ----------------------------
def estimate_alignment_shift(spots, masks, n_iter=4000, max_shift=200,
                             radius=20, cutoff=50, rng=None):
    """
    Estimate the translation that lines up DIC-frame `masks` with the
    fluorescence-frame `spots`.

    The Monte-Carlo + refine search finds the shift that maps the *spots*
    onto the *masks*. Downstream we want to move the masks (and the DIC
    image) instead, so the returned `mask_shift` is the sign-inverted
    version, ready to hand to ``scipy.ndimage.shift(mask, mask_shift)``.

    Parameters
    ----------
    spots : np.ndarray
        Detected spots, shape (N, >=3), columns [z, y, x, ...].
    masks : np.ndarray
        2D cell mask in the DIC frame. Must be at least as large as the
        spot coordinates index into (pad it first if not).
    n_iter, max_shift, radius, cutoff : see the functions above.
    rng : np.random.Generator or None, optional.

    Returns
    -------
    mask_shift : tuple(int, int)
        (dy, dx) to apply to masks / DIC so they line up with the spots.
    spot_shift : tuple(int, int)
        (dy, dx) that maps spots onto masks (the raw search result), kept
        for logging / debugging.
    """
    spots = np.asarray(spots)
    dist = build_distance_map(masks, cutoff=cutoff)

    coarse = monte_carlo_align(spots, dist, n_iter=n_iter, max_shift=max_shift, rng=rng)
    spot_shift = refine_shift(spots, dist, coarse, radius=radius)

    mask_shift = (-spot_shift[0], -spot_shift[1])
    return mask_shift, spot_shift


# -------------------------------------------------------------------------------------------------
# the alignment step of the pipeline (step 3)
# -------------------------------------------------------------------------------------------------
def pad_to(arr, target_h, target_w):
    """Place a 2D array in the top-left corner of a (target_h, target_w) canvas (zero padded / cropped)."""
    arr = np.asarray(arr)
    out = np.zeros((target_h, target_w), arr.dtype)
    h, w = min(arr.shape[0], target_h), min(arr.shape[1], target_w)
    out[:h, :w] = arr[:h, :w]
    return out


def align_image(files, spots, previous=None, overwrite_corrected=False, n_iter=4000, max_shift=200, radius=20, cutoff=50):
    """
    Estimate the DIC -> fluorescence shift of one image from its spots and apply it to the DIC image and the
    Omnipose cell masks (files: buddy.paths.ImageFiles of the image folder).

    Writes aligned_DIC.tif and aligned_DIC_masks.tif. With overwrite_corrected, corrected/DIC_corrected_masks.tif
    (what spot counting reads) is written too: created from the shifted Omnipose masks when it does not exist;
    when it exists and was aligned before (`previous` = the alignment recorded in img.json), it is only moved by
    the change in shift since then (normally none), so running the step again never shifts it twice.

    The shift is always estimated on the Omnipose masks with a fixed random seed (same result on every run).
    Returns (alignment record for img.json, aligned cell masks, aligned DIC) or None when nothing could be aligned.
    """
    from skimage import io
    spots = np.asarray(spots, float)
    if not len(spots):
        log.warning("..no spots found, skipping alignment")
        return None
    if not files.cell_masks.is_file():
        log.warning("..no cell masks found, skipping alignment")
        return None
    cell_masks = io.imread(files.cell_masks)
    dic = io.imread(files.dic) if files.dic.is_file() else None

    # common canvas so that every spot coordinate indexes validly into every layer
    shapes = [cell_masks.shape] + ([dic.shape] if dic is not None else [])
    target_h = max([s[0] for s in shapes] + [int(spots[:, 1].max()) + 1])
    target_w = max([s[1] for s in shapes] + [int(spots[:, 2].max()) + 1])
    cell_masks = pad_to(cell_masks, target_h, target_w)
    if dic is not None:
        dic = pad_to(dic, target_h, target_w)

    mask_shift, spot_shift = estimate_alignment_shift(spots, cell_masks, n_iter=n_iter, max_shift=max_shift,
                                                      radius=radius, cutoff=cutoff, rng=np.random.default_rng(0))
    dy, dx = mask_shift
    log.info("..spots->mask shift %s, applying (dy=%d, dx=%d) to DIC + cell masks (cell mask source: pp)", spot_shift, dy, dx)
    record = {"mask_shift": [int(dy), int(dx)], "spot_shift": [int(spot_shift[0]), int(spot_shift[1])],
              "cell_mask_source": "pp", "target_shape": [int(target_h), int(target_w)], "n_spots": int(len(spots))}

    move = lambda a, s: ndi_shift(a, shift=s, order=0, mode="constant", cval=0)
    aligned_masks = move(cell_masks, (dy, dx))
    io.imsave(files.aligned_cell_masks, aligned_masks, check_contrast=False)
    aligned_dic = None
    if dic is not None:
        aligned_dic = move(dic, (dy, dx))
        io.imsave(files.aligned_dic, aligned_dic, check_contrast=False)

    if overwrite_corrected:
        files.corrected.mkdir(parents=True, exist_ok=True)
        dst = files.corrected_cells
        if not dst.is_file():
            io.imsave(dst, aligned_masks, check_contrast=False)
            log.info("..wrote aligned cell masks to corrected/")
        elif previous is not None and previous.get("cell_mask_source") == "pp":
            ddy, ddx = dy - previous["mask_shift"][0], dx - previous["mask_shift"][1]
            if ddy or ddx:
                io.imsave(dst, move(pad_to(io.imread(dst), target_h, target_w), (ddy, ddx)), check_contrast=False)
                log.info(f"..moved corrected cell masks by the change in shift ({ddy}, {ddx})")
            else:
                log.info("..corrected cell masks already aligned, left as they are")
        elif previous is None:
            io.imsave(dst, move(pad_to(io.imread(dst), target_h, target_w), (dy, dx)), check_contrast=False)
            log.info("..shifted existing corrected cell masks into the fluorescence frame")
        else:
            log.warning("..corrected/DIC_corrected_masks.tif was aligned by an older buddy version (shift relative to "
                        "earlier corrected masks) - left as it is; delete it to rebuild it from the Omnipose masks")
    return record, aligned_masks, aligned_dic
