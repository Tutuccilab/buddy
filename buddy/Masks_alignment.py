# ================================================
# Spot Alignment Functions
# ================================================
# This module provides functions to align a set of points ("spots") to a 2D reference
# distribution using a combination of global Monte Carlo search and local refinement.
# Useful in image analysis, microscopy, or any spatial registration task.

# Sander van Otterdijk, 09-03-2026
# (distance-field fix + convenience wrappers, 08-2026)
# ================================================

# ----------------------------
# Imports
# ----------------------------
import numpy as np
from scipy.ndimage import distance_transform_edt


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
