"""
DIC lighting direction, so every image can be turned the same way before the bud model sees it.

A DIC image looks as if every cell is lit from one side. Which side depends on the microscope
(prism orientation), so it is the same for all cells in one image but can differ between imaging
sessions. The bud model was trained with all images turned (in steps of 90 degrees) so that the
bright side points the same way. These functions measure the direction and pick that rotation.

    angle = dic_light_direction(dic, cell_masks)     # degrees
    k = rotation_to_reference(angle)                  # use: np.rot90(image, k)
"""
import numpy as np
from scipy import ndimage as ndi

REFERENCE_DEG = 135.0     # lighting direction of most training images

# 90-degree rotations and the mirror operations that keep the lighting direction
# (only these are used to augment the training images; the others create DIC images
# that a microscope never produces)
TRANSFORMS = {
    "identity":       lambda a: a,
    "rot90":          lambda a: np.rot90(a, 1, axes=(-2, -1)),
    "rot180":         lambda a: np.rot90(a, 2, axes=(-2, -1)),
    "rot270":         lambda a: np.rot90(a, 3, axes=(-2, -1)),
    "flip_lr":        lambda a: a[..., ::-1],
    "flip_ud":        lambda a: a[..., ::-1, :],
    "transpose":      lambda a: np.swapaxes(a, -1, -2),
    "anti_transpose": lambda a: np.rot90(np.swapaxes(a, -1, -2), 2, axes=(-2, -1)),
}


def dic_light_direction(dic, cell_masks, min_px=500, max_cells=300):
    """
    Average direction of the bright side of the cells, in degrees (image coordinates, y down).
    Returns (angle, consistency 0-1, number of cells used). Any instance mask works
    (Omnipose output or curated masks).
    """
    dic = np.asarray(dic, np.float32)
    cell_masks = np.asarray(cell_masks)
    vectors = []
    for label, box in enumerate(ndi.find_objects(cell_masks), start=1):
        if box is None:
            continue
        box = tuple(slice(max(s.start - 4, 0), s.stop + 4) for s in box)
        cell = ndi.binary_dilation(cell_masks[box] == label, iterations=3)
        if cell.sum() < min_px:
            continue
        yy, xx = np.nonzero(cell)
        intensity = dic[box][cell] - dic[box][cell].mean()
        dy, dx = yy - yy.mean(), xx - xx.mean()
        r = np.hypot(dy, dx) + 1e-6
        v = np.array([(intensity * dy / r).mean(), (intensity * dx / r).mean()])
        vectors.append(v / (np.linalg.norm(v) + 1e-9))
        if len(vectors) >= max_cells:
            break
    if not vectors:
        return np.nan, 0.0, 0
    mean = np.mean(vectors, axis=0)
    return float(np.degrees(np.arctan2(mean[0], mean[1]))), float(np.linalg.norm(mean)), len(vectors)


def _angle_difference(a, b):
    return abs((a - b + 180) % 360 - 180)


def _angle_after(transform_name, angle):
    """Direction (degrees) after applying a transform, measured on a test ramp image."""
    yy, xx = np.mgrid[-20:21, -20:21].astype(float)
    ramp = np.sin(np.radians(angle)) * yy + np.cos(np.radians(angle)) * xx
    gy, gx = np.gradient(TRANSFORMS[transform_name](ramp))
    return float(np.degrees(np.arctan2(gy.mean(), gx.mean())))


def rotation_to_reference(angle, reference=REFERENCE_DEG):
    """k (0-3) such that np.rot90(image, k) turns the lighting direction closest to the reference."""
    if np.isnan(angle):
        return 0
    errors = [_angle_difference(_angle_after(name, angle), reference) for name in ("identity", "rot90", "rot180", "rot270")]
    return int(np.argmin(errors))


def lighting_preserving_transforms(reference=REFERENCE_DEG, tolerance=20):
    """Names of the transforms that keep the reference lighting direction (for training augmentation)."""
    return [name for name in TRANSFORMS if _angle_difference(_angle_after(name, reference), reference) <= tolerance]
