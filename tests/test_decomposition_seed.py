"""The decomposition of dense regions is repeatable: Big-FISH builds its reference spot from up to 2000 randomly
chosen spots, so with more spots than that the result depended on NumPy's random state until DECOMPOSITION_SEED."""
import numpy as np
import pytest

from buddy.spots import detection

SCALE, SPOT_RADIUS = (200, 65, 65), (880, 120, 120)


def synthetic_field(n_spots=2500, shape=(21, 300, 300), seed=1):
    """a noisy stack with n_spots Gaussian spots of varying brightness (more than Big-FISH's 2000-spot sample)"""
    rng = np.random.default_rng(seed)
    z = rng.integers(6, shape[0] - 6, n_spots)          # Big-FISH only uses spots >= 5 slices from the top and bottom
    y = rng.integers(6, shape[1] - 6, n_spots)
    x = rng.integers(6, shape[2] - 6, n_spots)
    stack = rng.normal(100, 5, shape)
    zz, yy, xx = np.mgrid[-3:4, -6:7, -6:7]
    blob = np.exp(-(zz ** 2) / (2 * 1.5 ** 2) - (yy ** 2 + xx ** 2) / (2 * 1.8 ** 2))
    for zi, yi, xi, a in zip(z, y, x, rng.uniform(50, 400, n_spots)):
        stack[zi - 3:zi + 4, yi - 6:yi + 7, xi - 6:xi + 7] += a * blob
    return np.clip(stack, 0, 65535).astype(np.uint16), np.stack([z, y, x], axis=1).astype(np.int64)


def test_decomposition_is_repeatable_and_leaves_random_state_alone():
    stack, spots = synthetic_field()
    results = []
    for outside_seed in (11, 22):                  # different random states before the call
        np.random.seed(outside_seed)
        before = np.random.get_state()[1].copy()
        decomposed, dense, reference, _ = detection.decompose_spots(stack, spots, SCALE, SPOT_RADIUS, device=None)
        assert np.array_equal(np.random.get_state()[1], before)       # the caller's random state is restored
        results.append((decomposed, dense, reference))
    (d1, r1, ref1), (d2, r2, ref2) = results
    assert np.array_equal(ref1, ref2)
    assert np.array_equal(d1, d2)
    assert np.array_equal(r1, r2)


def test_gpu_decomposition_accepts_a_float_gamma(monkeypatch):
    """gamma from the config may be a float (1.0, 1.5); the GPU version must scale the kernel per axis, as Big-FISH does."""
    from bigfish.detection.utils import get_object_radius_pixel

    from buddy.spots import gpu_detection
    seen = {}

    def fake_remove_background(image, sigma, device="cuda"):
        seen["sigma"] = sigma
        return image

    monkeypatch.setattr(gpu_detection, "remove_background", fake_remove_background)
    stack = (np.random.default_rng(0).random((5, 40, 40)) * 1000).astype(np.uint16)
    spots = np.array([[2, 10, 10], [2, 30, 30]])
    gpu_detection.decompose_dense(stack, spots, SCALE, SPOT_RADIUS, alpha=0.5, beta=2.0, gamma=1.5, device="cpu")
    radius = get_object_radius_pixel(voxel_size_nm=SCALE, object_radius_nm=SPOT_RADIUS, ndim=3)
    assert len(seen["sigma"]) == 3
    assert seen["sigma"] == pytest.approx(tuple(r * 1.5 for r in radius))
