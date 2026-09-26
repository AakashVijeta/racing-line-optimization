import numpy as np

from track_generator import generate_pool, is_valid, radius_profile, MIN_RADIUS
from track_preprocessing import resample_adaptive


def test_generate_pool_is_reproducible_and_valid():
    a, widths_a = generate_pool(5, seed=3)
    b, widths_b = generate_pool(5, seed=3)
    assert list(a) == list(b)
    assert widths_a == widths_b
    for tid, centerline in a.items():
        np.testing.assert_array_equal(centerline, b[tid])
        ok, reason = is_valid(centerline)
        assert ok, reason
        assert 8.0 <= widths_a[tid] <= 16.0


def test_different_seeds_give_different_tracks():
    a, _ = generate_pool(1, seed=1)
    b, _ = generate_pool(1, seed=2)
    assert not np.array_equal(next(iter(a.values())), next(iter(b.values())))


def test_radius_profile_of_a_circle():
    angles = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    circle = 200.0 * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    np.testing.assert_allclose(radius_profile(circle), 200.0, rtol=0.01)


def test_rejects_self_intersecting_figure_eight():
    t = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    eight = np.stack([800 * np.sin(t), 400 * np.sin(2 * t)], axis=1)  # ~4.4 km, within length limits
    ok, reason = is_valid(resample_adaptive(eight))
    assert not ok and reason == "intersects"


def test_rejects_corners_tighter_than_minimum():
    angles = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    small = (MIN_RADIUS * 0.7) * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    ok, reason = is_valid(resample_adaptive(small))
    assert not ok and reason in ("too_tight", "length")
