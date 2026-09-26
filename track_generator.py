"""
Procedural race-track generator.

Tracks are built like the real ones: control points -> centripetal Catmull-Rom
spline -> curvature-adaptive resampling (see track_preprocessing.py), then
rejected unless they are drivable (no self-intersection, no corners tighter
than MIN_RADIUS, no two parts of the track closer than MIN_SEPARATION).

    from track_generator import generate_pool
    tracks, widths = generate_pool(n=400, seed=0, prefix="proc")
"""
import numpy as np

from track_preprocessing import catmull_rom_closed, resample_adaptive

MIN_RADIUS = 9.0          # m, tightest allowed corner (15 m chord scale); Monaco's hairpin is ~10 m
MIN_SEPARATION = 20.0     # m, between parts of the track more than SEPARATION_ARC apart
SEPARATION_ARC = 80.0     # m
LENGTH_RANGE = (3000.0, 7000.0)
WIDTH_RANGE = (8.0, 16.0)


def _arc(points):
    ds = np.linalg.norm(np.diff(np.vstack([points, points[:1]]), axis=0), axis=1)
    return np.concatenate([[0], np.cumsum(ds)])


def _sample(points, s):
    """Positions at arc lengths s along a closed polyline."""
    arc = _arc(points)
    loop = np.vstack([points, points[:1]])
    s = np.mod(s, arc[-1])
    return np.stack([np.interp(s, arc, loop[:, 0]), np.interp(s, arc, loop[:, 1])], axis=1)


def radius_profile(points, chord=15.0, step=1.0):
    """Corner radius every `step` metres, from the circle through points `chord` m apart.

    Measuring over a chord (instead of vertex-to-vertex) makes the result
    independent of how densely the centerline is sampled.
    """
    s = np.arange(0.0, _arc(points)[-1], step)
    a, b, c = _sample(points, s - chord), _sample(points, s), _sample(points, s + chord)
    ab, bc, ac = (np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1),
                  np.linalg.norm(c - a, axis=1))
    cross = np.abs((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])
    return ab * bc * ac / np.maximum(2.0 * cross, 1e-9)


def _self_intersects(points):
    """True if any two non-adjacent segments of the closed polyline cross."""
    p = points
    q = np.roll(points, -1, axis=0)
    n = len(p)
    i, j = np.triu_indices(n, k=2)
    keep = ~((i == 0) & (j == n - 1))  # first and last segments share a vertex
    i, j = i[keep], j[keep]

    def orient(a, b, c):
        return np.sign((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))

    o1 = orient(p[i], q[i], p[j])
    o2 = orient(p[i], q[i], q[j])
    o3 = orient(p[j], q[j], p[i])
    o4 = orient(p[j], q[j], q[i])
    return bool(np.any((o1 != o2) & (o3 != o4)))


def _min_separation(points, arc_gap=SEPARATION_ARC):
    """Smallest distance between two points more than arc_gap apart along the track."""
    arc = _arc(points)[:-1]
    length = _arc(points)[-1]
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    along = np.abs(arc[:, None] - arc[None, :])
    along = np.minimum(along, length - along)
    d[along <= arc_gap] = np.inf
    return d.min()


def is_valid(centerline):
    """Check a candidate track is drivable. Returns (ok, reason)."""
    coarse = _sample(centerline, np.arange(0.0, _arc(centerline)[-1], 10.0))
    length = _arc(centerline)[-1]
    if not LENGTH_RANGE[0] <= length <= LENGTH_RANGE[1]:
        return False, "length"
    if _self_intersects(coarse):
        return False, "intersects"
    if radius_profile(centerline).min() < MIN_RADIUS:
        return False, "too_tight"
    if _min_separation(coarse) < MIN_SEPARATION:
        return False, "too_close"
    return True, "ok"


GEN = dict(n=(18, 33), radius=(350.0, 800.0), noise=0.4, stretch=(1.0, 2.2),
           sharp_frac=(0.3, 0.6), sharp_offset=(8.0, 30.0))


def _control_points(rng, gen=GEN):
    """Random control polygon: a noisy, stretched loop with tight corners added."""
    n = int(rng.integers(*gen["n"]))
    angles = np.sort((np.arange(n) + rng.uniform(-0.35, 0.35, n)) * 2 * np.pi / n)
    radius = rng.uniform(*gen["radius"])
    r = radius * (1.0 + rng.uniform(-gen["noise"], gen["noise"], n))
    pts = np.stack([r * np.cos(angles), r * np.sin(angles)], axis=1)
    pts[:, 0] *= rng.uniform(*gen["stretch"])          # elongate: longer straights

    # Tight corners: replace a control point with two close points, which
    # the spline turns into a short, sharp bend (hairpins, chicanes)
    out = []
    sharp = rng.random(n) < rng.uniform(*gen["sharp_frac"])
    for k in range(n):
        if sharp[k]:
            prev_pt, next_pt = pts[k - 1], pts[(k + 1) % n]
            t = next_pt - prev_pt
            t /= np.linalg.norm(t)
            offset = rng.uniform(*gen["sharp_offset"])
            out.append(pts[k] - t * offset)
            out.append(pts[k] + t * offset)
        else:
            out.append(pts[k])
    pts = np.array(out)

    theta = rng.uniform(0, 2 * np.pi)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return pts @ rot.T


def generate_track(rng, max_tries=200):
    """One valid random track as an (M, 2) centerline, like tracks/<id>.npy."""
    for _ in range(max_tries):
        ctrl = _control_points(rng)
        centerline = resample_adaptive(ctrl)
        ok, _ = is_valid(centerline)
        if ok:
            return centerline
    raise RuntimeError(f"No valid track after {max_tries} attempts")


def generate_pool(n, seed, prefix="proc"):
    """n tracks with reproducible IDs and widths: ({id: centerline}, {id: width})."""
    rng = np.random.default_rng(seed)
    tracks, widths = {}, {}
    for k in range(n):
        tid = f"{prefix}-{seed}-{k:04d}"
        tracks[tid] = generate_track(rng)
        widths[tid] = float(np.round(rng.uniform(*WIDTH_RANGE), 1))
    return tracks, widths


def procedural_split(split):
    """The fixed procedural pool for 'train', 'val' or 'test' (see config.PROC_SEEDS)."""
    from config import PROC_SEEDS, PROC_SIZES
    return generate_pool(PROC_SIZES[split], seed=PROC_SEEDS[split])
