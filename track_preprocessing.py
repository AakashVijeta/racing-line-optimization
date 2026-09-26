import json, os
import numpy as np

# Bump whenever preprocessing output changes, so cached tracks/*.npy are rebuilt
PREPROCESS_VERSION = 2  # 2: spline + spacing-based point count; drop duplicate closing point
MANIFEST_NAME = "manifest.json"


def load_geojson_circuit(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)

    # Accept a FeatureCollection (the circuits/ files) or a bare Feature
    feature = data["features"][0] if data.get("type") == "FeatureCollection" else data
    if feature["geometry"]["type"] == "LineString":
        coords = feature["geometry"]["coordinates"]
        return np.array(coords)
    raise ValueError("Geometry is not a LineString")


def project_to_local_xy(lon_lat_points, ref_lon, ref_lat):
    lon_lat_points = np.asarray(lon_lat_points, dtype=float)
    lon = lon_lat_points[:, 0]
    lat = lon_lat_points[:, 1]
    x = (lon - ref_lon) * 111_320 * np.cos(np.radians(ref_lat))
    y = (lat - ref_lat) * 111_320
    return np.stack([x, y], axis=1)


def deduplicate_points(points, min_gap=1e-6):
    gaps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep_mask = np.concatenate([[True], gaps > min_gap])
    points = points[keep_mask]
    return points


def resample_even_spacing(points, n_points):
    arc_length = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    total_distance = arc_length[-1]
    target_distances = np.linspace(0, total_distance, n_points)
    x_interp = np.interp(target_distances, arc_length, points[:, 0])
    y_interp = np.interp(target_distances, arc_length, points[:, 1])
    return np.stack([x_interp, y_interp], axis=1)


def compute_curvature(points, signed=False):
    """Compute curvature at each point using finite differences.
    
    Curvature = |dtheta/ds| where theta is the heading angle and s is arc length.
    With signed=True, returns dtheta/ds instead: positive for left (counter-clockwise) turns.
    Returns an array of curvature values, one per point.
    """
    n = len(points)
    # Direction vectors (closed loop)
    dirs = np.diff(points, axis=0)
    closing_vec = points[0] - points[-1]
    dirs = np.vstack([dirs, closing_vec])
    
    # Headings
    angles = np.arctan2(dirs[:, 1], dirs[:, 0])
    
    # Change in heading between consecutive segments
    dtheta = np.diff(angles)
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    
    # Arc lengths between consecutive points
    ds = np.linalg.norm(np.diff(points, axis=0), axis=1)
    
    # Curvature = |dtheta| / ds at each interior transition
    curvature_raw = (dtheta if signed else np.abs(dtheta)) / np.maximum(ds, 1e-6)
    
    # Assign curvature to each point (average of adjacent segment curvatures)
    curvature = np.zeros(n)
    curvature[0] = curvature_raw[0]
    curvature[-1] = curvature_raw[-1]
    curvature[1:-1] = (curvature_raw[:-1] + curvature_raw[1:]) / 2.0
    
    return curvature


def open_loop(points, tol=1e-6):
    """Drop the last point if it repeats the first (GeoJSON loops close explicitly).

    Closed-loop maths wraps from the last point back to the first; a repeated
    point would add a zero-length segment and a fake curvature spike there.
    """
    if len(points) > 2 and np.linalg.norm(points[0] - points[-1]) < tol:
        return points[:-1]
    return points


def catmull_rom_closed(points, step=0.5, alpha=0.5):
    """Densely sample a smooth closed curve through every input point.

    Centripetal Catmull-Rom (alpha=0.5) passes through the raw GeoJSON points
    without overshoot or cusps, even where point spacing varies a lot.
    The raw data is coarse (~20 m between points), so a plain polyline would
    have zero curvature along each segment and a kink at every vertex.
    """
    p = np.asarray(points, dtype=float)
    n = len(p)
    out = []
    for i in range(n):
        p0, p1, p2, p3 = p[i - 1], p[i], p[(i + 1) % n], p[(i + 2) % n]
        t1 = np.linalg.norm(p1 - p0) ** alpha
        t2 = t1 + np.linalg.norm(p2 - p1) ** alpha
        t3 = t2 + np.linalg.norm(p3 - p2) ** alpha
        m = max(1, int(np.ceil(np.linalg.norm(p2 - p1) / step)))
        t = np.linspace(t1, t2, m, endpoint=False)[:, None]
        # Barry-Goldman pyramidal formulation
        a1 = (t1 - t) / t1 * p0 + t / t1 * p1
        a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        b1 = (t2 - t) / t2 * a1 + t / t2 * a2
        b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
        out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    return np.vstack(out)


def circular_moving_average(values, window):
    """Moving average that wraps around a closed loop (no edge fall-off)."""
    window = max(1, int(window))
    pad = window // 2
    padded = np.concatenate([values[-pad:], values, values[:pad]]) if pad else values
    smoothed = np.convolve(padded, np.ones(window) / window, mode="same")
    return smoothed[pad:len(smoothed) - pad] if pad else smoothed


def resample_adaptive(points, straight_spacing=5.0, curvature_weight=3.0, smooth_m=20.0):
    """Curvature-adaptive resampling of a closed loop.

    Point density is 1 + curvature_weight * normalized_curvature, and the point
    count is chosen so straights get `straight_spacing` metres between points;
    the tightest corners get (1 + curvature_weight) times denser.

    Args:
        points: (N, 2) closed-loop centerline, first point not repeated at the end
        straight_spacing: target spacing (m) where curvature is zero
        curvature_weight: extra density at the track's highest curvature
        smooth_m: length (m) of the curvature smoothing window

    Returns:
        (M, 2) array of resampled centerline points (not repeating the first)
    """
    fine = catmull_rom_closed(points)
    ds = np.linalg.norm(np.diff(np.vstack([fine, fine[:1]]), axis=0), axis=1)  # includes closing segment
    arc = np.concatenate([[0], np.cumsum(ds)])

    curvature = compute_curvature(fine)
    curvature = circular_moving_average(curvature, smooth_m / np.mean(ds))
    curv_max = curvature.max()
    curv_normalized = curvature / curv_max if curv_max > 1e-6 else np.zeros_like(curvature)
    density = 1.0 + curvature_weight * curv_normalized

    # Weighted CDF over every segment, including the closing one
    weighted = np.concatenate([[0], np.cumsum(density * ds)])
    n_points = int(np.ceil(weighted[-1] / straight_spacing))
    targets = np.linspace(0, weighted[-1], n_points, endpoint=False)
    target_arc = np.interp(targets, weighted, arc)

    loop = np.vstack([fine, fine[:1]])
    return np.stack([np.interp(target_arc, arc, loop[:, 0]),
                     np.interp(target_arc, arc, loop[:, 1])], axis=1)


def preprocess_circuit(geojson_path, n_points=400, adaptive=True, straight_spacing=5.0):
    """GeoJSON circuit -> (M, 2) centerline in metres.

    adaptive=True: smooth spline, curvature-adaptive spacing (straight_spacing on straights).
    adaptive=False: plain polyline resampled to exactly n_points evenly spaced points.
    """
    raw = load_geojson_circuit(geojson_path)
    ref_lon, ref_lat = raw[0]
    local_xy = project_to_local_xy(raw, ref_lon, ref_lat)
    deduped = deduplicate_points(local_xy)

    if adaptive:
        return resample_adaptive(open_loop(deduped), straight_spacing=straight_spacing)
    else:
        return resample_even_spacing(deduped, n_points)


def build_track_pool(track_ids, tracks_dir="tracks"):
    pool = {}
    for track_id in track_ids:
        pool[track_id] = np.load(f"{tracks_dir}/{track_id}.npy")
    return pool


def prepare_tracks(track_ids, circuits_dir="circuits", tracks_dir="tracks", straight_spacing=5.0, force=False):
    """Build tracks/<id>.npy for each track that is missing or stale.

    A track is stale when the manifest records a different preprocessing
    version or spacing than the current call would produce.
    """
    os.makedirs(tracks_dir, exist_ok=True)
    manifest_path = os.path.join(tracks_dir, MANIFEST_NAME)
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

    wanted = {"version": PREPROCESS_VERSION, "straight_spacing": straight_spacing}
    changed = False
    for track_id in track_ids:
        npy_path = f"{tracks_dir}/{track_id}.npy"
        if not force and os.path.exists(npy_path) and manifest.get(track_id) == wanted:
            continue
        geojson_path = f"{circuits_dir}/{track_id}.geojson"
        centerline = preprocess_circuit(geojson_path, adaptive=True, straight_spacing=straight_spacing)
        np.save(npy_path, centerline)
        manifest[track_id] = wanted
        changed = True
        print(f"Prepared {track_id}: {centerline.shape[0]} points")

    if changed:
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)

if __name__ == "__main__":
    import argparse
    from config import TRACK_IDS

    parser = argparse.ArgumentParser(description="Preprocess circuits/*.geojson into tracks/*.npy")
    parser.add_argument("tracks", nargs="*", default=TRACK_IDS, help="track IDs (default: all)")
    parser.add_argument("--force", action="store_true", help="rebuild even if the cache is current")
    args = parser.parse_args()
    prepare_tracks(args.tracks, force=args.force)