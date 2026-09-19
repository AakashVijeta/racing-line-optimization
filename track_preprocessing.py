import json, os
import numpy as np


def load_geojson_circuit(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)

    feature = data["features"][0]
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


def compute_curvature(points):
    """Compute curvature at each point using finite differences.
    
    Curvature = |dtheta/ds| where theta is the heading angle and s is arc length.
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
    curvature_raw = np.abs(dtheta) / np.maximum(ds, 1e-6)
    
    # Assign curvature to each point (average of adjacent segment curvatures)
    curvature = np.zeros(n)
    curvature[0] = curvature_raw[0]
    curvature[-1] = curvature_raw[-1]
    curvature[1:-1] = (curvature_raw[:-1] + curvature_raw[1:]) / 2.0
    
    return curvature


def resample_adaptive(points, base_n_points=400, curvature_weight=3.0):
    """Curvature-adaptive resampling: places more points where curvature is high.
    
    Algorithm:
    1. Compute curvature at each raw point
    2. Build a density function: density(s) = 1 + curvature_weight * normalized_curvature(s)
    3. Integrate density to get a CDF, then sample uniformly from the CDF
    
    This gives ~3-4x more points at hairpins vs straights while keeping total
    point count manageable.
    
    Args:
        points: (N, 2) array of centerline coordinates
        base_n_points: base number of output points (auto-scaled up for complex tracks)
        curvature_weight: how much to amplify point density at high-curvature regions
    
    Returns:
        (M, 2) array of resampled centerline points
    """
    # Step 0: First do a fine uniform resampling to get smooth curvature estimates
    fine_n = max(2000, len(points) * 4)
    fine_points = resample_even_spacing(points, fine_n)
    
    # Step 1: Compute curvature on the fine mesh
    curvature = compute_curvature(fine_points)
    
    # Smooth the curvature with a moving average to avoid noise spikes
    kernel_size = max(5, fine_n // 100)
    kernel = np.ones(kernel_size) / kernel_size
    curvature_smooth = np.convolve(curvature, kernel, mode='same')
    
    # Step 2: Determine total output points based on track complexity
    # If the track has very tight corners (min radius < 15m), use more points
    max_curvature = np.max(curvature_smooth)
    min_radius = 1.0 / max(max_curvature, 1e-6)
    
    if min_radius < 8.0:
        n_points = int(base_n_points * 2.5)  # Very tight (Monaco-like): 1000 points
    elif min_radius < 15.0:
        n_points = int(base_n_points * 2.0)  # Tight corners: 800 points
    elif min_radius < 25.0:
        n_points = int(base_n_points * 1.5)  # Moderate corners: 600 points
    else:
        n_points = base_n_points  # Simple track: 400 points
    
    # Step 3: Build density function
    # Normalize curvature to [0, 1] range
    curv_max = np.max(curvature_smooth)
    if curv_max > 1e-6:
        curv_normalized = curvature_smooth / curv_max
    else:
        curv_normalized = np.zeros_like(curvature_smooth)
    
    # Density = base + weight * curvature (more points where curvature is high)
    density = 1.0 + curvature_weight * curv_normalized
    
    # Step 4: Compute arc lengths on fine mesh
    ds = np.linalg.norm(np.diff(fine_points, axis=0), axis=1)
    arc_lengths = np.concatenate([[0], np.cumsum(ds)])
    
    # Step 5: Build weighted CDF
    # weighted_ds[i] = density[i] * ds[i]  (how much "sampling weight" each segment gets)
    weighted_ds = density[:-1] * ds
    weighted_cumsum = np.concatenate([[0], np.cumsum(weighted_ds)])
    weighted_total = weighted_cumsum[-1]
    
    # Step 6: Sample n_points uniformly from the weighted CDF
    target_weights = np.linspace(0, weighted_total, n_points, endpoint=False)
    
    # Use proper interpolation to avoid duplicate points
    # Map target weights back to arc-length positions, then interpolate x,y
    target_arc_positions = np.interp(target_weights, weighted_cumsum, arc_lengths)
    
    # Interpolate x and y from the fine mesh using arc-length positions
    result_x = np.interp(target_arc_positions, arc_lengths, fine_points[:, 0])
    result_y = np.interp(target_arc_positions, arc_lengths, fine_points[:, 1])
    result = np.stack([result_x, result_y], axis=1)
    
    return result


def preprocess_circuit(geojson_path, n_points=400, adaptive=True):
    raw = load_geojson_circuit(geojson_path)
    ref_lon, ref_lat = raw[0]
    local_xy = project_to_local_xy(raw, ref_lon, ref_lat)
    deduped = deduplicate_points(local_xy)
    
    if adaptive:
        return resample_adaptive(deduped, base_n_points=n_points)
    else:
        return resample_even_spacing(deduped, n_points)


def build_track_pool(track_ids, tracks_dir="tracks"):
    pool = {}
    for track_id in track_ids:
        pool[track_id] = np.load(f"{tracks_dir}/{track_id}.npy")
    return pool


def prepare_tracks(track_ids, circuits_dir="circuits", tracks_dir="tracks", n_points=400):
    os.makedirs(tracks_dir, exist_ok=True)
    for track_id in track_ids:
        npy_path = f"{tracks_dir}/{track_id}.npy"
        if os.path.exists(npy_path):
            continue
        geojson_path = f"{circuits_dir}/{track_id}.geojson"
        centerline = preprocess_circuit(geojson_path, n_points=n_points, adaptive=True)
        np.save(npy_path, centerline)
        print(f"Prepared {track_id}: {centerline.shape[0]} points")

# if __name__ == "__main__":
#     circuit_id = "de-1927"  # Nürburgring GP-Strecke
#     centerline = preprocess_circuit(f"circuits/{circuit_id}.geojson")
#     np.save(f"tracks/{circuit_id}.npy", centerline)
#     print(f"Saved {circuit_id}: {centerline.shape[0]} points")