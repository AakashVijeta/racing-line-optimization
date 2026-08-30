import json, os
import numpy as np
from plot_track import make_oval

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


def preprocess_circuit(geojson_path, n_points=400):
    raw = load_geojson_circuit(geojson_path)
    ref_lon, ref_lat = raw[0]
    local_xy = project_to_local_xy(raw, ref_lon, ref_lat)
    deduped = deduplicate_points(local_xy)
    return resample_even_spacing(deduped, n_points)

def build_track_pool(track_ids, tracks_dir="tracks"):
    pool = {"oval": make_oval()}
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
        centerline = preprocess_circuit(geojson_path, n_points=n_points)
        np.save(npy_path, centerline)
        print(f"Prepared {track_id}: {centerline.shape[0]} points")

# if __name__ == "__main__":
#     circuit_id = "de-1927"  # Nürburgring GP-Strecke
#     centerline = preprocess_circuit(f"circuits/{circuit_id}.geojson")
#     np.save(f"tracks/{circuit_id}.npy", centerline)
#     print(f"Saved {circuit_id}: {centerline.shape[0]} points")