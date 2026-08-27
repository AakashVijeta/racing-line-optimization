import json
import numpy as np
import pytest

from track_preprocessing import (
    load_geojson_circuit,
    project_to_local_xy,
    deduplicate_points,
    resample_even_spacing,
    preprocess_circuit,
)


# ---------------------------------------------------------------------------
# load_geojson_circuit
# ---------------------------------------------------------------------------

def test_load_geojson_circuit_reads_linestring(tmp_path):
    geojson = {
        "type": "Feature",
        "properties": {"id": "test-track"},
        "geometry": {
            "type": "LineString",
            "coordinates": [[6.0, 50.0], [6.001, 50.001], [6.002, 50.0]],
        },
    }
    filepath = tmp_path / "test-track.geojson"
    filepath.write_text(json.dumps(geojson))

    coords = load_geojson_circuit(str(filepath))

    assert coords.shape == (3, 2)
    np.testing.assert_allclose(coords[0], [6.0, 50.0])


def test_load_geojson_circuit_rejects_non_linestring(tmp_path):
    geojson = {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [6.0, 50.0]},
    }
    filepath = tmp_path / "bad-track.geojson"
    filepath.write_text(json.dumps(geojson))

    with pytest.raises(ValueError):
        load_geojson_circuit(str(filepath))


# ---------------------------------------------------------------------------
# project_to_local_xy
# ---------------------------------------------------------------------------

def test_project_reference_point_maps_to_origin():
    points = np.array([[6.943, 50.334]])
    result = project_to_local_xy(points, ref_lon=6.943, ref_lat=50.334)
    np.testing.assert_allclose(result[0], [0.0, 0.0], atol=1e-9)


def test_project_known_offset_distance():
    # One point exactly 0.001 deg north and 0.001 deg east of the reference,
    # with a reference latitude of 0 (equator) so cos(ref_lat) == 1 and the
    # x/y scale factors are identical and easy to check by hand.
    points = np.array([[0.001, 0.001]])
    result = project_to_local_xy(points, ref_lon=0.0, ref_lat=0.0)
    expected = 0.001 * 111_320
    np.testing.assert_allclose(result[0], [expected, expected], rtol=1e-6)


def test_project_does_not_mutate_input():
    points = np.array([[6.943, 50.334], [6.944, 50.335]])
    original = points.copy()
    project_to_local_xy(points, ref_lon=6.943, ref_lat=50.334)
    np.testing.assert_array_equal(points, original)


# ---------------------------------------------------------------------------
# deduplicate_points
# ---------------------------------------------------------------------------

def test_deduplicate_removes_exact_duplicate():
    points = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 0.0]])
    result = deduplicate_points(points)
    assert len(result) == 2
    np.testing.assert_allclose(result, [[0.0, 0.0], [10.0, 0.0]])


def test_deduplicate_keeps_well_separated_points():
    points = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    result = deduplicate_points(points)
    assert len(result) == 3


def test_deduplicate_handles_negative_direction_of_travel():
    # Regression check: points moving in the -x direction (as any point on a
    # closed loop eventually will) must NOT be treated as near-duplicates
    # just because the raw coordinate difference is negative.
    points = np.array([[20.0, 0.0], [10.0, 0.0], [0.0, 0.0]])
    result = deduplicate_points(points)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# resample_even_spacing
# ---------------------------------------------------------------------------

def test_resample_returns_requested_point_count():
    points = np.array([[0.0, 0.0], [10.0, 0.0]])
    result = resample_even_spacing(points, n_points=5)
    assert result.shape == (5, 2)


def test_resample_preserves_endpoints_on_a_straight_line():
    points = np.array([[0.0, 0.0], [10.0, 0.0]])
    result = resample_even_spacing(points, n_points=5)
    np.testing.assert_allclose(result[0], [0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(result[-1], [10.0, 0.0], atol=1e-9)


def test_resample_produces_even_spacing():
    points = np.array([[0.0, 0.0], [10.0, 0.0]])
    result = resample_even_spacing(points, n_points=5)
    segment_lengths = np.linalg.norm(np.diff(result, axis=0), axis=1)
    np.testing.assert_allclose(segment_lengths, 2.5, atol=1e-9)


def test_resample_handles_unevenly_spaced_input():
    # Same straight line, but the raw points are clustered unevenly —
    # resample_even_spacing should still produce evenly spaced output.
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.5, 0.0], [10.0, 0.0]])
    result = resample_even_spacing(points, n_points=5)
    segment_lengths = np.linalg.norm(np.diff(result, axis=0), axis=1)
    np.testing.assert_allclose(segment_lengths, 2.5, atol=1e-9)


def test_resample_preserves_total_length_on_a_square():
    # A closed square, 10 units per side, 40 units total perimeter.
    points = np.array([
        [0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0],
    ])
    # rel=1e-2, not 1e-3: linear interpolation slightly "cuts" each of the
    # square's 4 sharp corners unless a sample lands exactly on the vertex,
    # so total length is expected to come in a little under 40, not exactly.
    result = resample_even_spacing(points, n_points=100)
    total_length = np.sum(np.linalg.norm(np.diff(result, axis=0), axis=1))
    assert total_length == pytest.approx(40.0, rel=1e-2)


# ---------------------------------------------------------------------------
# preprocess_circuit (integration)
# ---------------------------------------------------------------------------

def test_preprocess_circuit_end_to_end(tmp_path):
    # A small synthetic closed track near the equator, sized to make the
    # expected metre length easy to check: a "square" 0.01 deg per side.
    # At the equator, 0.01 deg ~= 1113.2 m in both directions.
    geojson = {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [
                [0.0, 0.0], [0.01, 0.0], [0.01, 0.01], [0.0, 0.01], [0.0, 0.0],
            ],
        },
    }
    filepath = tmp_path / "square-track.geojson"
    filepath.write_text(json.dumps(geojson))

    centerline = preprocess_circuit(str(filepath), n_points=200)

    assert centerline.shape == (200, 2)

    total_length = np.sum(np.linalg.norm(np.diff(centerline, axis=0), axis=1))
    expected_perimeter = 4 * 0.01 * 111_320  # ~4452.8 m
    assert total_length == pytest.approx(expected_perimeter, rel=1e-2)


def test_preprocess_circuit_removes_duplicate_geojson_points(tmp_path):
    # Real GeoJSON tracks sometimes repeat the closing point exactly.
    # preprocess_circuit should still produce a clean, evenly spaced result.
    geojson = {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [
                [0.0, 0.0], [0.0, 0.0], [0.01, 0.0], [0.01, 0.0], [0.01, 0.01],
            ],
        },
    }
    filepath = tmp_path / "dup-track.geojson"
    filepath.write_text(json.dumps(geojson))

    centerline = preprocess_circuit(str(filepath), n_points=50)
    assert centerline.shape == (50, 2)
    assert not np.isnan(centerline).any()