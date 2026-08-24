import numpy as np


def make_straight(start_point, end_point, n_points):
    """
    Generate n_points evenly spaced points from start_point to end_point.
    start_point, end_point: tuples like (x, y)
    """
    x_start, y_start = start_point
    x_end, y_end = end_point

    # TODO: generate n_points evenly spaced x-values between x_start and x_end
    xs = np.linspace(x_start, x_end, n_points, endpoint=True)

    # TODO: same for y-values (for our bottom/top straights, y doesn't change,
    # but write it generally so this function works for any straight)
    ys = np.linspace(y_start, y_end, n_points, endpoint=True)

    # combine xs and ys into an (n_points, 2) array of points
    points = np.stack([xs, ys], axis=1)
    return points


def make_arc(center, radius, start_angle_deg, end_angle_deg, n_points):
    """
    Generate n_points evenly spaced points along a circular arc.
    center: tuple (cx, cy)
    radius: float
    start_angle_deg, end_angle_deg: sweep range in degrees
    """
    cx, cy = center

    # TODO: convert start/end angles from degrees to radians
    # hint: np.radians() or np.deg2rad()
    start_rad = np.radians(start_angle_deg)
    end_rad = np.radians(end_angle_deg)

    # TODO: generate n_points evenly spaced angles between start_rad and end_rad
    angles = np.linspace(start_rad, end_rad, n_points, endpoint=True)

    # TODO: apply the circle formula to each angle to get x, y
    xs = cx + radius * np.cos(angles)
    ys = cy + radius * np.sin(angles)

    points = np.stack([xs, ys], axis=1)
    return points


def make_oval(
    straight_length=200.0, radius=60.0, points_per_straight=50, points_per_arc=40
):
    half_length = straight_length / 2.0

    bottom_straight = make_straight(
        (-half_length, -radius), (half_length, -radius), points_per_straight
    )
    right_arc = make_arc((half_length, 0), radius, -90, 90, points_per_arc)
    top_straight = make_straight(
        (half_length, radius), (-half_length, radius), points_per_straight
    )
    left_arc = make_arc((-half_length, 0), radius, 90, 270, points_per_arc)

    centerline = np.concatenate(
        [bottom_straight, right_arc, top_straight, left_arc], axis=0
    )

    return centerline


def compute_boundaries(centerline, track_width):
    centerline_array = np.asarray(centerline)

    next_line = np.roll(centerline_array, -1, axis=0)
    direction_vectors = next_line - centerline_array

    incoming_vectors = np.roll(direction_vectors, 1, axis=0)

    tangents = direction_vectors + incoming_vectors

    magnitudes = np.linalg.norm(tangents, axis=1)
    unit_tangents = tangents / magnitudes[:, np.newaxis]

    normals = np.stack([-unit_tangents[:, 1], unit_tangents[:, 0]], axis=1)
    left_boundary = centerline_array + normals * (track_width / 2)
    right_boundary = centerline_array - normals * (track_width / 2)
    return left_boundary, right_boundary


def close_loop(points):
    return np.concatenate([points, points[:1]])


def is_on_track(car_x, car_y, centerline, track_width):
    distances = np.linalg.norm(centerline - np.array([car_x, car_y]), axis=1)
    min_distance = np.min(distances)
    if min_distance <= (track_width / 2):
        return True
    else:
        return False
