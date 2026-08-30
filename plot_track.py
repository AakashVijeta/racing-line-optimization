import numpy as np

def make_straight(start_point, end_point, n_points):
    x_start, y_start = start_point
    x_end, y_end = end_point

    # Generate n_points evenly spaced x-values and y-values
    xs = np.linspace(x_start, x_end, n_points, endpoint=True)
    ys = np.linspace(y_start, y_end, n_points, endpoint=True)

    # combine xs and ys into an (n_points, 2) array of points
    points = np.stack([xs, ys], axis=1)
    return points


def make_arc(center, radius, start_angle_deg, end_angle_deg, n_points):
    cx, cy = center

    # Convert start/end angles from degrees to radians
    start_rad = np.radians(start_angle_deg)
    end_rad = np.radians(end_angle_deg)

    # Generate n_points evenly spaced angles
    angles = np.linspace(start_rad, end_rad, n_points, endpoint=True)

    # Apply the circle formula to each angle to get x, y
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
    """
    Checks if the car is within the track boundaries using true point-to-segment distance.
    """
    P = np.array([car_x, car_y])
    A = centerline
    B = np.roll(centerline, -1, axis=0)  # Next point for each segment
    
    # Vector A to B (the segment)
    AB = B - A
    # Vector A to P (car to start of segment)
    AP = P - A
    
    # Project AP onto AB to find the parameterized position 't' of the closest point
    AB_dot_AB = np.sum(AB * AB, axis=1)
    AP_dot_AB = np.sum(AP * AB, axis=1)
    
    # np.maximum prevents division by zero for duplicate points
    t = AP_dot_AB / np.maximum(AB_dot_AB, 1e-12)
    
    # Clamp 't' between 0 and 1 so we don't project past the ends of the segment
    t = np.clip(t, 0.0, 1.0)
    
    # Calculate the actual closest points on all segments
    closest_points = A + t[:, np.newaxis] * AB
    
    # Find the distances from the car to all closest points
    distances = np.linalg.norm(P - closest_points, axis=1)
    
    # If the absolute minimum distance is less than half the track width, we are on track
    min_distance = np.min(distances)
    
    return min_distance <= (track_width / 2.0)