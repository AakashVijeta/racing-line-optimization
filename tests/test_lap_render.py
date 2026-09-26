import math
import os

import numpy as np
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from lap_render import LapData, LapRenderer, format_laptime, speed_color, surface_to_array

RADIUS, SPEED, DT = 100.0, 30.0, 0.05


@pytest.fixture
def circle_lap(tmp_path):
    """A car driving a 100 m-radius circle anticlockwise at a steady 30 m/s, one full lap."""
    length = 2 * math.pi * RADIUS
    n = int(length / SPEED / DT) + 1
    t = np.arange(n) * DT
    ang = SPEED * t / RADIUS
    theta = (ang + math.pi / 2 + math.pi) % (2 * math.pi) - math.pi  # wrapped, as the sim stores it
    centre_ang = np.linspace(0, 2 * math.pi, 400, endpoint=False)
    centerline = np.column_stack([RADIUS * np.cos(centre_ang), RADIUS * np.sin(centre_ang)])
    path = tmp_path / "circle.npz"
    np.savez(path, x=RADIUS * np.cos(ang), y=RADIUS * np.sin(ang), theta=theta, v=np.full(n, SPEED),
             throttle=np.full(n, 0.3), steering=np.zeros(n), arc=SPEED * t,
             centerline=centerline, left_boundary=centerline * 1.06, right_boundary=centerline * 0.94,
             curvature=np.full(400, 1 / RADIUS), track_width=12.0, dt=DT, track_length=length,
             lap_time=(n - 1) * DT, track_id="circle", track_name="Test Circle", track_split="test",
             model_name="test")
    return LapData(str(path))


def test_derived_channels_match_circular_motion(circle_lap):
    mid = slice(10, -10)  # away from the smoothing edges
    assert np.allclose(circle_lap.g_lat[mid], SPEED ** 2 / RADIUS / 9.81, rtol=0.02)
    assert np.allclose(circle_lap.g_long[mid], 0.0, atol=1e-6)
    assert np.all(np.diff(circle_lap.theta) > 0)  # unwrapped: no jump at +/-pi


def test_state_interpolates_between_samples(circle_lap):
    s = circle_lap.state(1.5 * DT)
    assert s["arc"] == pytest.approx(SPEED * 1.5 * DT)
    # Past the end of the lap, the state holds at the final sample
    assert circle_lap.state(circle_lap.duration + 5)["arc"] == pytest.approx(circle_lap.arc[-1])


def test_renders_frames_at_requested_size(circle_lap):
    renderer = LapRenderer(circle_lap, width=480, height=270, supersample=1)
    for t in (0.0, circle_lap.duration / 2, circle_lap.duration + 1):
        frame = surface_to_array(renderer.render(t))
        assert frame.shape == (270, 480, 3)
        assert frame.std() > 10  # not a blank frame


def test_old_recordings_without_metadata_still_load(tmp_path, circle_lap):
    """Files from the first recorder have no width, curvature or names."""
    path = tmp_path / "old.npz"
    np.savez(path, x=circle_lap.x, y=circle_lap.y, theta=circle_lap.theta, v=circle_lap.v,
             throttle=circle_lap.throttle, steering=circle_lap.steering, arc=circle_lap.arc,
             centerline=circle_lap.centerline, left_boundary=circle_lap.centerline * 1.06,
             right_boundary=circle_lap.centerline * 0.94, dt=DT, track_length=circle_lap.track_length)
    old = LapData(str(path))
    assert old.track_width == pytest.approx(12.0, rel=0.01)
    assert old.lap_time == pytest.approx(old.duration)


def test_speed_colour_scale_and_time_format():
    slow, fast = speed_color(60), speed_color(330)
    assert slow[2] > slow[0] and fast[0] > fast[2]  # blue when slow, red/yellow when fast
    assert format_laptime(87.9) == "1:27.900"
