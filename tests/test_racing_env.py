import numpy as np
import pytest

from car import Car
from plot_track import make_oval
from racing_env import RacingEnv


def make_env(**kwargs):
    # 200 m straights, 60 m radius, 15 m wide: simple and fully predictable
    return RacingEnv(track_pool={"oval": make_oval()}, track_width=15.0, **kwargs)


# ---------------------------------------------------------------------------
# Car
# ---------------------------------------------------------------------------

def test_car_accelerates_straight_ahead():
    car = Car(x=0.0, y=0.0, theta=0.0)
    for _ in range(20):
        car.step(steering_angle=0.0, throttle=20.0, dt=0.05)
    assert car.v > 0
    assert car.x > 0
    assert car.y == pytest.approx(0.0, abs=1e-9)


def test_car_speed_never_negative_under_braking():
    car = Car(x=0.0, y=0.0, theta=0.0)
    car.v = 1.0
    for _ in range(20):
        car.step(steering_angle=0.0, throttle=-40.0, dt=0.05)
    assert car.v == 0.0


def test_car_traction_circle_limits_lateral_acceleration():
    # Full lock at 80 m/s would demand far more than the grip limit
    car = Car(x=0.0, y=0.0, theta=0.0)
    car.v = 80.0
    grip = car.a_max
    theta_before = car.theta
    car.step(steering_angle=0.4, throttle=0.0, dt=0.05)
    yaw_rate = (car.theta - theta_before) / 0.05
    assert 80.0 * yaw_rate <= grip + 1e-6


# ---------------------------------------------------------------------------
# RacingEnv
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("obs_version,size", [(1, 65), (2, 65), (3, 68)])
def test_reset_returns_observation_in_space(obs_version, size):
    env = make_env(obs_version=obs_version)
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape == (size,)
    assert np.all(np.isfinite(obs))


def test_driving_straight_makes_progress():
    env = make_env()
    env.reset(seed=0)
    total = 0.0
    for _ in range(40):
        obs, reward, terminated, truncated, info = env.step(np.array([0.0, 1.0], dtype=np.float32))
        total += reward
        assert not terminated
    assert env._cumulative_arc > 10.0
    assert total > 0


def test_leaving_the_track_terminates_with_penalty():
    env = make_env()
    env.reset(seed=0)
    for _ in range(500):
        obs, reward, terminated, truncated, info = env.step(np.array([1.0, 1.0], dtype=np.float32))
        if terminated:
            break
    assert info["termination_reason"] == "off_track"
    assert reward == -45.0


def test_boundary_distances_sum_to_track_width_on_a_straight():
    # Midway between centerline vertices on a straight, the true distances to
    # both edges add up to the full width (2 half-widths). The legacy vertex
    # distance overstates it, which is exactly the bug the new mode fixes.
    env = make_env()
    env.reset(seed=0)
    p0, p1 = env.centerline[5], env.centerline[6]
    env.car.x, env.car.y = (p0 + p1) / 2
    env._update_nearest_index()

    left, right = env._get_boundary_distances()
    assert left + right == pytest.approx(2.0, abs=1e-6)

    env.obs_version = 1
    left_legacy, right_legacy = env._get_boundary_distances()
    assert left_legacy + right_legacy > 2.0


def test_lookahead_is_interpolated_between_vertices():
    env = make_env()
    env.reset(seed=0)
    # Halfway along the first segment of the bottom straight
    half = env.segment_lengths[0] / 2
    point, _, _ = env._lookahead_point(half)
    np.testing.assert_allclose(point, (env.centerline[0] + env.centerline[1]) / 2)


def test_lookahead_wraps_through_closing_segment():
    env = make_env()
    env.reset(seed=0)
    closing_mid = (env.arc_length[-1] + env.track_length) / 2
    point, _, _ = env._lookahead_point(closing_mid)
    np.testing.assert_allclose(point, (env.centerline[-1] + env.centerline[0]) / 2)


def test_reset_seed_is_reproducible():
    pool = {"oval": make_oval(), "small": make_oval(straight_length=100.0, radius=40.0)}
    picks = []
    for _ in range(2):
        env = RacingEnv(track_pool=pool)
        env.reset(seed=123)
        picks.append(env.track_length)
    assert picks[0] == picks[1]


# ---------------------------------------------------------------------------
# v15: observation v3, domain randomization, random spawn, stall rule
# ---------------------------------------------------------------------------

LOOKAHEAD_START = 6  # v3: speed, lateral, heading, width, prev steer, prev throttle


def drive(env, obs, target_speed=15.0, max_steps=5000):
    """Simple pure-pursuit driver: steer at the 2nd lookahead point, hold a speed."""
    for _ in range(max_steps):
        angle = obs[LOOKAHEAD_START + 3 * 1 + 1] * np.pi
        steer = np.clip(angle / 0.4, -1.0, 1.0)
        throttle = np.clip((target_speed - env.car.v) / 5.0, -1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step(np.array([steer, throttle], dtype=np.float32))
        if terminated or truncated:
            return info
    raise AssertionError("episode did not end")


def test_v3_obs_contains_width_and_previous_action():
    env = make_env(obs_version=3)
    obs, _ = env.reset(seed=0)
    assert obs[3] == pytest.approx(15.0 / 15.0)
    assert obs[4] == obs[5] == 0.0
    obs, *_ = env.step(np.array([0.25, 0.5], dtype=np.float32))
    assert obs[4] == pytest.approx(0.25)
    assert obs[5] == pytest.approx(0.5)


def test_v3_curvature_is_signed_and_flips_when_mirrored():
    # The oval is driven counter-clockwise: every corner is a left-hander
    env = make_env(obs_version=3)
    env.reset(seed=0)
    assert env.curvature_signed.max() > 0.01
    assert env.curvature_signed.min() > -1e-9

    mirrored = make_env(obs_version=3, randomize=True, mirror_prob=1.0, reverse_prob=0.0,
                        random_spawn=False)
    mirrored.reset(seed=0)
    assert mirrored.track_mirrored
    assert mirrored.curvature_signed.min() < -0.01
    assert mirrored.curvature_signed.max() < 1e-9


def test_randomize_keeps_length_and_bounds_width():
    base = make_env()
    base.reset(seed=0)
    env = make_env(randomize=True, width_scale=(0.5, 2.0), width_limits=(8.0, 16.0))
    for seed in range(20):
        env.reset(seed=seed)
        assert 8.0 <= env.track_width <= 16.0
        assert env.track_length == pytest.approx(base.track_length, rel=1e-9)


def test_random_spawn_starts_on_track_and_lap_counts_full_length():
    env = make_env(randomize=True)
    for seed in range(5):
        obs, _ = env.reset(seed=seed)
        assert env._perp_distance <= env.track_width / 2
        info = drive(env, obs)
        assert info["termination_reason"] == "lap_completed"
        assert info["progress"] >= 1.0


def test_standard_start_lap_completion():
    env = make_env()
    obs, _ = env.reset(seed=0)
    info = drive(env, obs)
    assert info["termination_reason"] == "lap_completed"
    assert info["progress"] == pytest.approx(1.0, abs=0.01)


def test_parking_is_terminated_quickly_and_costs_more_than_a_crash():
    env = make_env()
    env.reset(seed=0)
    for step in range(1, 500):
        obs, reward, terminated, truncated, info = env.step(np.array([0.0, -1.0], dtype=np.float32))
        if terminated:
            break
    assert info["termination_reason"] == "stalled"
    assert step <= env.stall_grace_steps + env.stall_steps + 1
    assert reward == env.stall_penalty
    # Even discounted over the longest delay the agent can buy (braking from
    # ~90 m/s at 40 m/s^2, then the stall window), stalling is worse than crashing now
    braking_steps = int(np.ceil(90.0 / 40.0 / env.dt))
    assert env.stall_penalty * 0.995 ** (braking_steps + env.stall_steps) < -45.0 - 5.0


def test_track_weights_and_explicit_track_option():
    pool = {"oval": make_oval(), "small": make_oval(straight_length=100.0, radius=40.0)}
    env = RacingEnv(track_pool=pool, track_weights={"oval": 0.0, "small": 1.0})
    for seed in range(10):
        env.reset(seed=seed)
        assert env.track_name == "small"
    env.reset(seed=0, options={"track": "oval"})
    assert env.track_name == "oval"
