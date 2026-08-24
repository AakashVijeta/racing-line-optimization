import gymnasium as gym
import numpy as np
import pygame
from car import Car
from gymnasium import spaces
from plot_track import make_oval, is_on_track, compute_boundaries, close_loop


class RacingEnv(gym.Env):
    def __init__(
        self,
        render_mode=None,
        track_width=15.0,
        n_lookahead=5,
        dt=0.05,
        spacing=10.0,
        max_steps=2000,
    ):
        super().__init__()
        self.dt = dt
        self.render_mode = render_mode
        self.screen = None
        self.clock = pygame.time.Clock()
        self.max_steps = max_steps
        self.n_lookahead = n_lookahead
        self.spacing = spacing
        self.track_width = track_width
        self.max_lookahead = n_lookahead * spacing
        self.max_expected_v = 40
        self.throttle_max = 20
        self.throttle_min = -20
        self.steering_max = 0.4
        self.steering_min = -0.4
        self.smoothness_weight = 0.2
        self.centerline = make_oval()

        gaps = np.linalg.norm(np.diff(self.centerline, axis=0), axis=1)
        keep_mask = np.concatenate([[True], gaps > 1e-6])

        self.centerline = self.centerline[keep_mask]
        left, right = compute_boundaries(self.centerline, self.track_width)
        self.left_boundary = close_loop(left)
        self.right_boundary = close_loop(right)

        # RECOMPUTE after filtering
        self.segment_lengths = np.linalg.norm(np.diff(self.centerline, axis=0), axis=1)

        self.arc_length = np.concatenate([[0], np.cumsum(self.segment_lengths)])
        closing_segment = np.linalg.norm(self.centerline[0] - self.centerline[-1])
        self.track_length = self.arc_length[-1] + closing_segment

        self.action_space = spaces.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3 + self.n_lookahead * 2,),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self._prev_steering = 0.0
        self.car = Car(x=-100, y=-60, theta=0.0)  # starting position and heading
        self.trail = [(self.car.x, self.car.y)]
        self._prev_arc_length = (
            self.precise_arc_length()
        )  # initialize previous arc length

        observation = self._get_obs()
        info = {}

        return observation, info

    def precise_arc_length(self):
        distances = np.linalg.norm(
            self.centerline - np.array([self.car.x, self.car.y]), axis=1
        )
        min_dist_idx = np.argmin(distances)
        before_idx = (min_dist_idx - 1) % len(self.centerline)
        next_idx = (min_dist_idx + 1) % len(self.centerline)

        p1 = self.centerline[min_dist_idx]
        p2 = self.centerline[next_idx]
        p3 = self.centerline[before_idx]

        segment_vec_forward = p2 - p1
        segment_vec_backward = p1 - p3

        t_forward = np.dot(
            segment_vec_forward, np.array([self.car.x, self.car.y]) - p1
        ) / np.dot(segment_vec_forward, segment_vec_forward)
        t_backward = np.dot(
            segment_vec_backward, np.array([self.car.x, self.car.y]) - p3
        ) / np.dot(segment_vec_backward, segment_vec_backward)

        if 1 >= t_forward >= 0:
            t = t_forward
            segment_vec = segment_vec_forward
            base_arc = self.arc_length[min_dist_idx]
        elif 1 >= t_backward >= 0:
            t = t_backward
            segment_vec = segment_vec_backward
            base_arc = self.arc_length[before_idx]
        else:
            t = 0.0
            segment_vec = segment_vec_forward
            base_arc = self.arc_length[min_dist_idx]

        return base_arc + t * np.linalg.norm(segment_vec)

    def _get_obs(self):
        distances = np.linalg.norm(
            self.centerline - np.array([self.car.x, self.car.y]), axis=1
        )
        min_dist_idx = np.argmin(distances)
        next_idx = (min_dist_idx + 1) % len(self.centerline)
        car_arc = self.precise_arc_length()

        p1 = self.centerline[min_dist_idx]
        p2 = self.centerline[next_idx]

        track_vec = p2 - p1
        unit_track_vec = track_vec / np.linalg.norm(track_vec)

        car_vec = np.array([self.car.x, self.car.y]) - p1
        cross = unit_track_vec[0] * car_vec[1] - unit_track_vec[1] * car_vec[0]
        normalized_lateral_error = cross / (self.track_width / 2)

        track_heading = np.arctan2(track_vec[1], track_vec[0])
        heading_error = track_heading - self.car.theta
        heading_error = ((heading_error + np.pi) % (2 * np.pi)) - np.pi

        lookahead_points = []
        for k in range(1, self.n_lookahead + 1):
            target_arc = (car_arc + k * self.spacing) % self.track_length
            target_idx = np.searchsorted(self.arc_length, target_arc) % len(
                self.centerline
            )
            target_point = self.centerline[target_idx]

            dx = target_point[0] - self.car.x
            dy = target_point[1] - self.car.y
            distance = (np.sqrt(dx**2 + dy**2)) / self.max_lookahead

            world_angle = np.arctan2(dy, dx)
            relative_angle = world_angle - self.car.theta
            wrapped_angle = (((relative_angle + np.pi) % (2 * np.pi)) - np.pi) / np.pi

            lookahead_points.append((distance, wrapped_angle))
        flat = [value for pair in lookahead_points for value in pair]

        return np.array(
            [
                (self.car.v / self.max_expected_v),
                normalized_lateral_error,
                (heading_error / np.pi),
            ]
            + flat,
            dtype=np.float32,
        )

    def step(self, action):

        steering, throttle = action

        throttle_physical = self.throttle_min + ((throttle + 1) / 2) * (
            self.throttle_max - self.throttle_min
        )
        steering_physical = self.steering_min + ((steering + 1) / 2) * (
            self.steering_max - self.steering_min
        )

        self.car.step(steering_physical, throttle_physical, self.dt)
        self.trail.append((self.car.x, self.car.y))
        self.steps += 1
        steering_delta = abs(steering - self._prev_steering)
        self._prev_steering = steering

        car_arc = self.precise_arc_length()

        diff = car_arc - self._prev_arc_length
        crossed_finish_forward = False

        if diff > self.track_length / 2:
            # Car nudged backward across arc=0 (diff looks like a big positive
            # jump because precise_arc_length() snapped to the far end of the
            # loop). This is NOT a lap completion.
            arc_diff = diff - self.track_length

        elif diff < -self.track_length / 2:
            # Car moved forward across the finish line (arc wrapped from
            # near track_length back to near 0). This IS a real lap completion.
            arc_diff = diff + self.track_length
            crossed_finish_forward = True

        else:
            # Normal movement along the track.
            arc_diff = diff

        self._prev_arc_length = car_arc
        lap_completed = crossed_finish_forward

        on_track = is_on_track(
            self.car.x, self.car.y, self.centerline, self.track_width
        )

        if not on_track:
            reward = -50.0
            terminated = True
        elif lap_completed:
            reward = arc_diff - 0.1 + 100.0
            terminated = True
        else:
            reward = arc_diff - 0.1
            terminated = False

        reward -= self.smoothness_weight * steering_delta

        truncated = self.steps >= self.max_steps

        obs = self._get_obs()
        info = {}

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((800, 500))
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.screen = None
                return

        self.screen.fill((30, 30, 30))

        scale, offset_x, offset_y = 2.0, 400, 250

        def to_screen(x, y):
            return int(x * scale + offset_x), int(offset_y - y * scale)

        left_points = [to_screen(p[0], p[1]) for p in self.left_boundary]
        right_points = [to_screen(p[0], p[1]) for p in self.right_boundary]
        pygame.draw.lines(self.screen, (255, 255, 255), False, left_points, 2)
        pygame.draw.lines(self.screen, (255, 255, 255), False, right_points, 2)

        on_track = is_on_track(self.car.x, self.car.y, self.centerline, self.track_width)
        color = (0, 200, 0) if on_track else (200, 0, 0)
        if len(self.trail) >= 2:
            trail_points = [to_screen(p[0], p[1]) for p in self.trail]
            pygame.draw.lines(self.screen, (216, 90, 48), False, trail_points, 2)
        pygame.draw.circle(self.screen, color, to_screen(self.car.x, self.car.y), 5)

        pygame.display.flip()
        self.clock.tick(30)


# env = RacingEnv()

# obs, info = env.reset()

# print("observation:", obs)
# print("observation shape:", obs.shape)
# print("space shape:", env.observation_space.shape)
# print("contains:", env.observation_space.contains(obs))

# for i in range(20):
#     action = 8 # check your action_table — pick whichever index means (steering=0, throttle=0 or +20)
#     obs, reward, terminated, truncated, info = env.step(action)
#     print(f"step {i}: v={obs[0]:.2f}, heading_error={obs[1]:.2f}, reward={reward:.3f}, on_track={not terminated}")
#     if terminated:
#         break
