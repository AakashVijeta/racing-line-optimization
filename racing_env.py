import gymnasium as gym
import numpy as np
import pygame
from car import Car
from gymnasium import spaces
from plot_track import is_on_track, compute_boundaries, close_loop
from track_preprocessing import compute_curvature

class RacingEnv(gym.Env):
    def __init__(
        self,
        track_pool,
        render_mode=None,
        track_width=15.0,        
        track_widths=None,       
        n_lookahead=20,          # 20 lookahead points for long-range vision
        dt=0.05,
        spacing=10.0,
    ):
        super().__init__()
        self.dt = dt
        self.render_mode = render_mode
        self.screen = None
        self.clock = pygame.time.Clock()
        self.n_lookahead = n_lookahead
        self.spacing = spacing
        
        self.default_track_width = track_width
        self.track_widths_dict = track_widths if track_widths is not None else {}
        self.track_width = track_width

        self.max_lookahead = n_lookahead * spacing
        self.max_expected_v = 100.0       # Raised to 100 m/s (360 km/h) for faster driving
        self.throttle_max = 20.0
        self.throttle_min = -40.0         # Fix 2: 4G braking (was -5.0 = 0.5G)
        self.steering_max = 0.4
        self.steering_min = -0.4
        self.smoothness_weight = 0.15
        self.track_pool = track_pool
        
        self._load_track(np.random.choice(list(self.track_pool.keys())))
        
        self.action_space = spaces.Box(
            low=np.array([-1, -1], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32),
            dtype=np.float32,
        )
        
        # Fix 3: Enhanced observation space
        # Base: 3 (speed, lateral_error, heading_error)
        # Lookahead: n_lookahead * 3 (distance, angle, curvature per point)
        # Boundary: 2 (left_dist, right_dist)
        obs_size = 3 + self.n_lookahead * 3 + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32,
        )

    def _load_track(self, track_name):
        self.track_width = self.track_widths_dict.get(track_name, self.default_track_width)

        centerline = self.track_pool[track_name]
        gaps = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
        keep_mask = np.concatenate([[True], gaps > 1e-6])
        self.centerline = centerline[keep_mask]

        if np.linalg.norm(self.centerline[0] - self.centerline[-1]) < 1e-3:
            self.centerline = self.centerline[:-1]

        left, right = compute_boundaries(self.centerline, self.track_width)
        self.left_boundary = close_loop(left)
        self.right_boundary = close_loop(right)
        
        self.segment_lengths = np.linalg.norm(np.diff(self.centerline, axis=0), axis=1)
        self.arc_length = np.concatenate([[0], np.cumsum(self.segment_lengths)])
        closing_segment = np.linalg.norm(self.centerline[0] - self.centerline[-1])
        self.track_length = self.arc_length[-1] + closing_segment
        
        # Pre-compute curvature at each centerline point for observations
        self._precompute_curvature()
        
        min_avg_speed = 12.0  
        max_allowed_time = self.track_length / min_avg_speed
        self.max_steps = int(max_allowed_time / self.dt) + 200

        p0, p1 = self.centerline[0], self.centerline[1]
        self.spawn_x, self.spawn_y = p0
        self.spawn_theta = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
    
    def _precompute_curvature(self):
        """Pre-compute curvature at each centerline point using shared implementation."""
        self.curvature = compute_curvature(self.centerline)
        # Cap extreme curvature values for normalization
        self.max_curvature = 0.5  # Corresponds to ~2m radius, anything tighter is capped

    def _update_nearest_index(self):
        """Compute and cache the nearest centerline index for the current car position.
        
        Called once per step; all methods that need the nearest point reuse this
        cached value instead of doing their own O(N) scan.
        """
        car_pos = np.array([self.car.x, self.car.y])
        distances = np.linalg.norm(self.centerline - car_pos, axis=1)
        self._nearest_idx = np.argmin(distances)
        self._nearest_dist = distances[self._nearest_idx]
        # Cache the full distances array for boundary penalty / on-track checks
        self._distances_to_center = distances

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        track_name = self.np_random.choice(list(self.track_pool.keys()))
        self._load_track(track_name)
        self._cumulative_arc = 0.0
        self.steps = 0
        self._prev_steering = 0.0
        self.car = Car(x=self.spawn_x, y=self.spawn_y, theta=self.spawn_theta)
        self.trail = [(self.car.x, self.car.y)]
        self._update_nearest_index()
        self._prev_arc_length = self.precise_arc_length()

        observation = self._get_obs()
        return observation, {}

    def precise_arc_length(self):
        """Compute precise arc length using the cached nearest index."""
        min_dist_idx = self._nearest_idx
        before_idx = (min_dist_idx - 1) % len(self.centerline)
        next_idx = (min_dist_idx + 1) % len(self.centerline)

        p1 = self.centerline[min_dist_idx]
        p2 = self.centerline[next_idx]
        p3 = self.centerline[before_idx]

        segment_vec_forward = p2 - p1
        segment_vec_backward = p1 - p3

        car_pos = np.array([self.car.x, self.car.y])

        denom_forward = np.dot(segment_vec_forward, segment_vec_forward)
        denom_backward = np.dot(segment_vec_backward, segment_vec_backward)

        t_forward = (
            np.dot(segment_vec_forward, car_pos - p1) / denom_forward
            if denom_forward > 1e-12 else -1.0
        )
        t_backward = (
            np.dot(segment_vec_backward, car_pos - p3) / denom_backward
            if denom_backward > 1e-12 else -1.0
        )

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

    def _get_boundary_distances(self):
        """Compute distance from car to left and right track boundaries.
        
        Uses the cached nearest index to search only a local window of boundary
        points instead of scanning the entire boundary (O(window) vs O(N)).
        """
        car_pos = np.array([self.car.x, self.car.y])
        
        # Search a local window around the cached nearest centerline index
        # Boundaries have the same indexing as centerline (+1 for close_loop)
        window = 30  # ±30 points is more than enough for local search
        n_boundary = len(self.left_boundary)
        idx = self._nearest_idx
        indices = np.arange(idx - window, idx + window + 1) % n_boundary
        
        min_left = np.min(np.linalg.norm(self.left_boundary[indices] - car_pos, axis=1))
        min_right = np.min(np.linalg.norm(self.right_boundary[indices] - car_pos, axis=1))
        
        # Normalize by half track width
        half_width = self.track_width / 2.0
        return min_left / half_width, min_right / half_width

    def _get_obs(self):
        # Use cached nearest index instead of recomputing
        min_dist_idx = self._nearest_idx
        next_idx = (min_dist_idx + 1) % len(self.centerline)
        car_arc = self.precise_arc_length()

        p1 = self.centerline[min_dist_idx]
        p2 = self.centerline[next_idx]

        track_vec = p2 - p1
        track_vec_norm = np.linalg.norm(track_vec)
        if track_vec_norm < 1e-6:
            unit_track_vec = np.array([1.0, 0.0])
        else:
            unit_track_vec = track_vec / track_vec_norm

        car_vec = np.array([self.car.x, self.car.y]) - p1
        cross = unit_track_vec[0] * car_vec[1] - unit_track_vec[1] * car_vec[0]
        normalized_lateral_error = cross / (self.track_width / 2.0)

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

            # Fix 3: Add curvature at each lookahead point
            curv_at_point = self.curvature[target_idx]
            normalized_curv = min(curv_at_point / self.max_curvature, 1.0)
            
            lookahead_points.append((distance, wrapped_angle, normalized_curv))
            
        flat = [value for triple in lookahead_points for value in triple]
        
        # Fix 3: Add boundary distances
        left_dist, right_dist = self._get_boundary_distances()

        return np.array(
            [
                (self.car.v / self.max_expected_v),
                normalized_lateral_error,
                (heading_error / np.pi),
            ]
            + flat
            + [left_dist, right_dist],
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
        before_v = self.car.v
        self.car.step(steering_physical, throttle_physical, self.dt)
        self.trail.append((self.car.x, self.car.y))
        self.steps += 1
        steering_delta = abs(steering - self._prev_steering)
        self._prev_steering = steering

        # Single O(N) scan per step — all subsequent methods reuse cached index
        self._update_nearest_index()
        car_arc = self.precise_arc_length()
        diff = car_arc - self._prev_arc_length
        crossed_finish_forward = False

        if diff > self.track_length / 2:
            arc_diff = diff - self.track_length
        elif diff < -self.track_length / 2:
            arc_diff = diff + self.track_length
            crossed_finish_forward = True
        else:
            arc_diff = diff

        self._cumulative_arc += arc_diff
        if crossed_finish_forward and self._cumulative_arc < 0.9 * self.track_length:
            crossed_finish_forward = False

        self._prev_arc_length = car_arc
        lap_completed = crossed_finish_forward
        truncated = self.steps >= self.max_steps
        
        on_track = is_on_track(self.car.x, self.car.y, self.centerline, self.track_width)
        going_backward = arc_diff < -1.0

        # --- ADVANCED F1 REWARD SHAPING (Fix 4) ---
        
        # 1. Base Progress — scaled by track length for consistency across tracks
        normalized_progress = arc_diff / self.track_length
        progress_reward = normalized_progress * 500.0
        
        # 2. Time Penalty — increased to encourage faster laps
        time_penalty = 0.5
        
        # 3. Tire Slip / Understeer Penalty
        slip_penalty = 0.0
        effective_a_max = self.car.a_max  # Now includes downforce
        if abs(before_v) > 5.0:
            max_tan_delta = (effective_a_max * self.car.L) / (before_v ** 2)
            max_safe_steer = np.arctan(max_tan_delta)
            
            if abs(steering_physical) > max_safe_steer:
                slip_penalty = 2.5 * max(0, (abs(steering_physical) - max_safe_steer) / self.steering_max)
                
        # 4. Smooth Operator Penalty
        normalized_speed = abs(before_v) / self.max_expected_v
        speed_smoothness_penalty = (action[0] ** 2) * normalized_speed * 0.3
        
        # 5. Speed Bonus — reward carrying speed, especially on straights
        speed_bonus = 0.0
        if on_track and before_v > 10.0:
            # Use cached nearest index instead of recomputing
            local_curvature = self.curvature[self._nearest_idx]
            
            # On straights (low curvature), bonus for going fast
            straightness = max(0, 1.0 - local_curvature * 20.0)  # 1.0 on straights, 0 on tight corners
            speed_bonus = straightness * (before_v / self.max_expected_v) * 0.3
        
        # 6. Progressive boundary penalty — warn before crashing
        boundary_penalty = 0.0
        if on_track:
            # Use cached distances from _update_nearest_index()
            min_dist_to_center = self._nearest_dist
            half_width = self.track_width / 2.0
            # If within 20% of the edge, start penalizing
            edge_proximity = min_dist_to_center / half_width
            if edge_proximity > 0.8:
                boundary_penalty = 2.0 * (edge_proximity - 0.8) / 0.2  # Ramps 0→2 near edge
        
        # 7. Final Compilation
        if not on_track or going_backward:
            reward = -100.0  # Increased crash penalty
            terminated = True
        elif lap_completed:
            # Bonus inversely proportional to lap time — faster = bigger reward
            time_bonus = max(0, 200.0 - self.steps * self.dt)
            reward = progress_reward + 100.0 + time_bonus
            terminated = True
        else:
            reward = (progress_reward 
                     - time_penalty 
                     - slip_penalty 
                     - speed_smoothness_penalty 
                     + speed_bonus 
                     - boundary_penalty)
            # Keep jitter penalty
            reward -= self.smoothness_weight * steering_delta
            terminated = False

        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}

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

        all_points = np.concatenate([self.left_boundary, self.right_boundary], axis=0)
        min_x, min_y = all_points.min(axis=0)
        max_x, max_y = all_points.max(axis=0)
        track_w = max(max_x - min_x, 1e-6)
        track_h = max(max_y - min_y, 1e-6)

        margin = 40
        scale = min((800 - 2 * margin) / track_w, (500 - 2 * margin) / track_h)
        offset_x = 400 - scale * (min_x + max_x) / 2
        offset_y = 250 + scale * (min_y + max_y) / 2

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