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
        self.max_expected_v = 100.0     
        self.throttle_max = 20.0
        self.throttle_min = -40.0      
        self.steering_max = 0.4
        self.steering_min = -0.4
        
        # Reward constants (per-metre basis, track-length invariant)
        self.reward_per_metre = 0.1
        self.v_ref = 30.0  # Reference speed for scaling penalties
        self.stall_window = int(10.0 / dt)  # 10 seconds worth of steps
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
        # Cap extreme curvature values for normalization (real corners are ~0.02-0.1)
        self.max_curvature = 0.1

    def _update_nearest_index(self):
        """Compute and cache the nearest centerline index for the current car position.
        
        Uses a global scan at spawn, and a windowed search during normal stepping
        to prevent jumping across track crossovers (e.g. Suzuka).
        """
        car_pos = np.array([self.car.x, self.car.y])
        if getattr(self, '_nearest_idx', None) is None:
            # Global search at spawn
            distances = np.linalg.norm(self.centerline - car_pos, axis=1)
            self._nearest_idx = np.argmin(distances)
            self._nearest_dist = distances[self._nearest_idx]
        else:
            # Windowed search (+/- 30 points = 300m window)
            window = 30
            n = len(self.centerline)
            idx = self._nearest_idx
            
            indices = np.arange(idx - window, idx + window + 1) % n
            window_points = self.centerline[indices]
            
            distances = np.linalg.norm(window_points - car_pos, axis=1)
            local_min_idx = np.argmin(distances)
            
            self._nearest_idx = indices[local_min_idx]
            self._nearest_dist = distances[local_min_idx]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        track_name = self.np_random.choice(list(self.track_pool.keys()))
        self._load_track(track_name)
        self._cumulative_arc = 0.0
        self.steps = 0
        self._prev_steering = 0.0
        self._stall_arc_history = 0.0  # Rolling arc progress for stall detection
        self._stall_check_step = 0
        self.car = Car(x=self.spawn_x, y=self.spawn_y, theta=self.spawn_theta)
        self.trail = [(self.car.x, self.car.y)]
        self._nearest_idx = None  # Force global search on reset
        self._update_nearest_index()
        self._prev_arc_length = self.precise_arc_length()
        self._cached_arc = self._prev_arc_length

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

        fwd_valid = 0 <= t_forward <= 1
        bwd_valid = 0 <= t_backward <= 1
        
        if fwd_valid and bwd_valid:
            # Both projections valid (near a vertex on a sharp bend).
            # Pick the one with the smaller perpendicular distance.
            proj_fwd = p1 + t_forward * segment_vec_forward
            proj_bwd = p3 + t_backward * segment_vec_backward
            if np.linalg.norm(car_pos - proj_fwd) <= np.linalg.norm(car_pos - proj_bwd):
                t, segment_vec, base_arc, proj_point = t_forward, segment_vec_forward, self.arc_length[min_dist_idx], proj_fwd
            else:
                t, segment_vec, base_arc, proj_point = t_backward, segment_vec_backward, self.arc_length[before_idx], proj_bwd
        elif fwd_valid:
            proj_point = p1 + t_forward * segment_vec_forward
            t, segment_vec, base_arc = t_forward, segment_vec_forward, self.arc_length[min_dist_idx]
        elif bwd_valid:
            proj_point = p3 + t_backward * segment_vec_backward
            t, segment_vec, base_arc = t_backward, segment_vec_backward, self.arc_length[before_idx]
        else:
            t, segment_vec, base_arc = 0.0, segment_vec_forward, self.arc_length[min_dist_idx]
            proj_point = p1
            
        self._perp_distance = np.linalg.norm(car_pos - proj_point)

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
        car_arc = self._cached_arc

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
            curv_at_point = abs(self.curvature[target_idx])
            normalized_curv = np.clip(curv_at_point / self.max_curvature, 0.0, 1.0)
            
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

        if throttle >= 0:
            throttle_physical = throttle * self.throttle_max
        else:
            # throttle is negative, [-1, 0] maps to [-40, 0]
            throttle_physical = -throttle * self.throttle_min
        steering_physical = self.steering_min + ((steering + 1) / 2) * (
            self.steering_max - self.steering_min
        )
        before_v = self.car.v
        self.car.step(steering_physical, throttle_physical, self.dt)
        self.trail.append((self.car.x, self.car.y))
        self.steps += 1
        steering_delta = abs(steering - self._prev_steering)
        self._prev_steering = steering

        self._update_nearest_index()
        car_arc = self.precise_arc_length()
        self._cached_arc = car_arc
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
        
        on_track = self._perp_distance <= (self.track_width / 2.0)
        going_backward = arc_diff < -1.0

        # --- Stall detection ---
        self._stall_arc_history += arc_diff
        stalled = False
        if self.steps - self._stall_check_step >= self.stall_window:
            if self._stall_arc_history < 5.0:  # < 5m in 10 seconds = stalled
                stalled = True
            self._stall_arc_history = 0.0
            self._stall_check_step = self.steps

        # --- REWARD (per-metre basis, track-length invariant) ---
        
        # 1. Progress: constant reward per metre, independent of L
        progress_reward = arc_diff * self.reward_per_metre
        
        # 2. Time penalty: in same units as progress
        #    At v_ref, car earns v_ref * dt * reward_per_metre per step.
        #    Time penalty = α * that. Break-even speed = α * v_ref.
        ref_reward_per_step = self.v_ref * self.dt * self.reward_per_metre
        time_penalty = 0.2 * ref_reward_per_step  # Break-even at 6 m/s
        
        # 3. Jitter penalty (same units)
        jitter_penalty = (0.3 * ref_reward_per_step) * steering_delta

        # 4. Crash penalty: ~1.5 × H × ref_reward_per_step
        #    With γ=0.995, H=200, crash ≈ -45
        crash_penalty = -45.0

        # --- Termination & Logging ---
        info = {}
        
        if not on_track:
            reward = crash_penalty
            terminated = True
            info["termination_reason"] = "off_track"
        elif going_backward:
            reward = crash_penalty
            terminated = True
            info["termination_reason"] = "going_backward"
        elif stalled:
            reward = crash_penalty
            terminated = True
            info["termination_reason"] = "stalled"
        elif lap_completed:
            reward = progress_reward + 10.0  # Small bonus, rest comes from progress
            terminated = True
            info["termination_reason"] = "lap_completed"
        else:
            reward = (progress_reward 
                     - time_penalty 
                     - jitter_penalty)
            terminated = False
            info["termination_reason"] = "ongoing"
        
        # Fix truncation logging: override if truncated and not already terminated
        if truncated and not terminated:
            info["termination_reason"] = "timeout"

        obs = self._get_obs()
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

        on_track_render = self._perp_distance <= (self.track_width / 2.0)
        color = (0, 200, 0) if on_track_render else (200, 0, 0)
        if len(self.trail) >= 2:
            trail_points = [to_screen(p[0], p[1]) for p in self.trail]
            pygame.draw.lines(self.screen, (216, 90, 48), False, trail_points, 2)
        pygame.draw.circle(self.screen, color, to_screen(self.car.x, self.car.y), 5)

        pygame.display.flip()
        self.clock.tick(30)