import numpy as np


class Car:
    def __init__(self, x, y, theta, wheelbase=2.5, max_lateral_accel=8.0):
        """
        x, y: starting position
        theta: starting heading angle (radians)
        wheelbase: L, the tuning constant from our turning-radius formula
        max_lateral_accel: a_max, the grip limit constant
        """
        self.x = x
        self.y = y
        self.theta = theta
        self.v = 0.0  # starting speed is zero

        self.L = wheelbase
        self.a_max = max_lateral_accel

    def step(self, steering_angle, throttle, dt):
        """
        steering_angle: delta, in radians
        throttle: a, acceleration (positive = speed up, negative = brake)
        dt: Delta t, the timestep duration
        """
        # 1. Update speed from throttle FIRST so the car can start moving
        self.v = self.v + throttle * dt

        # Add a tiny bit of natural friction so the car slowly stops if you let go
        if throttle == 0.0:
            self.v *= 0.98

        # 2. Calculate turning radius
        if abs(steering_angle) >= 1e-6:
            r = self.L / np.tan(steering_angle)
            # Limit our speed based on how sharp the turn is (grip limit)
            v_max = np.sqrt(self.a_max * abs(r))
            if self.v > v_max:
                excess = self.v - v_max
                grip_loss_rate = (
                    5.0  # TODO: tune this - how many m/s per second bleeds off
                )
                self.v -= min(
                    grip_loss_rate * dt, excess
                )  # don't overshoot below v_max
            elif self.v < -v_max:  # Also cap it if reversing
                self.v = -v_max

        # 3. Update the car's facing angle (theta)
        self.theta = self.theta + (self.v / self.L) * np.tan(steering_angle) * dt

        # 4. Move the car's X and Y positions using the updated speed
        self.x = self.x + self.v * np.cos(self.theta) * dt
        self.y = self.y + self.v * np.sin(self.theta) * dt

    def get_state(self):
        return (self.x, self.y, self.theta, self.v)
