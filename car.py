import numpy as np

class Car:
    def __init__(self, x, y, theta, wheelbase=3.5, max_lateral_accel=44.1):
        """
        x, y: starting position
        theta: starting heading angle (radians)
        wheelbase: L, distance between front and rear axles (F1 cars are around 3.5m)
        max_lateral_accel: a_max, base mechanical grip limit (44.1 m/s^2 is ~4.5G for F1 cars)
        """
        self.x = x
        self.y = y
        self.theta = theta
        self.v = 0.0 

        self.L = wheelbase
        self.a_max_base = max_lateral_accel
        
        # F1 Aerodynamic drag coefficient
        # With throttle=20, a drag_coeff of 0.0022 caps top speed around 340 km/h (95 m/s)
        self.drag_coeff = 0.0022  
        
        # Downforce coefficient: models how aerodynamic downforce increases tire grip at speed.
        # Effective a_max = a_max_base + downforce_coeff * v^2
        # At v=70 m/s (252 km/h): extra grip = 0.005 * 4900 = 24.5 m/s^2
        # Total grip = 44.1 + 24.5 = 68.6 m/s^2 (~7G), realistic for modern F1
        self.downforce_coeff = 0.005

    @property
    def a_max(self):
        """Speed-dependent maximum lateral acceleration including downforce."""
        return self.a_max_base + self.downforce_coeff * (self.v ** 2)

    def step(self, steering_angle, throttle, dt):
        before_v = self.v 

        # --- 1. DEMANDED FORCES ---
        a_long_demand = throttle
        a_lat_demand = (before_v**2 * np.tan(steering_angle)) / self.L

        effective_a_max = self.a_max  

        # --- 2. TRACTION CIRCLE ---
        magnitude = np.sqrt(a_long_demand**2 + a_lat_demand**2)
        if magnitude > effective_a_max:
            scale = effective_a_max / magnitude
            a_long_achieved = a_long_demand * scale
            a_lat_achieved  = a_lat_demand * scale
        else:
            a_long_achieved = a_long_demand
            a_lat_achieved  = a_lat_demand

        # --- 3. DRAG ---
        drag = self.drag_coeff * before_v**2 * np.sign(before_v)
        a_long_final = a_long_achieved - drag

        # --- 4. LONGITUDINAL UPDATE ---
        self.v += a_long_final * dt
        self.v = max(self.v, 0)

        # --- 5. RECOVER ACHIEVED STEERING ANGLE ---
        if before_v > 1.0:
            achieved_steering_angle = np.arctan(a_lat_achieved * self.L / before_v**2)
        else:
            achieved_steering_angle = steering_angle

        # --- 6. KINEMATIC UPDATE ---
        self.theta += (before_v / self.L) * np.tan(achieved_steering_angle) * dt

        # --- 7. POSITION UPDATE ---
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt

    def get_state(self):
        return (self.x, self.y, self.theta, self.v)