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
        """
        steering_angle: delta, in radians (commanded by the RL agent)
        throttle: a, acceleration (positive = speed up, negative = brake)
        dt: Delta t, the timestep duration
        """
        # --- 1. LONGITUDINAL PHYSICS (Drag & Speed) ---
        # Drag scales with the square of velocity and opposes the direction of travel
        drag = self.drag_coeff * (self.v ** 2) * np.sign(self.v)
        acceleration = throttle - drag
                   
        self.v += acceleration * dt
        self.v = max(self.v, 0)
        
        # --- 2. LATERAL PHYSICS (Traction Circle / Understeer) ---
        # In a bicycle model: lateral_accel = (v^2 * tan(delta)) / L
        # To maintain grip, we must enforce: lateral_accel <= a_max
        # Therefore, the maximum physical turning angle the tires can sustain is:
        # tan(delta_max) = (a_max * L) / v^2
        # 
        # With downforce: a_max increases with v^2, which partially cancels the v^2
        # in the denominator, allowing tighter turns at higher speeds (as in real F1).
        
        effective_a_max = self.a_max  # Uses the property (includes downforce)
        
        if self.v > 1.0: # Only limit grip when actually moving
            max_tan_delta = (effective_a_max * self.L) / (self.v ** 2)
            max_steer = np.arctan(max_tan_delta)
            
            # If the agent steers harder than the physical grip allows, the tires slip (understeer).
            # The effective turn angle is capped at the maximum physical limit.
            if abs(steering_angle) > max_steer:
                steering_angle = np.sign(steering_angle) * max_steer

        # --- 3. KINEMATIC POSITION UPDATE ---
        self.theta = self.theta + (self.v / self.L) * np.tan(steering_angle) * dt

        self.x = self.x + self.v * np.cos(self.theta) * dt
        self.y = self.y + self.v * np.sin(self.theta) * dt

    def get_state(self):
        return (self.x, self.y, self.theta, self.v)