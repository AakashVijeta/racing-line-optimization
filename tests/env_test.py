from racing_env import RacingEnv
import numpy as np

env = RacingEnv()

obs, info = env.reset()

for action in [
    np.array([0.0, 0.0], dtype=np.float32),
    np.array([0.0, 1.0], dtype=np.float32),
    np.array([0.0, 1.0], dtype=np.float32),
    np.array([0.0, 1.0], dtype=np.float32),
]:
    obs, reward, terminated, truncated, info = env.step(action)

    print(
        f"action={action}, "
        f"x={env.car.x:.2f}, "
        f"y={env.car.y:.2f}, "
        f"v={env.car.v:.2f}, "
        f"reward={reward:.3f}, "
        f"terminated={terminated}"
    )
