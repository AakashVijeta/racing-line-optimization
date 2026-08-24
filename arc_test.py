import numpy as np
from racing_env import RacingEnv

env = RacingEnv()

obs, info = env.reset()

print("Initial:")
print(f"  position = ({env.car.x:.2f}, {env.car.y:.2f})")
print(f"  arc      = {env.precise_arc_length():.4f}")
print()

# Drive straight with full throttle
action = np.array([0.0, 1.0], dtype=np.float32)

for i in range(20):
    previous_arc = env.precise_arc_length()

    obs, reward, terminated, truncated, info = env.step(action)

    current_arc = env.precise_arc_length()
    diff = current_arc - previous_arc

    print(
        f"step {i+1:2d} | "
        f"v={env.car.v:6.2f} | "
        f"pos=({env.car.x:7.2f}, {env.car.y:7.2f}) | "
        f"arc={current_arc:8.3f} | "
        f"diff={diff:7.3f}"
    )

    if terminated or truncated:
        print("Episode ended.")
        break
