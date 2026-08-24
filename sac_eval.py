from stable_baselines3 import SAC
from racing_env import RacingEnv

env = RacingEnv(render_mode="human")
model = SAC.load("./sac_v7/checkpoints/sac_racing_127500_steps.zip", env=env)

obs, info = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    done = terminated or truncated

env.close()