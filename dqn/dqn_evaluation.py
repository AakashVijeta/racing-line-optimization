from stable_baselines3 import DQN
from racing_env import RacingEnv

model = DQN.load("racing_dqn_model")
env = RacingEnv()

obs, info = env.reset()
print("initial obs:", obs)

for i in range(env.max_steps):
    action, _states = model.predict(
        obs, deterministic=True
    )  # check your action_table — pick whichever index means (steering=0, throttle=0 or +20)
    obs, reward, terminated, truncated, info = env.step(action)
    print(
        f"step {i}: v={obs[0]:.2f}, heading_error={obs[1]:.2f}, reward={reward:.3f}, on_track={not terminated}"
    )
    if terminated:
        break
