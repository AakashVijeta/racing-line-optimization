from stable_baselines3 import SAC
from racing_env import RacingEnv
from stable_baselines3.common.callbacks import CheckpointCallback

env = RacingEnv()

model = SAC("MlpPolicy", env, verbose=1)

checkpoint_callback = CheckpointCallback(
    save_freq=7500, save_path="./sac_v7/checkpoints/", name_prefix="sac_racing"
)

model.learn(total_timesteps=150000, callback=checkpoint_callback)

model.save("racing_sac_test")
