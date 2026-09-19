from stable_baselines3 import DQN
from racing_env import RacingEnv
from stable_baselines3.common.utils import get_linear_fn

env = RacingEnv()
model = DQN("MlpPolicy", env, learning_rate=get_linear_fn(3e-4, 1e-5, 1.0), verbose=1)
model.learn(total_timesteps=5000000)
model.save("racing_dqn_model")
