import os
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from racing_env import RacingEnv
from track_preprocessing import build_track_pool, prepare_tracks

def main():
    # 1. Load all 40 tracks
    track_ids = [
        "mc-1929", "az-2016", "nl-1948", "it-1953", "hu-1986", "sg-2008",
        "sa-2021", "pt-1972", "fr-1960", "au-1953", "ca-1978", "at-1969",
        "jp-1962", "mx-1962", "br-1940", "br-1977", "ar-1952", "za-1961",
        "us-1956", "us-2022", "us-2023", "es-2026", "de-1927", "de-1932",
        "it-1914", "be-1925", "it-1922", "es-1991", "ae-2009", "pt-2008",
        "tr-2005", "ru-2014", "qa-2004", "us-1909", "gb-1948", "bh-2002",
        "cn-2004", "us-2012", "my-1999", "fr-1969"
    ]
    
    prepare_tracks(track_ids)
    track_pool = build_track_pool(track_ids=track_ids)

    # 2. Track Widths Dictionary
    track_widths = {
        "mc-1929": 8.5, "az-2016": 10.0, "nl-1948": 10.0,
        "it-1953": 11.0, "hu-1986": 11.0, "sg-2008": 11.0, "sa-2021": 11.0, "pt-1972": 11.0,
        "fr-1960": 11.5, "au-1953": 12.0, "ca-1978": 12.0, "at-1969": 12.0, "jp-1962": 12.0,
        "mx-1962": 12.0, "br-1940": 12.0, "br-1977": 12.0, "ar-1952": 12.0, "za-1961": 12.0,
        "us-1956": 12.0, "us-2022": 12.0, "us-2023": 12.0, "es-2026": 12.0,
        "de-1927": 13.0, "de-1932": 13.0, "it-1914": 13.0,
        "be-1925": 14.0, "it-1922": 14.0, "es-1991": 14.0, "ae-2009": 14.0, "pt-2008": 14.0,
        "tr-2005": 14.0, "ru-2014": 14.0, "qa-2004": 14.0, "us-1909": 14.0,
        "gb-1948": 15.0, "bh-2002": 15.0, "cn-2004": 15.0, "us-2012": 15.0, "my-1999": 15.0,
        "fr-1969": 15.0
    }

    # 3. Environment Factory
    def make_env():
        env = RacingEnv(track_pool=track_pool, track_widths=track_widths, render_mode=None)
        return Monitor(env)

    # 4. Vectorize Environments (Set n_envs to the number of CPU cores you want to use)
    n_envs = 8
    vec_env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)

    # 5. Evaluation Environment
    eval_env = make_vec_env(make_env, n_envs=1, vec_env_cls=SubprocVecEnv)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/sac_v10_40tracks/best_model/",
        log_path="./models/sac_v10_40tracks/eval_logs/",
        eval_freq=max(10000 // n_envs, 1),
        deterministic=True, 
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // n_envs, 1),
        save_path="./models/sac_v10_40tracks/checkpoints/",
        name_prefix="sac_racing",
    )

    # 6. Upgraded Neural Network Architecture
    policy_kwargs = dict(net_arch=[512, 512])

    # 7. Initialize SAC 
    model = SAC(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        batch_size=512,            # Larger batch size for the larger network
        buffer_size=1_000_000,     # Max out replay buffer to hold data from all 40 tracks
        learning_rate=3e-4,
        verbose=1,
        tensorboard_log="./tb_logs/",
    )

    # 8. Train (2.5 to 3 million steps recommended for 40 tracks)
    print(f"Starting training on {len(track_ids)} tracks...")
    model.learn(
        total_timesteps=2_500_000,
        callback=[eval_callback, checkpoint_callback],
        tb_log_name="multitrack_40_v1",
    )

    model.save("./models/sac_v10_40tracks/sac_racing_final")
    print("Training finished!")

if __name__ == "__main__":
    main()