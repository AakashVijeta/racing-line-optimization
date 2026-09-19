from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from racing_env import RacingEnv
from track_preprocessing import build_track_pool, prepare_tracks
from config import TRACK_IDS, TRACK_WIDTHS

def main():
    # 1. Load all tracks (optionally hold out tracks for testing)
    track_ids = [t for t in TRACK_IDS if t != "jp-1962"]  # Hold out jp-1962 for testing
    track_widths = TRACK_WIDTHS

    prepare_tracks(track_ids)
    track_pool = build_track_pool(track_ids=track_ids)

    # 3. Environment Factory
    def make_env():
        env = RacingEnv(track_pool=track_pool, track_widths=track_widths, render_mode=None)
        return Monitor(env)

    # 4. Vectorize Environments
    n_envs = 8
    vec_env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)

    # 5. Evaluation Environment
    eval_env = make_vec_env(make_env, n_envs=1, vec_env_cls=SubprocVecEnv)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/sac_v12/best_model/",
        log_path="./models/sac_v12/eval_logs/",
        eval_freq=max(10000 // n_envs, 1),
        deterministic=True, 
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // n_envs, 1),
        save_path="./models/sac_v12/checkpoints/",
        name_prefix="sac_racing",
    )

    # 6. Upgraded Neural Network Architecture
    # Deeper network for the richer observation space (65-dim instead of 43-dim)
    policy_kwargs = dict(net_arch=[512, 512, 256])

    # 7. Initialize SAC with improved hyperparameters
    model = SAC(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        batch_size=512,
        buffer_size=2_000_000,         # Doubled replay buffer for more diverse experience
        learning_rate=3e-4,
        learning_starts=10000,         # Collect more random exploration data before learning
        gamma=0.995,                   # Higher discount factor: think further ahead
        ent_coef="auto_0.2",          # More exploration via entropy regularization
        verbose=1,
        tensorboard_log="./tb_logs/",
    )

    # 8. Train — 8M steps (~200K per track for 39 tracks)
    print(f"Starting training on {len(track_ids)} tracks...")
    model.learn(
        total_timesteps=8_000_000,
        callback=[eval_callback, checkpoint_callback],
        tb_log_name="multitrack_39_v12",
    )

    model.save("./models/sac_v12/sac_racing_final")
    print("Training finished!")

if __name__ == "__main__":
    main()