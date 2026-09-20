"""
Multi-track SAC Training script for RacingEnv.
Includes Custom Eval Callback for per-track metrics and a warm-start from the oval model.
"""
import os
import numpy as np
import torch
from collections import Counter
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from racing_env import RacingEnv
from track_preprocessing import build_track_pool, prepare_tracks
from config import TRACK_IDS, TRACK_WIDTHS

class MultiTrackEvalCallback(BaseCallback):
    def __init__(self, eval_tracks_dict, track_widths, eval_freq=25000, best_model_path=None, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.last_eval = 0
        self.best_model_path = best_model_path
        self.best_mean_reward = -np.inf
        
        # Build eval environments for each track
        self.eval_envs = {}
        for track_name, track_data in eval_tracks_dict.items():
            pool = {track_name: track_data}
            def make_env():
                env = RacingEnv(track_pool=pool, track_widths=track_widths, render_mode=None)
                return Monitor(env)
            # Use single env for deterministic evaluation
            self.eval_envs[track_name] = make_vec_env(make_env, n_envs=1, vec_env_cls=SubprocVecEnv)
            
        # Training-time termination reason tracker
        self.term_reasons = Counter()

    def _on_step(self) -> bool:
        # 1. Track training termination reasons
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for i, done in enumerate(dones):
            if done:
                reason = infos[i].get("termination_reason", "unknown")
                self.term_reasons[reason] += 1
                
        # 2. Entropy guard
        if hasattr(self.model, "log_ent_coef"):
            ent_coef = torch.exp(self.model.log_ent_coef.detach()).item()
            if ent_coef > 0.35:
                print(f"Stopping training early: entropy coefficient {ent_coef:.3f} exceeded 0.35")
                return False

        # 3. Evaluation
        if self.num_timesteps - self.last_eval >= self.eval_freq:
            self.last_eval = self.num_timesteps
            
            # Log training term reasons
            total_dones = sum(self.term_reasons.values())
            if total_dones > 0:
                for reason, count in self.term_reasons.items():
                    self.logger.record(f"train_term_reasons/{reason}_pct", count / total_dones)
            self.term_reasons.clear()
            
            # Run evaluations
            all_rewards = []
            for track_name, env in self.eval_envs.items():
                obs = env.reset()
                done, state = False, None
                ep_reward = 0.0
                ep_steps = 0
                
                while not done:
                    action, state = self.model.predict(obs, state=state, deterministic=True)
                    obs, reward, done, info = env.step(action)
                    ep_reward += reward[0]
                    ep_steps += 1
                
                info = info[0] # SubprocVecEnv returns lists
                reason = info.get("termination_reason", "unknown")
                completed = 1.0 if reason == "lap_completed" else 0.0
                
                self.logger.record(f"eval_{track_name}/reward", ep_reward)
                self.logger.record(f"eval_{track_name}/lap_completed", completed)
                self.logger.record(f"eval_{track_name}/steps", ep_steps)
                all_rewards.append(ep_reward)
                
            mean_reward = np.mean(all_rewards)
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.best_model_path:
                    os.makedirs(self.best_model_path, exist_ok=True)
                    self.model.save(os.path.join(self.best_model_path, "best_model.zip"))
                    if self.verbose > 0:
                        print(f"New best model saved! Mean reward: {mean_reward:.1f}")

        return True

def main():
    # 1. Track pools
    holdout_tracks = ["jp-1962", "mc-1929", "be-1925"]  # Suzuka (Final), Monaco (Val), Spa (Val)
    train_track_ids = [t for t in TRACK_IDS if t not in holdout_tracks]
    track_widths = TRACK_WIDTHS

    prepare_tracks(train_track_ids + holdout_tracks)
    train_pool = build_track_pool(train_track_ids)
    
    # Eval specific pools
    eval_tracks_dict = {
        "mc-1929": build_track_pool(["mc-1929"])["mc-1929"],
        "be-1925": build_track_pool(["be-1925"])["be-1925"],
    }

    def make_env():
        env = RacingEnv(track_pool=train_pool, track_widths=track_widths, render_mode=None)
        return Monitor(env)

    n_envs = 8
    vec_env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)

    eval_callback = MultiTrackEvalCallback(
        eval_tracks_dict=eval_tracks_dict,
        track_widths=track_widths,
        eval_freq=25000,
        best_model_path="./models/sac_v13/best_model/"
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // n_envs, 1),
        save_path="./models/sac_v13/checkpoints/",
        name_prefix="sac_racing",
    )

    policy_kwargs = dict(net_arch=[512, 512, 256])

    model = SAC(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        batch_size=512,
        buffer_size=2_000_000,
        learning_rate=3e-4,
        learning_starts=2000,  # 2k uniform random steps to prime the buffer
        gamma=0.995,
        ent_coef="auto_0.05",  # Start entropy lower to preserve warm-started weights
        verbose=1,
        tensorboard_log="./tb_logs/",
    )

    print("Loading weights from oval pre-training...")
    oval_model = SAC.load("./models/sac_oval_test/sac_oval_final")
    model.policy.load_state_dict(oval_model.policy.state_dict())
    del oval_model 

    print(f"Starting V13 training on {len(train_track_ids)} tracks...")
    model.learn(
        total_timesteps=8_000_000,
        callback=[eval_callback, checkpoint_callback],
        tb_log_name="multitrack_v13",
    )

    model.save("./models/sac_v13/sac_racing_final")
    print("Training finished!")

if __name__ == "__main__":
    main()