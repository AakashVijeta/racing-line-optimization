"""
Multi-track SAC training for RacingEnv, built for generalization to unseen tracks.

Training mix: real circuits (config.TRAIN_TRACKS) plus procedurally generated
tracks, with per-episode domain randomization (track width, mirroring,
reversing, random start point and speed). Hard real tracks are sampled more
often, based on their recent failure rate.

Checkpoint selection uses a validation set of real (config.VAL_TRACKS) and
procedural tracks, scored on laps completed and lap time rather than reward.

    python sac_train.py --run sac_v4
    python sac_train.py --run sac_v4 --timesteps 10000000 --proc-frac 0.6
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from config import LATEST_OBS_VERSION, TEST_TRACKS, TRACK_WIDTHS, TRAIN_TRACKS, VAL_TRACKS
from racing_env import RacingEnv, drive_lap
from track_generator import procedural_split
from track_preprocessing import build_track_pool, prepare_tracks


def lap_score(completed, lap_time, track_length, progress):
    """Validation score for one lap: finishing always beats not finishing, faster beats slower.

    Finished: 1 + average speed / 100 m/s (so roughly 1.4-1.8). Not finished:
    the fraction of the lap covered (0-1).
    """
    if completed:
        return 1.0 + (track_length / lap_time) / 100.0
    return float(np.clip(progress, 0.0, 1.0))


def evaluate_policy_on(model, track_pool, widths, obs_version):
    """One deterministic standing-start lap per track: {track: (completed, lap_time, length, progress)}."""
    results = {}
    for tid, centerline in track_pool.items():
        env = RacingEnv(track_pool={tid: centerline}, track_widths=widths, obs_version=obs_version)
        steps, info = drive_lap(model, env)
        completed = info["termination_reason"] == "lap_completed"
        results[tid] = (completed, steps * env.dt, env.track_length, info["progress"])
    return results


class GeneralizationCallback(BaseCallback):
    """Validation-based model selection, adaptive track sampling and training statistics."""

    def __init__(self, val_pool, widths, obs_version, run_dir, real_train_ids, proc_train_ids,
                 proc_frac, eval_freq=100_000, adapt_freq=50_000, adaptive=True,
                 max_ent_coef=None, verbose=1):
        super().__init__(verbose)
        self.val_pool = val_pool
        self.widths = widths
        self.obs_version = obs_version
        self.run_dir = run_dir
        self.real_train_ids = list(real_train_ids)
        self.proc_train_ids = list(proc_train_ids)
        self.proc_frac = proc_frac
        self.eval_freq = eval_freq
        self.adapt_freq = adapt_freq
        self.adaptive = adaptive
        self.max_ent_coef = max_ent_coef

        self.best_score = -np.inf
        self.last_eval = 0
        self.last_adapt = 0
        self.term_reasons = Counter()
        self.group_stats = defaultdict(lambda: [0, 0, 0.0])  # group -> [episodes, laps, progress sum]
        # Failure-rate estimate per real training track (exponential moving average)
        self.fail_rate = dict.fromkeys(self.real_train_ids, 0.5)
        self.history_path = os.path.join(run_dir, "validation_history.csv")

    def track_weights(self):
        """Real tracks: 0.5 + failure rate, scaled to (1 - proc_frac). Procedural: uniform, proc_frac."""
        weights = {}
        if self.real_train_ids:
            raw = {t: 0.5 + (self.fail_rate[t] if self.adaptive else 0.5) for t in self.real_train_ids}
            total = sum(raw.values())
            real_share = 1.0 - self.proc_frac if self.proc_train_ids else 1.0
            weights.update({t: real_share * w / total for t, w in raw.items()})
        if self.proc_train_ids:
            proc_share = self.proc_frac if self.real_train_ids else 1.0
            weights.update({t: proc_share / len(self.proc_train_ids) for t in self.proc_train_ids})
        return weights

    def _on_training_start(self) -> None:
        self.training_env.env_method("set_track_weights", self.track_weights())

    def _on_step(self) -> bool:
        for done, info in zip(self.locals.get("dones", []), self.locals.get("infos", [])):
            if not done:
                continue
            reason = info.get("termination_reason", "unknown")
            track = info.get("track", "")
            completed = reason == "lap_completed"
            self.term_reasons[reason] += 1
            group = "proc" if track.startswith("proc-") else "real"
            stats = self.group_stats[group]
            stats[0] += 1
            stats[1] += completed
            stats[2] += min(info.get("progress", 0.0), 1.0)
            if track in self.fail_rate:
                self.fail_rate[track] = 0.9 * self.fail_rate[track] + 0.1 * (not completed)

        self._clamp_entropy()

        # Entropy guard: a runaway entropy coefficient means training has destabilised
        if hasattr(self.model, "log_ent_coef"):
            ent_coef = torch.exp(self.model.log_ent_coef.detach()).item()
            if ent_coef > 0.35:
                print(f"Stopping training early: entropy coefficient {ent_coef:.3f} exceeded 0.35")
                return False

        if self.adaptive and self.num_timesteps - self.last_adapt >= self.adapt_freq:
            self.last_adapt = self.num_timesteps
            self.training_env.env_method("set_track_weights", self.track_weights())
            rates = np.array(list(self.fail_rate.values()))
            self.logger.record("sampling/real_fail_rate_mean", float(rates.mean()))
            self.logger.record("sampling/real_fail_rate_max", float(rates.max()))

        if self.num_timesteps - self.last_eval >= self.eval_freq:
            self.last_eval = self.num_timesteps
            self._log_training_stats()
            self._validate()
        return True

    def _clamp_entropy(self):
        """Cap the auto-tuned entropy coefficient.

        SAC adds -alpha * log(pi) to every step's reward. If alpha keeps rising
        (the policy wants to be more precise than the entropy target allows),
        that cost approaches the progress reward (~0.1-0.4 per step). Driving
        then earns nothing net, ending episodes early costs nothing, and the
        policy collapses into stalling and crashing. This happened in the
        first v2 attempt: the cost grew from 0.04 to 0.13 per step by 2M steps.
        """
        if self.max_ent_coef is None or not hasattr(self.model, "log_ent_coef"):
            return
        with torch.no_grad():
            self.model.log_ent_coef.clamp_(max=float(np.log(self.max_ent_coef)))

    def _log_training_stats(self):
        total = sum(self.term_reasons.values())
        for reason, count in self.term_reasons.items():
            self.logger.record(f"train_term_reasons/{reason}_pct", count / max(total, 1))
        for group, (episodes, laps, progress) in self.group_stats.items():
            if episodes:
                # Progress fraction is comparable across tracks; ep_rew_mean is not
                self.logger.record(f"train_{group}/lap_completed_pct", laps / episodes)
                self.logger.record(f"train_{group}/progress_mean", progress / episodes)
        self.term_reasons.clear()
        self.group_stats.clear()

    def _validate(self):
        results = evaluate_policy_on(self.model, self.val_pool, self.widths, self.obs_version)
        scores = {t: lap_score(*r) for t, r in results.items()}
        score = float(np.mean(list(scores.values())))
        groups = {"real": [t for t in results if not t.startswith("proc-")],
                  "proc": [t for t in results if t.startswith("proc-")]}

        self.logger.record("val/score", score)
        for group, ids in groups.items():
            if ids:
                self.logger.record(f"val_{group}/lap_completed_pct", np.mean([results[t][0] for t in ids]))
                self.logger.record(f"val_{group}/score", np.mean([scores[t] for t in ids]))
        for t in groups["real"]:
            completed, lap_time, _, progress = results[t]
            self.logger.record(f"val_{t}/lap_time", lap_time if completed else np.nan)
            self.logger.record(f"val_{t}/progress", progress)

        new_file = not os.path.exists(self.history_path)
        with open(self.history_path, "a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["steps", "score", "completed", "n_tracks"])
            writer.writerow([self.num_timesteps, round(score, 4),
                             sum(r[0] for r in results.values()), len(results)])

        if score > self.best_score:
            self.best_score = score
            path = os.path.join(self.run_dir, "best_model")
            os.makedirs(path, exist_ok=True)
            self.model.save(os.path.join(path, "best_model.zip"))
            if self.verbose:
                done = sum(r[0] for r in results.values())
                print(f"New best model: val score {score:.3f} ({done}/{len(results)} laps)")


def build_parser():
    parser = argparse.ArgumentParser(description="Train a SAC racing agent that generalizes to unseen tracks")
    parser.add_argument("--run", required=True,
                        help="run name; outputs go to models/<run>/ (must not already exist)")
    parser.add_argument("--timesteps", type=int, default=8_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--proc-train", type=int, default=None,
                        help="number of procedural training tracks (default: config.PROC_SIZES['train'])")
    parser.add_argument("--proc-frac", type=float, default=0.5,
                        help="share of episodes on procedural tracks (0 disables them)")
    parser.add_argument("--no-randomize", action="store_true",
                        help="disable width/mirror/reverse/random-spawn randomization")
    parser.add_argument("--no-adaptive", action="store_true",
                        help="sample real tracks uniformly instead of by failure rate")
    parser.add_argument("--lr", type=float, default=3e-4, help="initial learning rate")
    parser.add_argument("--lr-final", type=float, default=3e-5, help="learning rate at the end (linear decay)")
    parser.add_argument("--gradient-steps", type=int, default=2,
                        help="gradient steps per vectorized env step (updates per sample = this / n-envs)")
    parser.add_argument("--eval-freq", type=int, default=100_000)
    parser.add_argument("--target-entropy", type=float, default=-4.0,
                        help="SAC entropy target (SB3 default is -2 for 2 actions); lower lets the policy be more precise")
    parser.add_argument("--max-ent-coef", type=float, default=0.02,
                        help="upper bound on the auto-tuned entropy coefficient (0 disables the cap)")
    parser.add_argument("--warm-start", default=None,
                        help="saved SAC model whose policy weights initialise training (must use the same observation layout)")
    return parser


def git_sha():
    """Current commit, so a run's config records which code it was trained with."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    args = build_parser().parse_args()
    run_dir = os.path.join("./models", args.run)
    if os.path.exists(run_dir):
        # Refuse rather than silently overwrite an earlier run's best model
        sys.exit(f"{run_dir} already exists. Pick a new --run name.")
    os.makedirs(run_dir)
    config = dict(vars(args), git_sha=git_sha())
    with open(os.path.join(run_dir, "train_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    obs_version = LATEST_OBS_VERSION

    # 1. Track pools. Real: Suzuka is the untouched test track, Monaco and Spa are validation.
    prepare_tracks(TRAIN_TRACKS + VAL_TRACKS + TEST_TRACKS)
    train_pool = build_track_pool(TRAIN_TRACKS)
    widths = dict(TRACK_WIDTHS)

    proc_train_ids = []
    if args.proc_frac > 0:
        from config import PROC_SEEDS
        from track_generator import generate_pool
        n_proc = args.proc_train
        if n_proc is None:
            proc_pool, proc_widths = procedural_split("train")
        else:
            proc_pool, proc_widths = generate_pool(n_proc, seed=PROC_SEEDS["train"])
        train_pool.update(proc_pool)
        widths.update(proc_widths)
        proc_train_ids = list(proc_pool)

    val_pool = build_track_pool(VAL_TRACKS)
    proc_val, proc_val_widths = procedural_split("val")
    val_pool.update(proc_val)
    widths.update(proc_val_widths)

    randomize = not args.no_randomize

    def make_env():
        env = RacingEnv(track_pool=train_pool, track_widths=widths, obs_version=obs_version,
                        randomize=randomize)
        return Monitor(env)

    n_envs = args.n_envs
    vec_env = make_vec_env(make_env, n_envs=n_envs, seed=args.seed, vec_env_cls=SubprocVecEnv)

    callback = GeneralizationCallback(
        val_pool=val_pool,
        widths=widths,
        obs_version=obs_version,
        run_dir=run_dir,
        real_train_ids=TRAIN_TRACKS,
        proc_train_ids=proc_train_ids,
        proc_frac=args.proc_frac,
        eval_freq=args.eval_freq,
        adaptive=not args.no_adaptive,
        max_ent_coef=args.max_ent_coef or None,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // n_envs, 1),
        save_path=os.path.join(run_dir, "checkpoints"),
        name_prefix="sac_racing",
    )

    lr0, lr1 = args.lr, args.lr_final

    def lr_schedule(progress_remaining):
        # progress_remaining goes 1 -> 0 over training
        return lr1 + (lr0 - lr1) * progress_remaining

    model = SAC(
        "MlpPolicy",
        vec_env,
        policy_kwargs={"net_arch": [512, 512, 256]},
        batch_size=512,
        buffer_size=2_000_000,
        learning_rate=lr_schedule,
        learning_starts=10_000,
        gradient_steps=args.gradient_steps,
        gamma=0.995,
        ent_coef="auto_0.02",
        target_entropy=args.target_entropy,
        verbose=1,
        tensorboard_log="./tb_logs/",
        seed=args.seed,
    )

    if args.warm_start:
        print(f"Loading policy weights from {args.warm_start}...")
        warm_model = SAC.load(args.warm_start)
        if warm_model.observation_space.shape != vec_env.observation_space.shape:
            sys.exit(f"--warm-start model expects observations of shape {warm_model.observation_space.shape}, "
                     f"but this environment produces {vec_env.observation_space.shape}.")
        model.policy.load_state_dict(warm_model.policy.state_dict())
        del warm_model

    n_real = len(TRAIN_TRACKS)
    print(f"Starting {args.run}: {n_real} real + {len(proc_train_ids)} procedural training tracks, "
          f"{len(val_pool)} validation tracks, randomize={randomize}")
    model.learn(
        total_timesteps=args.timesteps,
        callback=[callback, checkpoint_callback],
        tb_log_name=args.run,
    )

    model.save(os.path.join(run_dir, "sac_racing_final"))
    print("Training finished!")


if __name__ == "__main__":
    main()
