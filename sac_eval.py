"""
Watch the agent drive laps on randomly chosen tracks.

    python sac_eval.py                 # default model, all tracks
    python sac_eval.py --model v12
"""
import argparse
from stable_baselines3 import SAC
from racing_env import RacingEnv
from track_preprocessing import build_track_pool, prepare_tracks
from config import TRACK_IDS, TRACK_WIDTHS, model_path_for, obs_version_for

def main():
    parser = argparse.ArgumentParser(description="Visual evaluation on random tracks")
    parser.add_argument("--model", default=None, help="version key from config.MODEL_PATHS, or a .zip path")
    parser.add_argument("--obs-version", type=int, choices=[1, 2, 3], default=None,
                        help="observation layout the model expects (default: inferred from --model)")
    args = parser.parse_args()

    track_ids = TRACK_IDS
    track_widths = TRACK_WIDTHS

    prepare_tracks(track_ids)
    track_pool = build_track_pool(track_ids=track_ids)

    model_path = model_path_for(args.model)
    print(f"Loading model from {model_path}...")
    model = SAC.load(model_path)

    env = RacingEnv(
        track_pool=track_pool,
        track_widths=track_widths,
        render_mode="human",
        obs_version=obs_version_for(args.model, model, args.obs_version),
    )

    print("Starting evaluation... (Ctrl+C or close window to stop)")

    try:
        while not env.quit_requested:
            obs, info = env.reset()
            done = False
            step_count = 0

            print(f"\nTrack loaded. Start: ({env.car.x:.2f}, {env.car.y:.2f}), heading {env.car.theta:.2f} rad")

            while not done and not env.quit_requested:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

                env.render()
                done = terminated or truncated
                step_count += 1

            if done:
                reason = info.get("termination_reason", "unknown")
                print(f"Episode ended at step {step_count} ({step_count * env.dt:.2f} s). Reason: {reason}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    finally:
        print("Shutting down...")
        env.close()

if __name__ == "__main__":
    main()
