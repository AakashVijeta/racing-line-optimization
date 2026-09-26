"""
Watch the agent drive one track, with live telemetry in the terminal.

    python watch_agent.py                    # Suzuka, default model
    python watch_agent.py --track mc-1929    # Monaco
"""
import argparse
from stable_baselines3 import SAC
from racing_env import RacingEnv
from track_preprocessing import build_track_pool, prepare_tracks
from config import TRACK_IDS, TRACK_WIDTHS, model_path_for, obs_version_for

def main():
    parser = argparse.ArgumentParser(description="Watch the agent on a single track")
    parser.add_argument("--track", default="jp-1962", choices=TRACK_IDS, metavar="TRACK_ID",
                        help="track ID from config.TRACK_IDS (default: jp-1962, Suzuka)")
    parser.add_argument("--model", default=None, help="version key from config.MODEL_PATHS, or a .zip path")
    parser.add_argument("--obs-version", type=int, choices=[1, 2, 3], default=None,
                        help="observation layout the model expects (default: inferred from --model)")
    args = parser.parse_args()

    # Prepare and load ONLY the target track
    prepare_tracks([args.track])
    single_track_pool = build_track_pool(track_ids=[args.track])

    model_path = model_path_for(args.model)
    print(f"Loading model from {model_path}...")
    model = SAC.load(model_path)

    env = RacingEnv(
        track_pool=single_track_pool,
        track_widths=TRACK_WIDTHS,
        render_mode="human",
        obs_version=obs_version_for(args.model, model, args.obs_version),
    )

    print(f"Rendering agent on {args.track.upper()}...")
    print("Press Ctrl+C or close the window to stop.")

    obs, info = env.reset()
    steps = 0

    try:
        while not env.quit_requested:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

            # Render to the screen (the environment handles the 30 FPS clock internally)
            env.render()

            speed_kph = env.car.v * 3.6
            print(f"\rSpeed: {speed_kph:5.1f} km/h | Steer: {action[0]:5.2f} | Throttle: {action[1]:5.2f}", end="")

            if terminated or truncated:
                reason = info.get("termination_reason", "unknown")
                print(f"\nLap Concluded after {steps * env.dt:.2f} s. Reason: {reason}")
                print("Restarting...\n")
                obs, info = env.reset()
                steps = 0

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    finally:
        print("Shutting down...")
        env.close()

if __name__ == "__main__":
    main()
