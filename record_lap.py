"""
Record one lap of telemetry for replay.py / export_video.py.

    python record_lap.py                              # Suzuka -> suzuka_lap.npz
    python record_lap.py --track be-1925 --out spa_lap.npz
"""
import argparse
import sys
import numpy as np
from stable_baselines3 import SAC
from racing_env import RacingEnv
from track_preprocessing import build_track_pool, prepare_tracks
from config import TRACK_IDS, TRACK_WIDTHS, model_path_for, obs_version_for

def record_lap():
    parser = argparse.ArgumentParser(description="Record one lap of telemetry to an .npz file")
    parser.add_argument("--track", default="jp-1962", choices=TRACK_IDS, metavar="TRACK_ID",
                        help="track ID from config.TRACK_IDS (default: jp-1962, Suzuka)")
    parser.add_argument("--model", default=None, help="version key from config.MODEL_PATHS, or a .zip path")
    parser.add_argument("--obs-version", type=int, choices=[1, 2, 3], default=None,
                        help="observation layout the model expects (default: inferred from --model)")
    parser.add_argument("--out", default="suzuka_lap.npz")
    args = parser.parse_args()

    prepare_tracks([args.track])
    single_track_pool = build_track_pool(track_ids=[args.track])

    model_path = model_path_for(args.model)
    print(f"Loading optimal model from {model_path}...")
    model = SAC.load(model_path)

    env = RacingEnv(
        track_pool=single_track_pool,
        track_widths=TRACK_WIDTHS,
        render_mode=None,
        obs_version=obs_version_for(args.model, model, args.obs_version),
    )

    x_hist = []
    y_hist = []
    theta_hist = []
    v_hist = []
    throttle_hist = []
    steering_hist = []
    arc_hist = []
    
    obs, info = env.reset()
    done = False
    
    print("Recording lap...")
    while not done:
        x_hist.append(env.car.x)
        y_hist.append(env.car.y)
        theta_hist.append(env.car.theta)
        v_hist.append(env.car.v)
        arc_hist.append(env._cumulative_arc)
        
        action, _ = model.predict(obs, deterministic=True)
        steering_hist.append(action[0])
        throttle_hist.append(action[1])
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    # Append the post-step state so the final frame matches the end step exactly
    x_hist.append(env.car.x)
    y_hist.append(env.car.y)
    theta_hist.append(env.car.theta)
    v_hist.append(env.car.v)
    arc_hist.append(env._cumulative_arc)
    # Duplicate last action to keep arrays the same length
    steering_hist.append(steering_hist[-1] if steering_hist else 0.0)
    throttle_hist.append(throttle_hist[-1] if throttle_hist else 0.0)

    reason = info.get('termination_reason', 'unknown')
    steps = len(x_hist) - 1
    if reason != "lap_completed":
        sys.exit(f"Agent did not finish the lap ({reason} after {steps} steps); nothing saved.")

    # Get track geometry to save
    left_boundary = env.left_boundary
    right_boundary = env.right_boundary
    centerline = env.centerline

    out_file = args.out
    np.savez(
        out_file,
        x=np.array(x_hist),
        y=np.array(y_hist),
        theta=np.array(theta_hist),
        v=np.array(v_hist),
        throttle=np.array(throttle_hist),
        steering=np.array(steering_hist),
        arc=np.array(arc_hist),
        left_boundary=left_boundary,
        right_boundary=right_boundary,
        centerline=centerline,
        dt=env.dt,
        track_length=env.track_length
    )
    
    lap_time = steps * env.dt
    print(f"\nRecording finished!")
    print(f"Steps: {steps}")
    print(f"Lap Time: {lap_time:.2f} s")
    print(f"Reason: {reason}")
    print(f"Saved to {out_file}")

if __name__ == "__main__":
    record_lap()
