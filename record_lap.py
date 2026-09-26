"""
Record one lap of telemetry for replay.py / export_video.py.

    python record_lap.py                              # Suzuka -> suzuka_lap.npz
    python record_lap.py --track be-1925 --out spa_lap.npz

The .npz holds the car's state at every simulation step, the driver inputs,
and the full-resolution track (centerline, width, curvature) plus metadata
(track name, model, lap time) so the renderer needs nothing else.
"""
import argparse
import json
import sys

import numpy as np
from stable_baselines3 import SAC

from config import (DEFAULT_MODEL_VERSION, TRACK_IDS, TRACK_WIDTHS, model_path_for,
                    obs_version_for, track_split)
from racing_env import RacingEnv
from track_preprocessing import build_track_pool, prepare_tracks


def circuit_name(track_id, circuits_dir="circuits"):
    """Human-readable name from the GeoJSON properties, e.g. 'Suzuka International Racing Course'."""
    try:
        with open(f"{circuits_dir}/{track_id}.geojson") as f:
            props = json.load(f)["features"][0]["properties"]
        return props.get("Name") or props.get("Location") or track_id
    except (OSError, KeyError, IndexError, ValueError):
        return track_id


def record_lap(model, env):
    """Drive one deterministic lap. Returns (telemetry dict of per-step arrays, termination reason)."""
    keys = ("x", "y", "theta", "v", "arc", "steering", "throttle")
    hist = {k: [] for k in keys}

    def log_state():
        hist["x"].append(env.car.x)
        hist["y"].append(env.car.y)
        hist["theta"].append(env.car.theta)
        hist["v"].append(env.car.v)
        hist["arc"].append(env._cumulative_arc)

    obs, _ = env.reset()
    done = False
    info = {}
    while not done:
        log_state()
        action, _ = model.predict(obs, deterministic=True)
        hist["steering"].append(float(action[0]))
        hist["throttle"].append(float(action[1]))
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    # Post-step state, so the last frame is where the lap ended; repeat the last input
    log_state()
    hist["steering"].append(hist["steering"][-1] if hist["steering"] else 0.0)
    hist["throttle"].append(hist["throttle"][-1] if hist["throttle"] else 0.0)
    return {k: np.array(v, dtype=float) for k, v in hist.items()}, info.get("termination_reason", "unknown")


def main():
    parser = argparse.ArgumentParser(description="Record one lap of telemetry to an .npz file")
    parser.add_argument("--track", default="jp-1962", choices=TRACK_IDS, metavar="TRACK_ID",
                        help="track ID from config.TRACK_IDS (default: jp-1962, Suzuka)")
    parser.add_argument("--model", default=None, help="version key from config.MODEL_PATHS, or a .zip path")
    parser.add_argument("--obs-version", type=int, choices=[1, 2, 3], default=None,
                        help="observation layout the model expects (default: inferred from --model)")
    parser.add_argument("--out", default="suzuka_lap.npz")
    args = parser.parse_args()

    prepare_tracks([args.track])
    model_path = model_path_for(args.model)
    print(f"Loading model from {model_path}...")
    model = SAC.load(model_path)

    env = RacingEnv(
        track_pool=build_track_pool(track_ids=[args.track]),
        track_widths=TRACK_WIDTHS,
        render_mode=None,
        obs_version=obs_version_for(args.model, model, args.obs_version),
    )

    print("Recording lap...")
    telemetry, reason = record_lap(model, env)
    steps = len(telemetry["x"]) - 1
    if reason != "lap_completed":
        sys.exit(f"Agent did not finish the lap ({reason} after {steps} steps); nothing saved.")

    lap_time = steps * env.dt
    np.savez(
        args.out,
        **telemetry,
        centerline=env.centerline,
        left_boundary=env.left_boundary,
        right_boundary=env.right_boundary,
        curvature=env.curvature_signed,
        track_width=env.track_width,
        dt=env.dt,
        track_length=env.track_length,
        lap_time=lap_time,
        track_id=args.track,
        track_name=circuit_name(args.track),
        track_split=track_split(args.track),
        model_name=args.model or DEFAULT_MODEL_VERSION,
    )
    print(f"Lap time {lap_time:.2f} s ({steps} steps). Saved to {args.out}")


if __name__ == "__main__":
    main()
