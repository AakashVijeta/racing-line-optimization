import argparse
import csv
import os
from tqdm import tqdm
from stable_baselines3 import SAC
from racing_env import RacingEnv
from track_preprocessing import build_track_pool, prepare_tracks
from config import DEFAULT_MODEL_VERSION, TRACK_IDS, TRACK_WIDTHS, model_path_for, obs_version_for, track_split

def format_time(seconds):
    """Converts raw seconds into M:SS.ms format (e.g., 1:44.700)"""
    if seconds is None:
        return "DNF"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}:{secs:06.3f}"

def default_results_path(model):
    """results/<name>.csv, named after the model key or the .zip's run directory."""
    name = model or DEFAULT_MODEL_VERSION
    if name.endswith(".zip"):
        parts = os.path.normpath(name).split(os.sep)
        # models/<run>/best_model/best_model.zip -> <run>; otherwise the file stem
        name = parts[-3] if len(parts) >= 3 and parts[-2] == "best_model" else os.path.splitext(parts[-1])[0]
    return os.path.join("results", f"{name}.csv")


def run_lap(model, track_id, track_data, obs_version, widths=None):
    """Drive one deterministic lap. Returns (lap_time or None, avg_speed_kph, status)."""
    env = RacingEnv(track_pool={track_id: track_data}, track_widths=widths or TRACK_WIDTHS,
                    render_mode=None, obs_version=obs_version)
    obs, info = env.reset()
    done = False
    steps = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

    reason = info.get("termination_reason", "unknown")
    arc_frac = env._cumulative_arc / env.track_length
    env.close()

    if reason == "lap_completed":
        lap_time = steps * env.dt
        return lap_time, (env.track_length / lap_time) * 3.6, "Finished"
    if reason == "timeout" or truncated:
        return None, 0.0, f"Timeout at {steps} steps (Arc: {arc_frac:.2f})"
    return None, 0.0, f"{reason} at {steps} steps (Arc: {arc_frac:.2f})"


def run_time_trials(model, track_pool, obs_version, widths=None, progress=False):
    """One lap per track in track_pool -> {track_id: (lap_time or None, avg_speed_kph, status)}."""
    items = track_pool.items()
    if progress:
        items = tqdm(items, desc="Simulating Tracks", unit="track")
    return {tid: run_lap(model, tid, data, obs_version, widths) for tid, data in items}


def main():
    parser = argparse.ArgumentParser(description="Headless time trial on every track")
    parser.add_argument("--model", default=None, help="version key from config.MODEL_PATHS, or a .zip path")
    parser.add_argument("--obs-version", type=int, choices=[1, 2, 3], default=None,
                        help="observation layout the model expects (default: inferred from --model)")
    parser.add_argument("--procedural", choices=["val", "test"], default=None,
                        help="also time-trial a procedural track set (see config.PROC_SIZES)")
    parser.add_argument("--out", default=None, help="CSV path (default: results/<model>.csv)")
    args = parser.parse_args()

    track_ids = TRACK_IDS

    prepare_tracks(track_ids)
    full_track_pool = build_track_pool(track_ids=track_ids)
    widths = dict(TRACK_WIDTHS)
    if args.procedural:
        from track_generator import procedural_split
        proc_pool, proc_widths = procedural_split(args.procedural)
        full_track_pool.update(proc_pool)
        widths.update(proc_widths)
        track_ids = list(full_track_pool)

    model_path = model_path_for(args.model)
    print(f"Loading model from {model_path}...\n")
    model = SAC.load(model_path)
    obs_version = obs_version_for(args.model, model, args.obs_version)

    print(f"Starting time trials on {len(track_ids)} tracks (headless)...")
    results = run_time_trials(model, full_track_pool, obs_version, widths=widths, progress=True)

    # --- Print Formatted Leaderboard ---
    print("\n\n" + "=" * 65)
    print(f"{'TRACK LAP TIME LEADERBOARD':^65}")
    print("=" * 65)
    print(f"{'Track Name':<15} | {'Split':<5} | {'Lap Time':<12} | {'Avg Speed':<12} | {'Status'}")
    print("-" * 65)
    
    # Sort results: Finished tracks by time (fastest first), then DNFs
    sorted_results = sorted(
        results.items(), 
        key=lambda x: x[1][0] if x[1][0] is not None else float('inf')
    )
    
    for track_id, (lap_time, avg_speed, status) in sorted_results:
        time_str = format_time(lap_time)
        speed_str = f"{avg_speed:.1f} km/h" if lap_time is not None else "---"
        print(f"{track_id.upper():<15} | {track_split(track_id):<5} | {time_str:<12} | {speed_str:<12} | {status}")

    print("=" * 65)
    for real in (True, False):
        for split in ("train", "val", "test"):
            ids = [t for t in track_ids if track_split(t) == split and t.startswith("proc-") != real]
            if ids:
                done = sum(1 for t in ids if results[t][0] is not None)
                label = split if real else f"procedural {split}"
                print(f"{label:>16}: {done}/{len(ids)} laps completed")

    # --- Export to CSV ---
    csv_filename = args.out or default_results_path(args.model)
    os.makedirs(os.path.dirname(csv_filename) or ".", exist_ok=True)
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Track Name", "Split", "Lap Time (s)", "Formatted Time", "Avg Speed (km/h)", "Status"])
        for track_id, (lap_time, avg_speed, status) in sorted_results:
            writer.writerow([
                track_id.upper(),
                track_split(track_id),
                round(lap_time, 3) if lap_time else "", 
                format_time(lap_time), 
                round(avg_speed, 1), 
                status
            ])
    print(f"\nFull leaderboard saved to {csv_filename}")

if __name__ == "__main__":
    main()