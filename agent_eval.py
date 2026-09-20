import csv
import numpy as np
from tqdm import tqdm
from stable_baselines3 import SAC
from racing_env import RacingEnv
from track_preprocessing import build_track_pool, prepare_tracks
from config import TRACK_IDS, TRACK_WIDTHS, get_model_path

def format_time(seconds):
    """Converts raw seconds into M:SS.ms format (e.g., 1:44.700)"""
    if seconds is None:
        return "DNF"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}:{secs:06.3f}"

def main():
    track_ids = TRACK_IDS
    track_widths = TRACK_WIDTHS

    prepare_tracks(track_ids)
    full_track_pool = build_track_pool(track_ids=track_ids)

    model_path = get_model_path("v13")
    print(f"Loading optimum model from {model_path}...\n")
    model = SAC.load(model_path)

    results = {}

    print(f"🏁 Starting Time Trials on {len(track_ids)} tracks (Headless Mode)...")
    
    for track_id in tqdm(track_ids, desc="Simulating Tracks", unit="track"):
        single_track_pool = {track_id: full_track_pool[track_id]}
        
        # 4. Pass track_widths and let max_steps scale dynamically
        env = RacingEnv(track_pool=single_track_pool, track_widths=track_widths, render_mode=None)
        
        obs, info = env.reset()
        done = False
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            
        reason = info.get("termination_reason", "unknown")
        
        if reason == "lap_completed": 
            lap_time = steps * env.dt
            avg_speed_kph = (env.track_length / lap_time) * 3.6
            results[track_id] = (lap_time, avg_speed_kph, "Finished")
        elif reason == "timeout" or truncated:
            arc_frac = env._cumulative_arc / env.track_length
            results[track_id] = (None, 0.0, f"Timeout at {steps} steps (Arc: {arc_frac:.2f})")
        else:
            arc_frac = env._cumulative_arc / env.track_length
            results[track_id] = (None, 0.0, f"{reason} at {steps} steps (Arc: {arc_frac:.2f})")
            
        env.close()

    # --- Print Formatted Leaderboard ---
    print("\n\n" + "=" * 65)
    print(f"{'🏆 TRACK LAP TIME LEADERBOARD 🏆':^65}")
    print("=" * 65)
    print(f"{'Track Name':<15} | {'Lap Time':<12} | {'Avg Speed':<12} | {'Status'}")
    print("-" * 65)
    
    # Sort results: Finished tracks by time (fastest first), then DNFs
    sorted_results = sorted(
        results.items(), 
        key=lambda x: x[1][0] if x[1][0] is not None else float('inf')
    )
    
    for track_id, (lap_time, avg_speed, status) in sorted_results:
        time_str = format_time(lap_time)
        speed_str = f"{avg_speed:.1f} km/h" if lap_time is not None else "---"
        print(f"{track_id.upper():<15} | {time_str:<12} | {speed_str:<12} | {status}")
        
    print("=" * 65)

    # --- Export to CSV ---
    csv_filename = "lap_times_leaderboard.csv"
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Track Name", "Lap Time (s)", "Formatted Time", "Avg Speed (km/h)", "Status"])
        for track_id, (lap_time, avg_speed, status) in sorted_results:
            writer.writerow([
                track_id.upper(), 
                round(lap_time, 3) if lap_time else "", 
                format_time(lap_time), 
                round(avg_speed, 1), 
                status
            ])
    print(f"\n📁 Full leaderboard saved to {csv_filename}")

if __name__ == "__main__":
    main()