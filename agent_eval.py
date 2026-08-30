import csv
import numpy as np
from tqdm import tqdm
from stable_baselines3 import SAC
from racing_env import RacingEnv
from track_preprocessing import build_track_pool, prepare_tracks

def format_time(seconds):
    """Converts raw seconds into M:SS.ms format (e.g., 1:44.700)"""
    if seconds is None:
        return "DNF"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}:{secs:06.3f}"

def main():
    # 1. The full list of 40 track IDs
    track_ids = [
        "mc-1929", "az-2016", "nl-1948", "it-1953", "hu-1986", "sg-2008",
        "sa-2021", "pt-1972", "fr-1960", "au-1953", "ca-1978", "at-1969",
        "jp-1962", "mx-1962", "br-1940", "br-1977", "ar-1952", "za-1961",
        "us-1956", "us-2022", "us-2023", "es-2026", "de-1927", "de-1932",
        "it-1914", "be-1925", "it-1922", "es-1991", "ae-2009", "pt-2008",
        "tr-2005", "ru-2014", "qa-2004", "us-1909", "gb-1948", "bh-2002",
        "cn-2004", "us-2012", "my-1999", "fr-1969"
    ]

    # 2. The custom track widths dictionary
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

    prepare_tracks(track_ids)
    full_track_pool = build_track_pool(track_ids=track_ids)

    # 3. Update to the new 40-track model path
    model_path = "./models/sac_v10_40tracks/best_model/best_model.zip"
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
            
        if reward > 50.0: 
            lap_time = steps * env.dt
            avg_speed_kph = (env.track_length / lap_time) * 3.6
            results[track_id] = (lap_time, avg_speed_kph, "Finished")
        elif truncated:
            results[track_id] = (None, 0.0, f"DNF (Timeout at {steps} steps)")
        else:
            results[track_id] = (None, 0.0, f"DNF (Crashed at {steps} steps)")
            
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