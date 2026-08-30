import pygame
from stable_baselines3 import SAC
from racing_env import RacingEnv
from track_preprocessing import build_track_pool, prepare_tracks

def main():
    # --- 1. CHOOSE YOUR TRACK ---
    # Try "mc-1929" (Monaco), "be-1925" (Spa), or "it-1922" (Monza)
    TARGET_TRACK = "es-2026" 
    
    # --- 2. TRACK CONFIGURATION ---
    # We must provide the exact same widths the agent was trained on
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

    # Prepare and load ONLY the target track
    prepare_tracks([TARGET_TRACK])
    single_track_pool = build_track_pool(track_ids=[TARGET_TRACK])

    # --- 3. LOAD MODEL & ENVIRONMENT ---
    model_path = "./models/sac_v10_40tracks/best_model/best_model.zip"
    print(f"Loading optimum model from {model_path}...")
    model = SAC.load(model_path)

    env = RacingEnv(
        track_pool=single_track_pool, 
        track_widths=track_widths,
        render_mode="human"  # Enable PyGame rendering
    )

    print(f"🏎️ Rendering Agent on {TARGET_TRACK.upper()}...")
    print("Press Ctrl+C or close the window to stop.")

    # --- 4. RENDER LOOP ---
    obs, info = env.reset()
    done = False
    
    while True:
        # Predict the optimal action
        action, _ = model.predict(obs, deterministic=True)
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render to the screen (the environment handles the 30 FPS clock internally)
        env.render()
        
        # Print live telemetry to the terminal
        speed_kph = env.car.v * 3.6
        print(f"\rSpeed: {speed_kph:5.1f} km/h | Steer: {action[0]:5.2f} | Throttle: {action[1]:5.2f}", end="")

        if terminated or truncated:
            print(f"\nLap Concluded! Reason: {'Finished' if reward > 50 else 'Crashed/Timeout'}")
            print("Restarting...\n")
            obs, info = env.reset()

if __name__ == "__main__":
    main()