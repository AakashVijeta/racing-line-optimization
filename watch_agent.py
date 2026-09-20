import sys
import pygame
from stable_baselines3 import SAC
from racing_env import RacingEnv
from track_preprocessing import build_track_pool, prepare_tracks
from config import TRACK_WIDTHS, get_model_path

def main():
    # --- 1. CHOOSE YOUR TRACK ---
    # Try "mc-1929" (Monaco), "be-1925" (Spa), or "it-1922" (Monza)
    TARGET_TRACK = "jp-1962" 
    
    # --- 2. TRACK CONFIGURATION ---
    track_widths = TRACK_WIDTHS

    # Prepare and load ONLY the target track
    prepare_tracks([TARGET_TRACK])
    single_track_pool = build_track_pool(track_ids=[TARGET_TRACK])

    # --- 3. LOAD MODEL & ENVIRONMENT ---
    model_path = get_model_path("v13")
    print(f"Loading optimum model from {model_path}...")
    model = SAC.load(model_path)

    env = RacingEnv(
        track_pool=single_track_pool, 
        track_widths=track_widths,
        render_mode="human"  # Enable PyGame rendering
    )

    print(f"🏎️ Rendering Agent on {TARGET_TRACK.upper()}...")
    print("Press Ctrl+C or close the window to stop.")

    # Initialize pygame so event.get() doesn't crash before render() is called
    pygame.init()

    # --- 4. RENDER LOOP (with graceful exit) ---
    obs, info = env.reset()
    running = True
    
    try:
        while running:
            # Check for Pygame QUIT events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
            
            if not running:
                break

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
                reason = info.get("termination_reason", "unknown")
                print(f"\nLap Concluded! Reason: {reason}")
                print("Restarting...\n")
                obs, info = env.reset()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    finally:
        print("Shutting down...")
        env.close()
        pygame.quit()

if __name__ == "__main__":
    main()