import sys
import pygame
from stable_baselines3 import SAC
from racing_env import RacingEnv
from track_preprocessing import build_track_pool, prepare_tracks
from config import TRACK_IDS, TRACK_WIDTHS, get_model_path

def main():
    track_ids = TRACK_IDS
    track_widths = TRACK_WIDTHS

    prepare_tracks(track_ids)
    track_pool = build_track_pool(track_ids=track_ids)

    env = RacingEnv(
        track_pool=track_pool,
        track_widths=track_widths,
        render_mode="human"
    )

    model_path = get_model_path("v13")
    print(f"Loading model from {model_path}...")
    model = SAC.load(model_path)

    print("Starting evaluation... (Ctrl+C or close window to stop)")
    running = True

    try:
        while running:
            obs, info = env.reset()
            done = False
            step_count = 0

            print(f"\n[SPAWN] Track loaded. Pos: ({env.car.x:.2f}, {env.car.y:.2f}), Heading: {env.car.theta:.2f}")

            while not done and running:
                # Check for Pygame QUIT events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break

                if not running:
                    break

                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

                env.render()
                done = terminated or truncated
                step_count += 1

            if terminated:
                print(f"Terminated at step {step_count}. Reason: {'Off-track' if reward <= -50 else 'Completed'}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    finally:
        print("Shutting down...")
        env.close()
        pygame.quit()
        sys.exit(0)

if __name__ == "__main__":
    main()