import os 
from stable_baselines3 import SAC
from racing_env import RacingEnv
from track_preprocessing import build_track_pool, prepare_tracks

track_ids = [f.split('.')[0] for f in os.listdir('./track_data') if f.endswith('.json')]
prepare_tracks(track_ids)
track_pool = build_track_pool(track_ids=track_ids)

# 1. Instantiate plain env (do NOT pass env directly into SAC.load)
env = RacingEnv(render_mode="human", track_pool=track_pool)

# 2. Load model
model = SAC.load("./models/sac_v10/best_model/best_model.zip")

print("Starting evaluation...")

while True:
    obs, info = env.reset()
    done = False
    step_count = 0
    
    # Print initial state
    print(f"\n[SPAWN] Track loaded. Pos: ({env.car.x:.2f}, {env.car.y:.2f}), Heading: {env.car.theta:.2f}")

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # print(f"Step {step_count} | Action: [Steer: {action[0]:.2f}, Thr: {action[1]:.2f}] | Pos: ({env.car.x:.2f}, {env.car.y:.2f}) | v={env.car.v:.2f}")
        
        env.render()
        done = terminated or truncated
        step_count += 1
        
        if terminated:
            print(f"Terminated at step {step_count}. Reason: {'Off-track' if reward <= -50 else 'Completed'}")