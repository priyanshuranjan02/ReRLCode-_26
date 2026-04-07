import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from env.human_aware_env import HumanAwareNavEnv
import numpy as np

env = HumanAwareNavEnv()
model = PPO.load("ppo_human_aware")

episodes = 20

success = 0
collisions = 0
total_steps = 0

for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    steps = 0

    while not done and steps < 500:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated
        steps += 1

        # Collision check
        if obs[6] < 0.3:  # nearest human distance
            collisions += 1
            break

    total_steps += steps

    if obs[4] < 0.5:  # goal distance
        success += 1

print("Episodes:", episodes)
print("Success rate:", success / episodes)
print("Collisions:", collisions)
print("Average steps:", total_steps / episodes)