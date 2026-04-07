import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import numpy as np

from env.human_aware_env import HumanAwareNavEnv

# Create environment
def make_env():
    env = HumanAwareNavEnv()
    env = Monitor(env, filename="train/monitor.csv")
    return env

env = DummyVecEnv([make_env])

# PPO Model
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    gamma=0.99,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    verbose=1
)

# Train
TIMESTEPS = 100_000
model.learn(total_timesteps=TIMESTEPS)

# Save model
model.save("ppo_human_aware")

print("✅ Training completed and model saved")