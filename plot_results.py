import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load monitor file
log = pd.read_csv("train/monitor.csv", skiprows=1)

# Timesteps = cumulative episode lengths
timesteps = np.cumsum(log["l"])

# Episode rewards
rewards = log["r"]

# Moving average smoothing
window = 10
rewards_smooth = rewards.rolling(window).mean()

# Plot
plt.figure()
plt.plot(timesteps, rewards, alpha=0.3, label="Episode Reward")
plt.plot(timesteps, rewards_smooth, linewidth=2, label="Smoothed Reward")

plt.xlabel("Timesteps")
plt.ylabel("Episode Reward")
plt.title("PPO Training – Human-Aware Navigation")
plt.legend()
plt.grid(True)
plt.show()