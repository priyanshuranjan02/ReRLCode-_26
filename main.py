from env.human_aware_env import HumanAwareNavEnv
import numpy as np

env = HumanAwareNavEnv()
obs, _ = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)

print(obs)

obs, reward, terminated, truncated, info = env.step(action)
print("Reward:", reward)