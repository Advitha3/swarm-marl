import numpy as np
from gym_pybullet_drones.envs import HoverAviary

env = HoverAviary(gui=False)
obs, info = env.reset()
print(f"Observation shape: {obs.shape}")
print(f"Action shape: {env.action_space.shape}")

for step in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

print("Environment working correctly ✅")
env.close()