import sys
sys.path.append('src')
from envs.multi_drone_env import FormationEnv
import numpy as np

env = FormationEnv(num_drones=4, gui=False)
obs, _ = env.reset()

print('Drone 0 full observation:')
print(obs[0])
print()
print('First 6 values:', obs[0, :6])
print('Obs min:', obs[0].min())
print('Obs max:', obs[0].max())

env = FormationEnv(num_drones=4, gui=False)
obs, _ = env.reset()

print('Normalized obs first 3:', obs[0, :3])

# Get real position
real_pos = env.env._getDroneStateVector(0)[:3]
print('Real position (meters):', real_pos)

# Check distance to target
target = env.targets[0]
distance = np.linalg.norm(real_pos - target)
print(f'Real distance to target: {distance:.3f}m')
print(f'Expected starting reward: {-distance + 0.1:.3f}')
env.close()