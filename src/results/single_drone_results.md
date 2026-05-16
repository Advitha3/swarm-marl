## Single Drone Baseline Results

### Environment
- HoverAviary (gym-pybullet-drones)
- Observation: 72-dimensional state vector
- Action: 4 continuous motor thrusts [-1, 1]

### Algorithm
- Policy Gradient with Advantage Estimation
- Policy Network: 72 → 64 → 64 → 4 (Tanh)
- Value Network: 72 → 64 → 64 → 1
- Learning rate: 1e-4
- Steps per update: 1024
- Total updates: 200

### Results
| Agent | Mean Episode Reward | Avg Steps |
|-------|-------------------|-----------|
| Random baseline | 18.068 | ~12 |
| Policy Gradient (ours) | 17.863 | ~11 |

### Key Finding
Vanilla policy gradient struggles with the sparse, 
flat reward signal in HoverAviary. Reward shaping 
and richer environments are needed for meaningful 
learning — motivating our MAPPO approach for 
multi-agent coordination.