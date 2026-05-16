# Swarm-MARL: Multi-Drone Coordination using Multi-Agent RL

Training drone swarms to cooperatively navigate using 
Multi-Agent Reinforcement Learning (MARL).

## Project Goal
Implement and compare MARL algorithms (PPO → MAPPO) for 
decentralized swarm coordination in simulation.

## Current Progress
- [x] Single drone RL baseline (Policy Gradient)
- [ ] Multi-drone environment (4 drones)
- [ ] Cooperative navigation task
- [ ] MAPPO implementation
- [ ] Swarm benchmark results

## Single Drone Results
| Agent | Mean Reward | Avg Steps |
|-------|-------------|-----------|
| Random baseline | 18.068 | ~12 |
| Policy Gradient (ours) | 17.863 | ~11 |

**Finding:** Vanilla policy gradient struggles with sparse 
rewards, motivating MAPPO for the multi-agent setting.

## Stack
- Python 3.10
- PyTorch
- Gym-PyBullet-Drones
- Ubuntu 24

## Setup
```bash
git clone https://github.com/Advitha3/swarm-marl.git
cd swarm-marl
pip install -e gym-pybullet-drones/
pip install torch numpy matplotlib
python src/train_ppo.py
```

## References
- Gym-PyBullet-Drones (UTias DSL)
- MAPPO: Multi-Agent PPO
- PPO: Proximal Policy Optimization (Schulman et al. 2017)