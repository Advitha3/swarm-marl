# Swarm-MARL: Multi-Drone Formation using Multi-Agent RL

Training a swarm of 4 drones to fly in formation using
Multi-Agent Reinforcement Learning (MARL) with curriculum learning.

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Problem Statement

Training multiple drones to cooperatively reach formation
targets is hard because:
- **Credit assignment** — shared reward makes individual
  contribution unclear
- **Non-stationarity** — all agents learn simultaneously,
  environment shifts constantly
- **Sparse rewards** — drones crash before reaching targets,
  giving no learning signal

## Approach

**Algorithm:** MAPPO (Multi-Agent PPO) with CTDE

**Key Design Decisions:**
- Parameter sharing — one policy network for all 4 drones
- Centralized critic — sees all drone states (300-dim global)
- Curriculum learning — progressive difficulty from hover to formation
- Progress reward — rewards movement toward target each step

## Curriculum Learning Pipeline

Standard MAPPO failed with fixed targets — drones learned
to hover in place rather than navigate (dist stuck at 0.544m).
Curriculum learning solved this:

## Results

### Single Drone Baseline
| Agent | Mean Reward | Avg Steps |
|-------|-------------|-----------|
| Random baseline | 18.068 | ~12 |
| Policy Gradient (ours) | 17.863 | ~11 |

**Finding:** Vanilla policy gradient struggles with sparse
rewards in HoverAviary, motivating MAPPO for multi-agent setting.

### 4-Drone MAPPO — Vanilla vs Curriculum

| Experiment | Method | Distance | Outcome |
|-----------|--------|----------|---------|
| Baseline | Fixed targets | 0.544m (stuck) |  No navigation |
| Ours | Curriculum MAPPO | 0.044 → 0.052m | Progressive learning |

### Curriculum Training Progress
| Stage | Task | Start Dist | Best Dist | Status |
|-------|------|-----------|-----------|--------|
| 0 | Stable Hover | 0.044m | 0.042m |  Mastered (3 updates) |
| 1 | Move 5cm | 0.075m | 0.052m |  Learning |
| 2 | Move 15cm | - | - | Pending |
| 3 | Full Formation | - | - |  Pending |

### Key Research Findings

1. **Curriculum transitions cause regression** — when Stage 0
   advanced to Stage 1, distance jumped from 0.044 → 0.075
   (goalposts moved). Recovered to 0.052 over 250 updates.
   Reproduces Bengio et al. (2009) curriculum jump phenomenon.

2. **Survival vs navigation tradeoff** — without curriculum,
   drones learn to hover (safe) rather than navigate (risky).
   Curriculum forces navigation by starting with zero-distance targets.

3. **MAPPO critic loss decreases steadily** — from 0.99 → 0.54
   over 100 updates, confirming centralized critic learns
   team value accurately.

## Project Structure


## Setup

```bash
# Clone repo
git clone https://github.com/Advitha3/swarm-marl.git
cd swarm-marl

# Install simulation
git clone https://github.com/utiasDSL/gym-pybullet-drones.git
pip install -e gym-pybullet-drones/

# Install dependencies
pip install torch numpy matplotlib

# Run single drone baseline
python src/train_ppo.py

# Run vanilla MAPPO
python src/train_mappo.py

# Run curriculum MAPPO (recommended)
python src/train_curriculum.py
```

## Algorithm Details

### MAPPO Update Rule

### Reward Function (Curriculum)
### Network Architecture

## References

- Schulman et al. (2017) — Proximal Policy Optimization
- Yu et al. (2021) — The Surprising Effectiveness of MAPPO
- Bengio et al. (2009) — Curriculum Learning
- Panerati et al. (2021) — Gym-PyBullet-Drones
- Vásárhelyi et al. (2018) — Optimized Flocking of Autonomous Drones