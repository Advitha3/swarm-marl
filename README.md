# Swarm-MARL: Multi-Drone Formation using Multi-Agent RL

Training a swarm of 4 drones to fly in formation using Multi-Agent Reinforcement Learning (MARL) with curriculum learning.

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Demo

[Watch Demo Video](results/demo.mp4)

*4 drones learning formation flying using MAPPO + Curriculum Learning. Shows Stage 3 (30cm formation) — drones navigating to target positions.*

---

## Problem Statement

Training multiple drones to cooperatively reach formation targets is hard because:

- **Credit assignment** — shared reward makes individual contribution unclear
- **Non-stationarity** — all agents learn simultaneously, environment shifts constantly
- **Sparse rewards** — drones crash before reaching targets, giving no useful learning signal
- **Survival vs navigation tradeoff** — without curriculum, drones learn to hover safely rather than navigate

---

## Approach

**Algorithm:** MAPPO (Multi-Agent PPO) with CTDE
TRAINING (simulation):
Global State (300-dim) → Centralized Critic → Team Value
EXECUTION (real world):
Local Obs (75-dim) → Drone Policy → Motor Commands
No communication needed between drones

**Key Design Decisions:**
- **Parameter sharing** — one policy network shared by all 4 drones
- **Centralized critic** — sees all drone states (300-dim global state)
- **Curriculum learning** — progressive difficulty from hover to formation
- **Progress reward** — rewards movement toward target each step
- **Augmented observation** — each drone sees relative target position

---

## Curriculum Learning Pipeline

Standard MAPPO with fixed targets failed — drones learned to hover in place rather than navigate (distance stuck at 0.544m). Curriculum learning solved this:
Stage 0: Stable Hover     ✅ Mastered in 3 updates
Stage 1: Move 5cm         ✅ Mastered in ~38 updates
Stage 2: Move 15cm        ✅ Mastered in ~49 updates
Stage 3: Full Formation   ✅ Learning — dist 0.314 → 0.229m

Each stage automatically advances when mean distance stays below threshold for 3 consecutive updates.

---

## Results

### Single Drone Baseline

| Agent | Mean Reward | Avg Steps |
|-------|-------------|-----------|
| Random baseline | 18.068 | ~12 |
| Policy Gradient (ours) | 17.863 | ~11 |

**Finding:** Vanilla policy gradient struggles with sparse rewards, motivating MAPPO for the multi-agent setting.

---

### 4-Drone MAPPO — Vanilla vs Curriculum

| Experiment | Method | Distance | Outcome |
|-----------|--------|----------|---------|
| Baseline | Fixed targets | 0.544m stuck | ❌ No navigation |
| Ours | Curriculum MAPPO | 0.314 → 0.229m | ✅ Formation learned |

---

### Stage 3 — Full Formation Training Progress

| Update | Mean Distance | Reward | Episodes/batch |
|--------|--------------|--------|----------------|
| 50 | 0.314m | -17 | 91 |
| 164 | 0.303m | +0.7 | 94 |
| 256 | 0.299m | +179 | 54 |
| 313 | 0.246m | +331 | 48 |
| 381 | 0.229m | +739 | 28 |

**Drones learned to approach 30cm formation targets.**
- Distance reduced: 0.314m → 0.229m (27% improvement)
- Reward improved: -17 → +739
- Episodes per batch: 91 → 28 (drones surviving 3x longer)

---

### Key Research Findings

**1. Curriculum learning is essential for drone navigation**
Without curriculum, drones learned to hover safely rather than navigate. Curriculum forced progressive skill building: hover → tiny move → small move → formation.

**2. Stage transitions cause temporary regression**
When advancing stages, distance jumps before recovering. This reproduces Bengio et al. (2009) curriculum jump phenomenon. Example: Stage 0→1 caused dist 0.044→0.078 before recovering.

**3. Threshold sensitivity**
Too strict (0.04): never advances. Too generous (0.09): advances prematurely, fails at hard stages. Balanced (0.075): enables stable progression.

**4. MAPPO centralized critic stabilizes training**
Critic loss decreased steadily 0.99→0.54 over 100 updates, confirming centralized value estimation works for swarm tasks.

**5. Survival vs navigation tradeoff**
Reward increase in early Stage 3 was entirely explained by longer survival, not navigation. Policy learned to stay alive longer before learning to move toward targets.

---

## Network Architecture
Policy Network (shared across all 4 drones):
Input:  75-dim (72 drone state + 3 relative target)
Hidden: 64 → 64 (ReLU)
Output: 4 motor commands (Tanh)
Params: 9,288
Centralized Critic:
Input:  300-dim (75 × 4 drones concatenated)
Hidden: 128 → 128 (ReLU)
Output: 1 team value estimate
Params: 55,169

---

## Reward Function
For each drone i:
reward_i = -distance * 2.0          # distance penalty
+ progress * 10.0          # reward moving closer
+ 20.0 if reached target   # arrival bonus
- 5.0 if collision nearby  # collision penalty
team_reward = mean(reward_0, reward_1, reward_2, reward_3)

---

## Algorithm — MAPPO Update
ratio    = exp(log_prob_new - log_prob_old)
surr1    = ratio * advantages
surr2    = clip(ratio, 0.8, 1.2) * advantages
L_policy = -min(surr1, surr2) - 0.01 * entropy
L_critic = MSE(V(global_state), returns)
clip_grad_norm_(parameters, 0.5)
ppo_epochs = 4

---

## Project Structure
swarm-marl/
├── src/
│   ├── agents/
│   │   ├── policy_network.py    # PolicyNetwork + CentralizedCritic
│   │   ├── mappo_agent.py       # MAPPO with PPO clipping + CTDE
│   │   └── random_agent.py      # Random baseline agent
│   ├── envs/
│   │   ├── curriculum_env.py    # Progressive 4-stage curriculum
│   │   └── multi_drone_env.py   # Fixed formation environment
│   ├── train_curriculum.py      # Main training (curriculum MAPPO)
│   ├── train_mappo.py           # Vanilla MAPPO baseline
│   ├── train_ppo.py             # Single drone PPO baseline
│   └── evaluate.py              # GUI visualization
└── results/
├── demo.mp4                 # Training demo video
├── curriculum_log.txt       # Full training log
└── single_drone_results.md  # Single drone analysis

---

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

# Run vanilla MAPPO baseline
python src/train_mappo.py

# Run curriculum MAPPO (recommended)
python src/train_curriculum.py

# Visualize trained policy with GUI
python src/evaluate.py
```

---

## References

- Schulman et al. (2017) — Proximal Policy Optimization (PPO)
- Yu et al. (2021) — The Surprising Effectiveness of MAPPO
- Bengio et al. (2009) — Curriculum Learning (ICML)
- Panerati et al. (2021) — Gym-PyBullet-Drones (IROS)
- Vásárhelyi et al. (2018) — Optimized Flocking of Autonomous Drones
- Lowe et al. (2017) — MADDPG: Multi-Agent Actor-Critic

---

## Future Work

- Extend Stage 3 training to full convergence (dist < 0.15m)
- Add inter-drone communication channel
- Scale to 8-16 drones
- Sim-to-real transfer on physical hardware
- Automated curriculum threshold scheduling