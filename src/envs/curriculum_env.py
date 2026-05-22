import numpy as np
from gym_pybullet_drones.envs import MultiHoverAviary


class CurriculumFormationEnv:
    """
    Curriculum learning environment for 4-drone formation.
    
    Automatically advances through stages based on performance.
    
    Stage 0: Hover stably (targets = current position)
    Stage 1: Move 5cm to target
    Stage 2: Move 15cm to target  
    Stage 3: Move 30cm to target (full formation)
    
    Advances when mean_distance < threshold for 3 consecutive updates.
    
    Reference: Curriculum learning concept from Bengio et al. (2009)
    "Curriculum Learning" - ICML
    """

    # Stage definitions — distance from start position
    STAGES = [
        {
            'name': 'Stable Hover',
            'scale': 0.00,
            'threshold': 0.06,
            'description': 'Learn to hover without crashing'
        },
        {
            'name': 'Tiny Move',
            'scale': 0.05,
            'threshold': 0.055,
            'description': 'Move 5cm to target'
        },
        {
            'name': 'Small Move',
            'scale': 0.15,
            'threshold': 0.13,
            'description': 'Move 15cm to target'
        },
        {
            'name': 'Full Formation',
            'scale': 0.30,
            'threshold': 0.25,
            'description': 'Reach full formation targets'
        },
    ]

    def __init__(self, num_drones=4, gui=False):
        self.num_drones = num_drones
        self.gui = gui

        self.env = MultiHoverAviary(
            num_drones=num_drones,
            gui=gui
        )

        # Formation directions — unit vectors
        # Each drone flies in its assigned direction
        self.formation_dirs = np.array([
            [ 1.0,  0.0,  0.5],   # drone 0: forward + up
            [-1.0,  0.0,  0.5],   # drone 1: backward + up
            [ 0.0,  1.0,  0.5],   # drone 2: left + up
            [ 0.0, -1.0,  0.5],   # drone 3: right + up
        ])
        # Normalize direction vectors
        norms = np.linalg.norm(self.formation_dirs, axis=1, keepdims=True)
        self.formation_dirs = self.formation_dirs / norms

        # Current curriculum stage
        self.current_stage = 0
        self.consecutive_successes = 0
        self.required_successes = 3

        # Current targets (set on reset)
        self.targets = None
        self.start_positions = None

        self.obs_dim = 72 + 3   # 75
        self.collision_threshold = 0.2
        self.prev_distances = None
        self.step_count = 0

        print(f"Starting Stage 0: {self.STAGES[0]['name']}")
        print(f"  → {self.STAGES[0]['description']}")

    def _compute_targets(self, start_positions):
        """
        Compute targets based on current stage.
        Targets = start_position + direction * stage_scale
        """
        scale = self.STAGES[self.current_stage]['scale']
        targets = start_positions + self.formation_dirs * scale
        return targets

    def reset(self):
        self.step_count = 0
        self.prev_distances = None
        obs, info = self.env.reset()

        # Get starting positions
        self.start_positions = self._get_real_positions()

        # Set targets based on current stage
        self.targets = self._compute_targets(self.start_positions)

        return self._augment_obs(obs), info

    def try_advance_stage(self, mean_distance):
        """
        Check if we should advance to next stage.
        Called after each training update with mean distance.
        Returns True if stage advanced.
        """
        if self.current_stage >= len(self.STAGES) - 1:
            return False  # already at final stage

        threshold = self.STAGES[self.current_stage]['threshold']

        if mean_distance < threshold:
            self.consecutive_successes += 1
            print(f"  ✓ Success {self.consecutive_successes}"
                  f"/{self.required_successes} "
                  f"(dist={mean_distance:.3f} < {threshold})")

            if self.consecutive_successes >= self.required_successes:
                self.current_stage += 1
                self.consecutive_successes = 0
                stage = self.STAGES[self.current_stage]
                print(f"\n{'='*50}")
                print(f"ADVANCING TO STAGE {self.current_stage}: "
                      f"{stage['name']}")
                print(f"  → {stage['description']}")
                print(f"  → Target distance: {stage['scale']*100:.0f}cm")
                print(f"{'='*50}\n")
                return True
        else:
            # Reset consecutive counter if we fail
            self.consecutive_successes = 0

        return False

    def _get_real_positions(self):
        return np.array([
            self.env._getDroneStateVector(i)[:3]
            for i in range(self.num_drones)
        ])

    def _augment_obs(self, obs):
        """Add relative target to each drone's observation."""
        positions = self._get_real_positions()
        augmented = []
        for i in range(self.num_drones):
            if self.targets is not None:
                relative_target = self.targets[i] - positions[i]
            else:
                relative_target = np.zeros(3)
            drone_obs = np.concatenate([obs[i], relative_target])
            augmented.append(drone_obs)
        return np.array(augmented)

    def step(self, actions):
        obs, _, terminated, truncated, info = self.env.step(actions)
        self.step_count += 1

        positions = self._get_real_positions()
        individual_rewards = []
        distances = []

        for i in range(self.num_drones):
            drone_pos = positions[i]
            target_pos = self.targets[i]
            distance = np.linalg.norm(drone_pos - target_pos)
            distances.append(distance)

            # Distance reward
            reward_i = -distance * 2.0

            # Arrival bonus
            stage_threshold = self.STAGES[self.current_stage]['threshold']
            if distance < stage_threshold:
                reward_i += 20.0

            # Progress reward — key for learning movement
            if self.prev_distances is not None:
                progress = self.prev_distances[i] - distance
                reward_i += progress * 10.0

            individual_rewards.append(reward_i)

        # Collision penalty
        for i in range(self.num_drones):
            for j in range(i + 1, self.num_drones):
                dist_ij = np.linalg.norm(positions[i] - positions[j])
                if dist_ij < self.collision_threshold:
                    individual_rewards[i] -= 5.0
                    individual_rewards[j] -= 5.0

        team_reward = np.mean(individual_rewards)
        self.prev_distances = distances.copy()

        info['individual_rewards'] = individual_rewards
        info['distances'] = distances
        info['mean_distance'] = np.mean(distances)
        info['current_stage'] = self.current_stage
        info['stage_name'] = self.STAGES[self.current_stage]['name']

        return self._augment_obs(obs), team_reward, terminated, truncated, info

    def close(self):
        self.env.close()

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space