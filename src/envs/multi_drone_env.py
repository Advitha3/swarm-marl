import numpy as np
from gym_pybullet_drones.envs import MultiHoverAviary


class FormationEnv:
    """
    Custom formation flying environment for 4 drones.
    
    Reward Design v3:
    - Distance reward:   -distance * 2.0 (amplified)
    - Progress reward:   +10 * (prev_dist - curr_dist) per step
    - Arrival bonus:     +50 when within threshold
    - No survival bonus: removed to prevent hovering in place
    
    Key insight: Progress reward directly incentivizes movement
    toward target every step, not just being close.
    """

    def __init__(self, num_drones=4, gui=False):
        self.num_drones = num_drones
        self.gui = gui

        self.env = MultiHoverAviary(
            num_drones=num_drones,
            gui=gui
        )

        self.targets = np.array([
            [ 0.3,  0.0,  0.3],
            [-0.3,  0.0,  0.3],
            [ 0.0,  0.3,  0.3],
            [ 0.0, -0.3,  0.3],
        ])

        self.obs_dim = 72 + 3  # 75
        self.collision_threshold = 0.2
        self.arrival_threshold = 0.15
        self.prev_distances = None
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        self.prev_distances = None
        obs, info = self.env.reset()
        return self._augment_obs(obs), info

    def _get_real_positions(self):
        return np.array([
            self.env._getDroneStateVector(i)[:3]
            for i in range(self.num_drones)
        ])

    def _augment_obs(self, obs):
        """Add relative target position to each drone's observation."""
        positions = self._get_real_positions()
        augmented = []
        for i in range(self.num_drones):
            relative_target = self.targets[i] - positions[i]
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

            # Distance reward — amplified signal
            reward_i = -distance * 2.0

            # Arrival bonus — strong incentive to reach target
            if distance < self.arrival_threshold:
                reward_i += 50.0

            # Progress reward — reward for moving CLOSER this step
            # This is the key fix — incentivizes movement directly
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

        # Store for next step's progress calculation
        self.prev_distances = distances.copy()

        info['individual_rewards'] = individual_rewards
        info['distances'] = distances
        info['mean_distance'] = np.mean(distances)
        info['positions'] = positions.tolist()

        return self._augment_obs(obs), team_reward, terminated, truncated, info

    def close(self):
        self.env.close()

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space