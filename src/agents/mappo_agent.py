import torch
import numpy as np


class MAPPOAgent:
    """
    Multi-Agent PPO Agent implementing CTDE.
    
    One shared policy for all drones (parameter sharing).
    One centralized critic seeing global state.
    
    Key difference from single drone PPO:
    - Policy sees local obs (72) per drone
    - Critic sees global obs (288) = all drones concatenated
    - PPO clipping added — prevents destructive updates
    """

    def __init__(self, policy, critic, num_drones=4,
                 lr_policy=1e-4, lr_critic=1e-4,
                 gamma=0.99, clip_eps=0.2, entropy_coef=0.01,
                 ppo_epochs=4):

        self.policy = policy        # shared across all drones
        self.critic = critic        # centralized — sees global state
        self.num_drones = num_drones

        # clip_eps = epsilon in PPO paper = 0.2
        # ratio stays within [1-0.2, 1+0.2] = [0.8, 1.2]
        self.clip_eps = clip_eps

        # entropy_coef = how much we reward exploration
        # 0.01 = small nudge toward diversity, not chaos
        self.entropy_coef = entropy_coef

        # ppo_epochs = how many times we reuse collected data
        # standard PPO reuses data 4-10 times per collection
        # more efficient than throwing away after one update
        self.ppo_epochs = ppo_epochs

        self.gamma = gamma

        # Separate optimizers for policy and critic
        # They learn different things — separate learning rates
        self.policy_optimizer = torch.optim.Adam(
            policy.parameters(), lr=lr_policy
        )
        self.critic_optimizer = torch.optim.Adam(
            critic.parameters(), lr=lr_critic
        )

    def select_actions(self, obs):
        """
        Select actions for ALL drones given their local observations.
        
        obs shape: (4, 72) — 4 drones, 72 numbers each
        
        We pass each drone's obs through the SAME policy network.
        This is parameter sharing.
        """
        # Convert numpy observation to PyTorch tensor
        obs_tensor = torch.FloatTensor(obs)
        # obs_tensor shape: (4, 72)

        with torch.no_grad():
            # Pass ALL 4 drone observations through policy at once
            # The network processes each row independently
            # (4, 72) → network → (4, 4) actions
            actions, log_probs = self.policy.get_action(obs_tensor)

        # Return as numpy for environment
        return actions.numpy(), log_probs.numpy()

    def compute_returns(self, rewards, dones, gamma=0.99):
        """
        Compute discounted returns — same as single drone.
        Walk backwards through episode, accumulate discounted reward.
        
        rewards: list of scalar team rewards
        dones:   list of booleans — did episode end?
        """
        returns = []
        R = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0  # reset at episode boundary
            R = reward + gamma * R
            returns.insert(0, R)
        return torch.FloatTensor(returns)

    def update(self, obs_list, global_obs_list, act_list,
               old_log_probs_list, returns):
        """
        Core MAPPO update — runs ppo_epochs times on collected data.
        
        obs_list:          local observations per drone (T, 4, 72)
        global_obs_list:   all drones concatenated    (T, 288)
        act_list:          actions taken              (T, 4, 4)
        old_log_probs_list: log probs at collection   (T, 4)
        returns:           discounted returns          (T,)
        """

        # Stack lists into tensors
        # (T, 4, 72) — T timesteps, 4 drones, 72 obs each
        obs_tensor = torch.FloatTensor(np.array(obs_list))

        # (T, 288) — T timesteps, global state
        global_obs_tensor = torch.FloatTensor(np.array(global_obs_list))

        # (T, 4, 4) — T timesteps, 4 drones, 4 motors each
        act_tensor = torch.FloatTensor(np.array(act_list))

        # (T, 4) — T timesteps, 4 drones, 1 log_prob each
        old_log_probs = torch.FloatTensor(np.array(old_log_probs_list))

        # Normalize returns — stabilizes training
        # subtract mean, divide by std → zero-centered unit variance
        if returns.std() > 1e-8:
            returns_norm = (returns - returns.mean()) / (returns.std() + 1e-8)
        else:
            returns_norm = returns - returns.mean()

        # Compute value estimates from centralized critic
        with torch.no_grad():
            # (T, 288) → critic → (T, 1) → squeeze → (T,)
            values = self.critic(global_obs_tensor).squeeze()

        # Advantage = reality - expectation
        # (T,) - (T,) = (T,)
        # expand to (T, 4) so each drone gets same advantage
        # team reward → same advantage signal for all drones
        advantages = (returns_norm - values).unsqueeze(1).expand(-1, self.num_drones)

        # Normalize advantages — further stabilizes training
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO runs multiple epochs on same data — more efficient
        for epoch in range(self.ppo_epochs):

            # Reshape for policy: (T*4, 72) — treat each drone-step as separate sample
            obs_flat = obs_tensor.view(-1, obs_tensor.shape[-1])
            act_flat = act_tensor.view(-1, act_tensor.shape[-1])

            # Evaluate actions under CURRENT policy (with gradients)
            # This gives us new_log_probs for PPO ratio
            new_log_probs, entropy = self.policy.evaluate_actions(
                obs_flat, act_flat
            )

            # Reshape back: (T*4,) → (T, 4)
            new_log_probs = new_log_probs.view(-1, self.num_drones)

            # PPO RATIO — core of PPO clipping
            # ratio = new_prob / old_prob
            # in log space: log(new/old) = log(new) - log(old)
            # exp() converts back to probability ratio
            ratio = (new_log_probs - old_log_probs).exp()

            # Unclipped objective — standard policy gradient
            surr1 = ratio * advantages

            # Clipped objective — prevents ratio going outside [0.8, 1.2]
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages

            # Take minimum — conservative update
            # if ratio too high: surr2 clips it down
            # if ratio too low:  surr2 clips it up
            # always take the pessimistic (lower) estimate
            policy_loss = -torch.min(surr1, surr2).mean()

            # Subtract entropy bonus — encourages exploration
            # minimize (policy_loss - entropy_coef * entropy)
            # = maximize reward AND maximize entropy
            policy_loss = policy_loss - self.entropy_coef * entropy

            # Update policy
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            # Gradient clipping — prevents exploding gradients
            # if gradient norm > 0.5, scale it down
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.policy_optimizer.step()

            # Update centralized critic
            current_values = self.critic(global_obs_tensor).squeeze()
            # MSE loss — critic learns to predict returns accurately
            critic_loss = torch.nn.functional.mse_loss(
                current_values, returns_norm
            )

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()

        return policy_loss.item(), critic_loss.item()