import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    """
    Each drone's local policy — the Actor in Actor-Critic.
    
    Input:  local observation (72 numbers — this drone only)
    Output: action distribution (4 motor thrust values)
    
    UNCHANGED from single drone — each drone sees only itself.
    This is the DECENTRALIZED part of CTDE.
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()

        # Same 3-layer network as before
        # obs_dim=72 → hidden=64 → hidden=64 → action_dim=4
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # bounds output to [-1, 1] for motor commands
        )

        nn.init.uniform_(self.network[-2].weight, -0.01, 0.01)
        nn.init.zeros_(self.network[-2].bias)

        # Learnable log standard deviation
        # exp(log_std) = std → always positive
        # starts at -0.5 → std = 0.6 → less random than before
        self.log_std = nn.Parameter(torch.ones(action_dim) * -2.0)

    def forward(self, obs):
        # obs shape: (batch_size, 72)
        # returns:   (batch_size, 4) — action means
        return self.network(obs)

    def get_action(self, obs):
        """
        Sample action from stochastic policy.
        Used during experience collection (with no_grad).
        """
        mean = self.forward(obs)
        std = self.log_std.exp()

        # Normal distribution: mean from network, std learned separately
        dist = torch.distributions.Normal(mean, std)

        # Sample action — stochastic during training, enables exploration
        action = dist.sample()

        # log probability of this action — needed for PPO ratio
        log_prob = dist.log_prob(action).sum(dim=-1)

        # Clamp to valid motor range — prevents out-of-bounds actions
        action = torch.clamp(action, -1.0, 1.0)

        return action, log_prob

    def evaluate_actions(self, obs, actions):
        """
        Compute log_probs and entropy for ALREADY TAKEN actions.
        Used during PPO update with gradient tracking.
        
        Different from get_action() which samples NEW actions.
        This evaluates actions we already collected.
        """
        mean = self.forward(obs)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mean, std)

        # log probability of the actions we already took
        log_prob = dist.log_prob(actions).sum(dim=-1)

        # entropy — how spread out is our distribution?
        # higher entropy = more exploration
        entropy = dist.entropy().mean()

        return log_prob, entropy


class CentralizedCritic(nn.Module):
    """
    The centralized critic — the Critic in Actor-Critic.
    
    Input:  GLOBAL state (all 4 drones concatenated = 288 numbers)
    Output: single value V(s) — how good is the TEAM doing?
    
    This is the CENTRALIZED part of CTDE.
    Only used during training — thrown away at execution time.
    """
    def __init__(self, global_obs_dim, hidden_dim=128):
        super(CentralizedCritic, self).__init__()

        # Larger hidden dim than policy (128 vs 64)
        # Because global state is more complex — needs more capacity
        # 288 inputs vs 72 — 4x more information to process
        self.network = nn.Sequential(
            nn.Linear(global_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
            # NO Tanh here — value is unbounded
            # could be 5, 50, 500 — we don't want to squish it
        )

    def forward(self, global_obs):
        # global_obs shape: (batch_size, 288)
        # returns:          (batch_size, 1) — team value estimate
        return self.network(global_obs)