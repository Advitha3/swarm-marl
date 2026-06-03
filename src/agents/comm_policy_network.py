import torch
import torch.nn as nn
import numpy as np


class CommPolicyNetwork(nn.Module):
    """
    Policy network with learned communication.
    
    Each drone:
    - Receives messages from other drones (prev timestep)
    - Combines own obs + received messages
    - Outputs action AND message for next timestep
    
    Two sub-networks:
    1. policy_net:  (obs + messages) → action
    2. message_net: obs → message vector
    
    Reference: Sukhbaatar et al. (2016) CommNet
               Das et al. (2019) TarMAC
    """

    def __init__(self, obs_dim, action_dim, message_dim=16,
                 num_agents=4, hidden_dim=64):
        super(CommPolicyNetwork, self).__init__()

        self.obs_dim = obs_dim          # 75
        self.action_dim = action_dim    # 4
        self.message_dim = message_dim  # 16
        self.num_agents = num_agents    # 4

        # How many message numbers each drone receives
        # = (num_agents - 1) other drones × message_dim
        self.received_message_dim = (num_agents - 1) * message_dim
        # = 3 × 16 = 48

        # Combined input to policy network
        self.combined_dim = obs_dim + self.received_message_dim
        # = 75 + 48 = 123

        # Network 1: Policy network
        # Takes combined obs + messages → outputs action
        self.policy_net = nn.Sequential(
            nn.Linear(self.combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()   # bounds motor commands to [-1, 1]
        )

        # Network 2: Message network
        # Takes only own obs → outputs message
        # Separate from policy — learns what to TELL others
        # independently from what to DO
        self.message_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, message_dim),
            nn.Tanh()   # bounds message values to [-1, 1]
        )

        # Learnable log std for stochastic policy
        # Starts at -1.0 → std ≈ 0.37 → moderate exploration
        self.log_std = nn.Parameter(
            torch.ones(action_dim) * -1.0
        )

        # Initialize policy final layer near zero
        # → stable hover before any learning
        nn.init.uniform_(self.policy_net[-2].weight, -0.01, 0.01)
        nn.init.zeros_(self.policy_net[-2].bias)

        # Initialize message final layer near zero
        # → empty messages before any learning
        nn.init.uniform_(self.message_net[-2].weight, -0.01, 0.01)
        nn.init.zeros_(self.message_net[-2].bias)

    def forward(self, obs, received_messages):
        """
        Forward pass — compute action mean and new message.

        obs shape:               (batch, obs_dim)        = (B, 75)
        received_messages shape: (batch, received_msg_dim) = (B, 48)

        Returns:
            action_mean: (B, 4)
            message:     (B, 16)
        """
        # Combine own observation with received messages
        # Concatenate along feature dimension
        combined = torch.cat([obs, received_messages], dim=-1)
        # combined shape: (B, 123)

        # Policy network: combined → action mean
        action_mean = self.policy_net(combined)
        # action_mean shape: (B, 4)

        # Message network: own obs only → message
        # Why only own obs? Message is what I know about myself
        # that others might find useful. Not what others told me.
        message = self.message_net(obs)
        # message shape: (B, 16)

        return action_mean, message

    def get_action(self, obs, received_messages):
        """
        Sample action from stochastic policy.
        Also returns new message for next timestep.
        Used during experience collection (with no_grad).
        """
        action_mean, message = self.forward(obs, received_messages)

        std = self.log_std.exp()
        dist = torch.distributions.Normal(action_mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        # Clamp to valid motor range
        action = torch.clamp(action, -1.0, 1.0)

        return action, log_prob, message

    def evaluate_actions(self, obs, received_messages, actions):
        """
        Evaluate already-taken actions under current policy.
        Used during PPO update with gradient tracking.
        """
        action_mean, _ = self.forward(obs, received_messages)

        std = self.log_std.exp()
        dist = torch.distributions.Normal(action_mean, std)

        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().mean()

        return log_prob, entropy


class CommCentralizedCritic(nn.Module):
    """
    Centralized critic with communication awareness.

    Input: global state = all drone obs + all messages
    = (4 × 75) + (4 × 16) = 300 + 64 = 364 numbers

    Including messages in critic input lets it evaluate
    not just what drones are doing but what they're communicating.
    """

    def __init__(self, obs_dim, message_dim, num_agents,
                 hidden_dim=128):
        super(CommCentralizedCritic, self).__init__()

        # Global state includes all obs AND all messages
        self.global_dim = (obs_dim + message_dim) * num_agents
        # = (75 + 16) * 4 = 364

        self.network = nn.Sequential(
            nn.Linear(self.global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
            # No Tanh — value is unbounded
        )

    def forward(self, global_obs, all_messages):
        """
        global_obs shape:   (batch, 4*75) = (B, 300)
        all_messages shape: (batch, 4*16) = (B, 64)
        """
        # Combine all observations and all messages
        combined = torch.cat([global_obs, all_messages], dim=-1)
        # combined shape: (B, 364)

        return self.network(combined)