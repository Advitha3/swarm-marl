import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):

    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

        self.log_std = nn.Parameter(torch.zeros(action_dim)* -0.5)

    def forward(self, obs):
        return self.network(obs)

    def get_action(self, obs):
        mean = self.forward(obs)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Clip actions to valid range
        action = torch.clamp(action, -1.0, 1.0)
        
        return action, log_prob
class ValueNetwork(nn.Module):
    """
    The drone's critic — estimates how good a state is.
    Input: observation (72 numbers)
    Output: single number — expected future reward from this state
    """

    def __init__(self, obs_dim, hidden_dim=64):
        super(ValueNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # single value output — no activation
        )

    def forward(self, obs):
        return self.network(obs)


if __name__ == "__main__":
    net = PolicyNetwork(obs_dim=72, action_dim=4)
    dummy_obs = torch.randn(1, 72)
    action, log_prob = net.get_action(dummy_obs)
    print(f"Action shape: {action.shape}")
    print(f"Action values: {action}")
    print(f"Log prob: {log_prob}")