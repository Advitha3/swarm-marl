import torch
import numpy as np
from gym_pybullet_drones.envs import HoverAviary
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.policy_network import PolicyNetwork, ValueNetwork


def collect_experience(env, policy, device, min_steps=1024):
    all_obs, all_acts, all_rews, all_dones = [], [], [], []
    total_steps = 0
    episode_rewards = []

    while total_steps < min_steps:
        obs, _ = env.reset()
        episode_reward = 0

        for step in range(500):
            obs_tensor = torch.FloatTensor(obs).to(device)

            with torch.no_grad():
                action, log_prob = policy.get_action(obs_tensor)

            action_np = action.cpu().numpy()
            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            r = float(np.mean(reward))
            

            all_obs.append(obs_tensor)
            all_acts.append(action)
            all_rews.append(r)
            all_dones.append(terminated or truncated)

            episode_reward += r
            obs = next_obs
            total_steps += 1

            if terminated or truncated:
                break

        episode_rewards.append(episode_reward)

    return all_obs, all_acts, all_rews, all_dones, episode_rewards


def compute_returns(rewards, dones, gamma=0.99):
    returns = []
    R = 0
    for reward, done in zip(reversed(rewards), reversed(dones)):
        if done:
            R = 0
        R = reward + gamma * R
        returns.insert(0, R)
    return torch.FloatTensor(returns)


def inspect_policy(env, policy, device, num_episodes=3):
    print("\n--- POLICY INSPECTION ---")

    for ep in range(num_episodes):
        obs, _ = env.reset()
        print(f"\nEpisode {ep+1}:")
        print(f"{'Step':>6} {'Action (4 motors)':>35} {'Reward':>8}")
        print("-" * 55)

        for step in range(20):
            obs_tensor = torch.FloatTensor(obs).to(device)

            with torch.no_grad():
                action, _ = policy.get_action(obs_tensor)

            action_np = action.cpu().numpy()
            next_obs, reward, terminated, truncated, _ = env.step(action_np)

            r = float(np.mean(reward))
            a = action_np.flatten()
            print(f"{step+1:>6} [{a[0]:>6.3f}, {a[1]:>6.3f}, {a[2]:>6.3f}, {a[3]:>6.3f}] {r:>8.3f}")

            obs = next_obs
            if terminated or truncated:
                print(f"         >>> CRASHED at step {step+1}")
                break

    print("\n--- RANDOM AGENT COMPARISON ---")
    obs, _ = env.reset()
    print(f"\n{'Step':>6} {'Action (4 motors)':>35} {'Reward':>8}")
    print("-" * 55)

    for step in range(20):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        r = float(np.mean(reward))
        a = action.flatten()
        print(f"{step+1:>6} [{a[0]:>6.3f}, {a[1]:>6.3f}, {a[2]:>6.3f}, {a[3]:>6.3f}] {r:>8.3f}")
        obs = next_obs
        if terminated or truncated:
            print(f"         >>> CRASHED at step {step+1}")
            break


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    env = HoverAviary(gui=False)
    obs_dim = 72
    action_dim = 4

    policy = PolicyNetwork(obs_dim, action_dim).to(device)
    value_net = ValueNetwork(obs_dim).to(device)

    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=1e-4)

    print(f"\n{'Update':>8} {'Mean Ep Reward':>16} {'Episodes':>10} {'Steps':>8}")
    print("-" * 48)

    best_reward = -float('inf')

    for update in range(200):
        obs_list, act_list, rew_list, done_list, ep_rewards = \
            collect_experience(env, policy, device, min_steps=1024)

        returns = compute_returns(rew_list, done_list).to(device)
        obs_tensor = torch.stack(obs_list)
        act_tensor = torch.stack(act_list)

        if returns.std() > 1e-8:
            returns_norm = (returns - returns.mean()) / (returns.std() + 1e-8)
        else:
            returns_norm = returns - returns.mean()

        mean = policy(obs_tensor)
        std = policy.log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(act_tensor).sum(dim=-1)

        values = value_net(obs_tensor).squeeze()
        value_loss = torch.nn.functional.mse_loss(values, returns_norm)

        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        advantages = (returns_norm - values.detach())
        entropy = dist.entropy().mean()
        policy_loss = -(log_probs * advantages).mean()

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        mean_ep_reward = np.mean(ep_rewards)
        num_episodes = len(ep_rewards)
        total_steps = len(rew_list)

        if mean_ep_reward > best_reward:
            best_reward = mean_ep_reward
            torch.save(policy.state_dict(), "results/best_policy.pt")

        print(f"{update+1:>8} {mean_ep_reward:>16.3f} {num_episodes:>10} {total_steps:>8}")

    print("-" * 48)
    print(f"Best mean episode reward: {best_reward:.3f}")
    print(f"Random agent baseline:    18.068")
    print(f"Improvement: {((best_reward - 18.068) / 18.068 * 100):.1f}%")

    policy.load_state_dict(torch.load("results/best_policy.pt"))
    policy.eval()
    inspect_policy(env, policy, device)

    env.close()


if __name__ == "__main__":
    main()