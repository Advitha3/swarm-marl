import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.multi_drone_env import FormationEnv
from agents.policy_network import PolicyNetwork, CentralizedCritic
from agents.mappo_agent import MAPPOAgent


def collect_experience(env, agent, min_steps=1024):
    """
    Collect experience from ALL 4 drones simultaneously.
    
    Returns separate lists for:
    - local observations (each drone sees only itself)
    - global observations (all drones concatenated for critic)
    - actions, log_probs, rewards, dones
    """
    # Storage for experience
    obs_list = []           # (T, 4, 72) local observations
    global_obs_list = []    # (T, 288)   global state for critic
    act_list = []           # (T, 4, 4)  actions per drone
    log_prob_list = []      # (T, 4)     log probs per drone
    reward_list = []        # (T,)       team reward
    done_list = []          # (T,)       episode end flag

    episode_rewards = [] 
    episode_distance = []   # track full episode rewards for logging
    total_steps = 0

    while total_steps < min_steps:
        # Reset environment — all 4 drones back to start
        obs, _ = env.reset()
        # obs shape: (4, 72)

        episode_reward = 0
        episode_dist =  []

        for step in range(500):
            # Global state = concatenate all drone observations
            # (4, 72) → flatten → (288,)
            global_obs = obs.flatten()

            # Select actions for all 4 drones
            # actions shape: (4, 4)
            # log_probs shape: (4,)
            actions, log_probs = agent.select_actions(obs)

            # Step environment with all 4 drone actions
            next_obs, reward, terminated, truncated, info = env.step(actions)
            # reward is scalar — team reward
            # terminated is bool — any drone crashed?

            done = terminated or truncated

            # Store this timestep
            obs_list.append(obs.copy())           # (4, 72)
            global_obs_list.append(global_obs)    # (288,)
            act_list.append(actions.copy())        # (4, 4)
            log_prob_list.append(log_probs.copy()) # (4,)
            reward_list.append(float(reward))      # scalar
            done_list.append(done)                 # bool
    

            episode_reward += float(reward)

            #track mean distance to target 
            if 'mean_distance' in info:
                episode_dist.append(info['mean_distance'])
            obs = next_obs
            total_steps += 1

            if done:
                break

        episode_rewards.append(episode_reward)
        if episode_dist:
            episode_distance.append(np.mean(episode_dist))

    return (obs_list, global_obs_list, act_list,
            log_prob_list, reward_list, done_list,
            episode_rewards,episode_distance)


def main():
    device = torch.device("cpu")
    print("Initializing MAPPO for 4-drone swarm...")

    # Environment
    num_drones = 4
    
    env = FormationEnv(num_drones=num_drones, gui=False)

    # Dimensions
    obs_dim = 75          # each drone sees only itself
    action_dim = 4        # 4 motor commands per drone
    global_obs_dim = obs_dim * num_drones  # 288 for centralized critic

    print(f"Local obs dim:  {obs_dim}")
    print(f"Global obs dim: {global_obs_dim}")
    print(f"Action dim:     {action_dim}")

    # Networks
    # ONE shared policy for all drones — parameter sharing
    policy = PolicyNetwork(obs_dim, action_dim, hidden_dim=64)

    # ONE centralized critic — sees all drones
    critic = CentralizedCritic(global_obs_dim, hidden_dim=128)

    print(f"Policy parameters:  {sum(p.numel() for p in policy.parameters())}")
    print(f"Critic parameters:  {sum(p.numel() for p in critic.parameters())}")

    # MAPPO Agent
    agent = MAPPOAgent(
        policy=policy,
        critic=critic,
        num_drones=num_drones,
        lr_policy=1e-4,
        lr_critic=1e-4,
        gamma=0.99,
        clip_eps=0.2,       # PPO clip range [0.8, 1.2]
        entropy_coef=0.01,  # exploration bonus
        ppo_epochs=4        # reuse data 4 times
    )

    print(f"\n{'Update':>8} {'Mean Ep Reward':>16} {'Mean Dist':>10} "
          f"{'Episodes':>10} {'Steps':>8} {'P Loss':>8} {'C Loss':>8}")
    print("-" * 80)
    best_reward = -float('inf')

    for update in range(500):
        # Collect 1024+ steps from 4-drone environment
        (obs_list, global_obs_list, act_list,
         log_prob_list, reward_list, done_list,
         ep_rewards,ep_distances) = collect_experience(env, agent, min_steps=2048)

        # Compute discounted returns
        returns = agent.compute_returns(reward_list, done_list)

        # MAPPO update
        policy_loss, critic_loss = agent.update(
            obs_list, global_obs_list, act_list,
            log_prob_list, returns
        )

        # Logging
        mean_ep_reward = np.mean(ep_rewards)
        num_episodes = len(ep_rewards)
        total_steps = len(reward_list)

        if mean_ep_reward > best_reward:
            best_reward = mean_ep_reward
            # Save best policy
            torch.save(policy.state_dict(), "results/best_mappo_policy.pt")

        mean_dist = np.mean(ep_distances) if ep_distances else 0
        print(f"{update+1:>8} {mean_ep_reward:>16.3f} {mean_dist:>10.3f} "
              f"{num_episodes:>10} {total_steps:>8} {policy_loss:>8.4f} {critic_loss:>8.4f}")


    print("-" * 75)
    print(f"Best mean episode reward: {best_reward:.3f}")
    env.close()


if __name__ == "__main__":
    main()