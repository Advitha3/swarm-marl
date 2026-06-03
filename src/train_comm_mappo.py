import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.curriculum_env import CurriculumFormationEnv
from agents.comm_policy_network import CommPolicyNetwork, CommCentralizedCritic
from agents.mappo_agent import MAPPOAgent


def collect_experience_with_comm(env, policy, device, min_steps=2048):
    """
    Collect experience with inter-drone communication.
    
    Key difference from standard collection:
    - Drones maintain message buffers from previous timestep
    - Each drone receives messages from all other drones
    - Messages stored alongside observations for training
    """
    # Experience storage
    obs_list = []           # (T, 4, 75)
    global_obs_list = []    # (T, 300)
    act_list = []           # (T, 4, 4)
    log_prob_list = []      # (T, 4)
    reward_list = []        # (T,)
    done_list = []          # (T,)
    msg_list = []           # (T, 4, 16) — messages sent this step
    recv_msg_list = []      # (T, 4, 48) — messages received this step

    episode_rewards = []
    episode_distances = []
    total_steps = 0

    num_drones = 4
    message_dim = 16

    while total_steps < min_steps:
        obs, _ = env.reset()
        episode_reward = 0
        episode_dist = []

        # Initialize message buffer — zeros at episode start
        # Shape: (num_drones, message_dim)
        # Each drone starts with empty messages
        prev_messages = np.zeros((num_drones, message_dim))

        for step in range(500):
            obs_tensor = torch.FloatTensor(obs)  # (4, 75)

            # Build received messages for each drone
            # Drone i receives messages from all j ≠ i
            received_messages = np.zeros(
                (num_drones, (num_drones - 1) * message_dim)
            )  # (4, 48)

            for i in range(num_drones):
                # Collect messages from all other drones
                others = [j for j in range(num_drones) if j != i]
                recv = np.concatenate(
                    [prev_messages[j] for j in others]
                )  # 3 × 16 = 48
                received_messages[i] = recv

            recv_tensor = torch.FloatTensor(received_messages)  # (4, 48)

            with torch.no_grad():
                # Get actions AND new messages
                actions, log_probs, new_messages = policy.get_action(
                    obs_tensor, recv_tensor
                )

            actions_np = actions.numpy()      # (4, 4)
            new_messages_np = new_messages.numpy()  # (4, 16)

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(
                actions_np
            )
            done = terminated or truncated

            # Store experience
            obs_list.append(obs.copy())
            global_obs_list.append(obs.flatten())   # (300,)
            act_list.append(actions_np.copy())
            log_prob_list.append(log_probs.numpy().copy())
            reward_list.append(float(reward))
            done_list.append(done)
            msg_list.append(new_messages_np.copy())
            recv_msg_list.append(received_messages.copy())

            episode_reward += float(reward)
            if 'mean_distance' in info:
                episode_dist.append(info['mean_distance'])

            # Update for next step
            obs = next_obs
            prev_messages = new_messages_np  # ← delayed communication
            total_steps += 1

            if done:
                break

        episode_rewards.append(episode_reward)
        if episode_dist:
            episode_distances.append(np.mean(episode_dist))

    return (obs_list, global_obs_list, act_list, log_prob_list,
            reward_list, done_list, msg_list, recv_msg_list,
            episode_rewards, episode_distances)


def compute_returns(rewards, dones, gamma=0.99):
    """Discounted returns — same as before."""
    returns = []
    R = 0
    for reward, done in zip(reversed(rewards), reversed(dones)):
        if done:
            R = 0
        R = reward + gamma * R
        returns.insert(0, R)
    return torch.FloatTensor(returns)


def update_with_comm(policy, critic, policy_optimizer, critic_optimizer,
                     obs_list, global_obs_list, act_list, old_log_probs_list,
                     msg_list, recv_msg_list, returns,
                     num_drones=4, clip_eps=0.2, entropy_coef=0.01,
                     ppo_epochs=4):
    """
    MAPPO update with communication.
    
    Same as standard MAPPO but:
    - Policy evaluation includes received messages
    - Critic evaluation includes all messages
    """
    # Convert to tensors
    obs_tensor = torch.FloatTensor(np.array(obs_list))          # (T, 4, 75)
    global_obs_tensor = torch.FloatTensor(np.array(global_obs_list))  # (T, 300)
    act_tensor = torch.FloatTensor(np.array(act_list))          # (T, 4, 4)
    old_log_probs = torch.FloatTensor(np.array(old_log_probs_list))   # (T, 4)
    msg_tensor = torch.FloatTensor(np.array(msg_list))          # (T, 4, 16)
    recv_msg_tensor = torch.FloatTensor(np.array(recv_msg_list))      # (T, 4, 48)

    # Normalize returns
    if returns.std() > 1e-8:
        returns_norm = (returns - returns.mean()) / (returns.std() + 1e-8)
    else:
        returns_norm = returns - returns.mean()

    # All messages concatenated for critic
    # (T, 4, 16) → (T, 64)
    all_messages_flat = msg_tensor.view(msg_tensor.shape[0], -1)

    # Compute values with no_grad
    with torch.no_grad():
        values = critic(global_obs_tensor, all_messages_flat).squeeze()

    # Advantages
    advantages = (returns_norm - values).unsqueeze(1).expand(
        -1, num_drones
    )
    advantages = (advantages - advantages.mean()) / (
        advantages.std() + 1e-8
    )

    for epoch in range(ppo_epochs):
        # Flatten for policy evaluation
        # (T, 4, 75) → (T*4, 75)
        obs_flat = obs_tensor.view(-1, obs_tensor.shape[-1])
        act_flat = act_tensor.view(-1, act_tensor.shape[-1])
        recv_flat = recv_msg_tensor.view(-1, recv_msg_tensor.shape[-1])

        # Evaluate actions with current policy
        new_log_probs, entropy = policy.evaluate_actions(
            obs_flat, recv_flat, act_flat
        )
        new_log_probs = new_log_probs.view(-1, num_drones)

        # PPO ratio
        ratio = (new_log_probs - old_log_probs).exp()

        # Clipped objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio, 1 - clip_eps, 1 + clip_eps
        ) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        policy_loss = policy_loss - entropy_coef * entropy

        # Update policy
        policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        policy_optimizer.step()

        # Update critic
        current_values = critic(
            global_obs_tensor, all_messages_flat
        ).squeeze()
        critic_loss = torch.nn.functional.mse_loss(
            current_values, returns_norm
        )

        critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
        critic_optimizer.step()

    return policy_loss.item(), critic_loss.item()


def main():
    print("MAPPO with Communication — 4 Drone Formation")
    print("=" * 50)

    num_drones = 4
    obs_dim = 75
    action_dim = 4
    message_dim = 16

    env = CurriculumFormationEnv(num_drones=num_drones, gui=False)

    # Communication policy — larger input than standard
    policy = CommPolicyNetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        message_dim=message_dim,
        num_agents=num_drones,
        hidden_dim=64
    )

    # Communication critic — sees obs + messages
    critic = CommCentralizedCritic(
        obs_dim=obs_dim,
        message_dim=message_dim,
        num_agents=num_drones,
        hidden_dim=128
    )

    print(f"Policy parameters:  {sum(p.numel() for p in policy.parameters())}")
    print(f"Critic parameters:  {sum(p.numel() for p in critic.parameters())}")
    print(f"Policy input dim:   {policy.combined_dim}")
    print(f"Critic input dim:   {critic.global_dim}")

    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4)

    print(f"\n{'Update':>8} {'Reward':>12} {'MeanDist':>10} "
          f"{'Stage':>6} {'Episodes':>10}")
    print("-" * 55)

    best_reward = -float('inf')

    for update in range(500):
        # Collect experience with communication
        (obs_list, global_obs_list, act_list, log_prob_list,
         reward_list, done_list, msg_list, recv_msg_list,
         ep_rewards, ep_distances) = collect_experience_with_comm(
            env, policy, device=None, min_steps=2048
        )

        returns = compute_returns(reward_list, done_list)

        policy_loss, critic_loss = update_with_comm(
            policy, critic,
            policy_optimizer, critic_optimizer,
            obs_list, global_obs_list, act_list, log_prob_list,
            msg_list, recv_msg_list, returns
        )

        mean_ep_reward = np.mean(ep_rewards)
        mean_dist = np.mean(ep_distances) if ep_distances else 0
        stage = env.current_stage

        env.try_advance_stage(mean_dist)

        if mean_ep_reward > best_reward:
            best_reward = mean_ep_reward
            torch.save(
                policy.state_dict(),
                "results/best_comm_policy.pt"
            )

        print(f"{update+1:>8} {mean_ep_reward:>12.3f} "
              f"{mean_dist:>10.3f} {stage:>6} "
              f"{len(ep_rewards):>10}")

    print("-" * 55)
    print(f"Best reward: {best_reward:.3f}")
    env.close()


if __name__ == "__main__":
    main()