import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.curriculum_env import CurriculumFormationEnv
from agents.policy_network import PolicyNetwork, CentralizedCritic
from agents.mappo_agent import MAPPOAgent


def collect_experience(env, agent, min_steps=2048):
    obs_list, global_obs_list, act_list = [], [], []
    log_prob_list, reward_list, done_list = [], [], []
    episode_rewards, episode_distances = [], []
    total_steps = 0

    while total_steps < min_steps:
        obs, _ = env.reset()
        episode_reward = 0
        episode_dist = []

        for step in range(500):
            global_obs = obs.flatten()
            actions, log_probs = agent.select_actions(obs)
            next_obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated

            obs_list.append(obs.copy())
            global_obs_list.append(global_obs)
            act_list.append(actions.copy())
            log_prob_list.append(log_probs.copy())
            reward_list.append(float(reward))
            done_list.append(done)

            episode_reward += float(reward)
            if 'mean_distance' in info:
                episode_dist.append(info['mean_distance'])

            obs = next_obs
            total_steps += 1

            if done:
                break

        episode_rewards.append(episode_reward)
        if episode_dist:
            episode_distances.append(np.mean(episode_dist))

    return (obs_list, global_obs_list, act_list,
            log_prob_list, reward_list, done_list,
            episode_rewards, episode_distances)


def main():
    device = torch.device("cpu")
    print("MAPPO with Curriculum Learning — 4 Drone Formation")
    print("=" * 55)

    num_drones = 4
    env = CurriculumFormationEnv(num_drones=num_drones, gui=False)

    obs_dim = 75
    action_dim = 4
    global_obs_dim = obs_dim * num_drones  # 300

    policy = PolicyNetwork(obs_dim, action_dim, hidden_dim=64)
    critic = CentralizedCritic(global_obs_dim, hidden_dim=128)

    # Initialize policy to near-zero outputs for stable hover
    nn_last = policy.network[-2]
    torch.nn.init.uniform_(nn_last.weight, -0.01, 0.01)
    torch.nn.init.zeros_(nn_last.bias)

    agent = MAPPOAgent(
        policy=policy,
        critic=critic,
        num_drones=num_drones,
        lr_policy=1e-4,
        lr_critic=1e-4,
        gamma=0.99,
        clip_eps=0.2,
        entropy_coef=0.01,
        ppo_epochs=4
    )

    print(f"\n{'Update':>8} {'Reward':>12} {'MeanDist':>10} "
          f"{'Stage':>6} {'Episodes':>10} {'Steps':>8}")
    print("-" * 60)

    best_reward = -float('inf')

    for update in range(500):
        (obs_list, global_obs_list, act_list,
         log_prob_list, reward_list, done_list,
         ep_rewards, ep_distances) = collect_experience(
            env, agent, min_steps=2048
        )

        returns = agent.compute_returns(reward_list, done_list)

        policy_loss, critic_loss = agent.update(
            obs_list, global_obs_list, act_list,
            log_prob_list, returns
        )

        mean_ep_reward = np.mean(ep_rewards)
        mean_dist = np.mean(ep_distances) if ep_distances else 0
        num_episodes = len(ep_rewards)
        total_steps = len(reward_list)
        stage = env.current_stage

        # Check if we should advance curriculum stage
        env.try_advance_stage(mean_dist)

        if mean_ep_reward > best_reward:
            best_reward = mean_ep_reward
            torch.save(policy.state_dict(),
                      "results/best_curriculum_policy.pt")

        print(f"{update+1:>8} {mean_ep_reward:>12.3f} {mean_dist:>10.3f} "
              f"{stage:>6} {num_episodes:>10} {total_steps:>8}")

    print("-" * 60)
    print(f"Best reward: {best_reward:.3f}")
    print(f"Final stage: {env.current_stage} — "
          f"{env.STAGES[env.current_stage]['name']}")
    env.close()


if __name__ == "__main__":
    main()