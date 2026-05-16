import numpy as np
from gym_pybullet_drones.envs import HoverAviary
from agents.random_agent import RandomAgent

def run_episode(env, agent, max_steps=500):
    """Run one full episode, return total reward."""
    obs, info = env.reset()
    total_reward = 0

    for step in range(max_steps):
        # Agent picks action based on observation
        action = agent.select_action(obs)

        # Environment steps forward
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Agent learns from experience (random agent ignores this)
        agent.learn(obs, action, reward, next_obs, terminated)

        total_reward += reward
        obs = next_obs

        if terminated or truncated:
            break

    return total_reward, step + 1

def main():
    # Setup
    env = HoverAviary(gui=False)
    agent = RandomAgent(env.action_space)

    print("Training random agent baseline...")
    print(f"{'Episode':>8} {'Reward':>12} {'Steps':>8}")
    print("-" * 32)

    episode_rewards = []

    for episode in range(20):
        reward, steps = run_episode(env, agent)
        episode_rewards.append(float(np.mean(reward)))

        print(f"{episode+1:>8} {float(np.mean(reward)):>12.3f} {steps:>8}")

    print("-" * 32)
    print(f"Mean reward over 20 episodes: {np.mean(episode_rewards):.3f}")
    print("\nThis is your baseline. Your PPO agent must beat this.")

    env.close()

if __name__ == "__main__":
    main()