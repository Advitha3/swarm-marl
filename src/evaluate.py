import torch
import numpy as np
import sys
import time
sys.path.append('src')

from envs.curriculum_env import CurriculumFormationEnv
from agents.policy_network import PolicyNetwork


def evaluate():
    print("Loading best curriculum policy...")

    # Use GUI for visualization
    env = CurriculumFormationEnv(num_drones=4, gui=True)

    # Force to Stage 3 for evaluation
    env.current_stage = 3

    obs_dim = 75
    action_dim = 4

    policy = PolicyNetwork(obs_dim, action_dim, hidden_dim=64)

    try:
        policy.load_state_dict(
            torch.load("results/best_curriculum_policy.pt")
        )
        print("✅ Loaded trained curriculum policy")
    except:
        print("⚠️  No saved policy — using untrained policy")

    policy.eval()

    print("\nWatching 3 episodes...")
    print("Recording tip: use Kazam or OBS to capture screen\n")

    for episode in range(3):
        obs, _ = env.reset()
        episode_reward = 0

        print(f"Episode {episode+1} — Stage: {env.current_stage}")

        for step in range(300):
            obs_tensor = torch.FloatTensor(obs)

            with torch.no_grad():
                actions, _ = policy.get_action(obs_tensor)

            actions_np = actions.numpy()
            obs, reward, terminated, truncated, info = env.step(
                actions_np
            )
            episode_reward += reward

            # Slow down for visualization
            time.sleep(0.02)

            if terminated or truncated:
                print(f"  Ended at step {step+1}")
                break

        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Mean distance to targets: "
              f"{info['mean_distance']:.3f}m")
        time.sleep(1)

    env.close()
    print("\nDone!")


if __name__ == "__main__":
    evaluate()