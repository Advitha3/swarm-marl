import numpy as np

class RandomAgent:
    """
    A random agent for a single drone.
    Takes random actions regardless of state.
    This is our baseline - any trained agent must beat this.
    """
    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self, observation):
        """
        Given an observation, return an action.
        For now: completely random.
        Later: this is where our neural network will go.
        """
        return self.action_space.sample()

    def learn(self, obs, action, reward, next_obs, done):
        """
        Update the agent based on experience.
        For now: do nothing (random agent doesn't learn).
        Later: this is where PPO update will go.
        """
        pass