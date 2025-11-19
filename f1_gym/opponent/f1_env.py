import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .dynamics import (
    compounds, TrackParams, TyreCompound, calculate_lap_time
)

class F1OpponentEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.num_opponents = 19
        self.compounds = compounds

        # Action Space: 0=Stay Out, 1=Soft, 2=Medium, 3=Hard
        self.action_space = spaces.Discrete(4)

        # Observation Space:
        self.observation_space = spaces.Box()

        self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        
        return self.make_obs()
    
    def make_obs(self) -> np.ndarray:

        return obs


if __name__ == "__main__":
    # Test the environment
    env = F1OpponentEnv()
    obs, _ = env.reset()
    
    print("Environment created successfully!")
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    # Run a few random steps
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Reward: {reward:.2f}, Position: {env.agent_position}")
        
        if terminated:
            break