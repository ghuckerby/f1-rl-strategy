import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dynamics import (
    compounds, TrackParams, TyreCompound, calculate_lap_time
)
from typing import List, Dict, Any, Tuple

class F1OpponentEnv(gym.Env):

    def __init__(self, track: TrackParams | None = None, starting_compound: TyreCompound = 1):
        super().__init__()

        self.track = track or TrackParams()
        self.num_opponents = 19
        self.compounds = compounds
        self.starting_compound = starting_compound
        self.num_pit_stops = 0

        # Action Space: 0=Stay Out, 1=Soft, 2=Medium, 3=Hard
        self.action_space = spaces.Discrete(4)

        # Observation Space:
            # lap_fraction
            # compound_one_hot
            # tyre_age_norm
            # tyre_wear_norm

            # position
            # time_to_leader
            # time_to_ahead
        self.obs_size = 1 + 3 + 1 + 1 + 1 + 1 + 1
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.obs_size,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.current_lap = 0
        self.current_compound = self.starting_compound
        self.tyre_wear = 0.0
        self.tyre_age = 0
        self.total_time = 0.0
        self.num_pit_stops = 0
        self.position = 1

        obs = self.make_obs()
        
        return obs, {}
    
    def make_obs(self) -> np.ndarray:
        lap_fraction = self.current_lap / self.track.laps
        compound_one_hot = np.array([1.0 if self.current_compound == c else 0.0 for c in (1, 2, 3)], dtype=np.float32)
        tyre_age_norm = min(self.tyre_age / self.track.laps, 1.0)
        tyre_wear_norm = np.clip(self.tyre_wear, 0.0, 1.0)

        position = 1
        time_to_leader = 0.0
        time_to_ahead = 0.0

        obs = np.concatenate([
            np.array([lap_fraction], dtype=np.float32),
            compound_one_hot,
            np.array([tyre_age_norm, tyre_wear_norm, position, time_to_leader, time_to_ahead], dtype=np.float32)
        ])
        return obs
    
    def step(self, action: int):

        self.update_agent(action)
        self.update_opponents()
        reward = self.calculate_reward()
        terminated = self.current_lap > self.track.laps

        if terminated:
            final_position_reward = (20 - self.position) * 10
            reward += final_position_reward
        
        return self.make_obs(), reward, terminated, False, {}
    
    def update_agent(self, action: int):
        # Update agent state for current lap
        pit_time = 0.0

        if action == 0:
            pass
        
        elif action in (1, 2, 3):
            self.num_pit_stops += 1
            pit_time = self.track.pit_loss
            new_compound = action
            self.current_compound = new_compound
            self.tyre_age = 0

        # Calculate lap time using the compound object
        compound_obj = compounds[self.current_compound]
        lap_time = calculate_lap_time(compound_obj, self.tyre_age) + pit_time
        self.total_time += lap_time
        self.current_lap += 1
        self.tyre_age += 1
        self.tyre_wear = min(self.tyre_age / self.track.laps, 1.0)

    def update_opponents(self):
        # Update all opponent drivers
        # Placeholder: would track opponent pit strategies and positions
        pass
        
        # Update positions based on times
        self.update_positions()

    def update_positions(self):
        # Placeholder: would calculate position based on times vs opponents
        self.position = 1

    def calculate_reward(self) -> float:
        reward = 0.0
        reward = -self.position

        return reward


if __name__ == "__main__":
    # Test the environment
    env = F1OpponentEnv(starting_compound=1)
    obs, info = env.reset()
    
    print("Environment created successfully!")
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    # Run a few random steps
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Reward: {reward:.2f}, Position: {env.position}")
        
        if terminated:
            break