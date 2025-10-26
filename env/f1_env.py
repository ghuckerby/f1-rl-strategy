
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .dynamics import (
    SOFT, MEDIUM, HARD, calculate_lap_time, TrackParams
)

class F1PitStopEnv(gym.Env):

    def __init__(self, track: TrackParams | None = None):
        super().__init__()
        self.track = track or TrackParams()

        # Discrete actions: stay out, pit soft, pit med, pit hard
        self.action_space = spaces.Discrete(4)

        # [lap_fraction, compound(3), stint_age_norm, tyre_wear_norm, pit_loss_norm]
        self.obs_size = 1 + 3 + 1 + 1 + 1

        # Observation Space
        self.observation_space = spaces.Box(
            low=0,                      # min value for each element
            high=1,                     # max value for each element                    
            shape=(self.obs_size,),     
            dtype=np.float32
        )
        
        self.current_lap = 0
        self.compound = SOFT
        self.stint_age = 0
        self.tire_wear = 0.0
        self.total_time = 0.0

    def make_obs(self) -> np.ndarray:
        lap_fraction = self.current_lap / self.track.laps
        compound_oh = np.array([1.0 if self.compound == c else 0.0 for c in (SOFT, MEDIUM, HARD)], dtype=np.float32)
        stint_age_norm = min(self.stint_age / self.track.max_stint_age, 1.0)
        tire_wear_norm = np.clip(self.tire_wear, 0.0, 1.0)
        pit_loss_norm = min(self.track.pit_loss / 30.0, 1.0)

        obs = np.concatenate([
            np.array([lap_fraction], dtype=np.float32),
            compound_oh,
            np.array([stint_age_norm, tire_wear_norm, pit_loss_norm], dtype=np.float32)
        ]).astype(np.float32)

        return obs
    
    def apply_pit(self, new_compound: int) -> float:
        self.compound = new_compound
        self.stint_age = 0
        self.tire_wear = 0.0

        return self.track.pit_loss

    # Called at the start of each new episode (race)
    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed, options=options)
        self.current_lap = 0
        self.compound = SOFT
        self.stint_age = 0
        self.tire_wear = 0.0
        self.total_time = 0.0

        return self.make_obs(), {}
    
    # Advances environment by one time step, given an action from agent
    def step(self, action: int):
        pit_time = 0.0
        if action == 1:
            pit_time = self.apply_pit(SOFT) # pit soft 
        elif action == 2:
            pit_time = self.apply_pit(MEDIUM) # pit medium 
        elif action == 3:
            pit_time = self.apply_pit(HARD) # pit hard
        else:
            pass # stay out

        lap_time = calculate_lap_time(self.compound, self.stint_age)
        lap_time += pit_time

        self.total_time += lap_time
        self.current_lap += 1

        self.stint_age += 1
        self.tire_wear = min(self.stint_age / self.track.max_stint_age, 1.0)

        terminated = self.current_lap >= self.track.laps

        reward = -lap_time

        return self.make_obs(), reward, terminated, False, {}
