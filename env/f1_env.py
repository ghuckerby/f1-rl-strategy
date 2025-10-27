
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .dynamics import (
    SOFT, MEDIUM, HARD, calculate_lap_time, TrackParams
)

from typing import List, Dict, Any

class F1PitStopEnv(gym.Env):

    def __init__(self, track: TrackParams | None = None):
        super().__init__()

        self.track = track or TrackParams()
        self.action_space = spaces.Discrete(4) # 0 = stay_out, 1=S, 2=M, 3=H
        # [lap_fraction, compound(3), stint_age_norm, tyre_wear_norm, pit_loss_norm]
        self.obs_size = 1 + 3 + 1 + 1 + 1

        # Observation Space
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.obs_size,), dtype=np.float32)
        
        self.current_lap = 0
        self.compound = SOFT
        self.stint_age = 0
        self.tire_wear = 0.0
        self.total_time = 0.0

        self.race_log: List[Dict[str, Any]] = []

    def make_obs(self) -> np.ndarray:
        lap_fraction = self.current_lap / self.track.laps
        # one hot encoding for compound
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

    # Called at the start of each new episode (race)
    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed, options=options)

        self.current_lap = 0
        self.compound = SOFT
        self.stint_age = 0
        self.tire_wear = 0.0
        self.total_time = 0.0
        obs = self.make_obs()

        self.race_log = []
        info = {
            "lap": self.current_lap,
            "compound": int(self.compound),
            "stint_age": int(self.stint_age),
            "tire_wear": float(self.tire_wear),
            "total_time": float(self.total_time),
            "pitted": False,
            "action": None,
            "lap_time": None, 
        }

        return obs, info
    
    # Advances environment by one time step, given an action from agent
    def step(self, action: int):

        pitted = False
        pit_time = 0.0
        if action in (1, 2, 3):
            new_compound = {1: SOFT, 2: MEDIUM, 3: HARD}[action]
            if new_compound != self.compound:
                pitted = True
                pit_time = self.track.pit_loss
                self.compound = new_compound
                self.stint_age = 0

        # lap time dynamics
        lap_time = calculate_lap_time(self.compound, self.stint_age) + pit_time
        self.total_time += lap_time
        self.current_lap += 1
        self.stint_age += 1
        self.tire_wear = min(self.stint_age / self.track.max_stint_age, 1.0)
        terminated = self.current_lap >= self.track.laps
        reward = -lap_time

        # info for tracking
        info = {
            "lap": int(self.current_lap),
            "compound": int(self.compound),
            "stint_age": int(self.stint_age),
            "tire_wear": float(self.tire_wear),
            "total_time": float(self.total_time),
            "pitted": bool(pitted),
            "action": int(action),
            "lap_time": float(lap_time),
        }
        self.race_log.append(info)

        # if race ends, add full log
        if terminated:
            info["episode_log"] = list(self.race_log)

        return self.make_obs(), reward, terminated, False, info
