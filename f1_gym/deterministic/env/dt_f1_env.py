
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .dt_dynamics import (
    SOFT, MEDIUM, HARD, calculate_lap_time, TrackParams
)

from typing import List, Dict, Any

class F1PitStopEnv(gym.Env):

    def __init__(self, track: TrackParams | None = None, starting_compound: int = SOFT):
        super().__init__()

        self.track = track or TrackParams()
        self.starting_compound = starting_compound

        # # Number of sets allowed for each tyre
        #     # Used to enforce compound limit rule (teams only have 1 or 2 of each tyre)
        self.allowed_tyres = {c: 2 for c in (SOFT, MEDIUM, HARD)}
        self.allowed_tyres[self.starting_compound] -= 1

        self.action_space = spaces.Discrete(4) # 4 Actions: 0 = stay_out, 1=S, 2=M, 3=H
        # Observations: [lap_fraction, compound(3), stint_age_norm, tyre_wear_norm, pit_loss_norm]
        self.obs_size = 1 + 3 + 1 + 1 + 1

        # Observation Space
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.obs_size,), dtype=np.float32)

        self.current_lap = 0
        self.compound = SOFT
        self.stint_age = 0
        self.tyre_wear = 0.0
        self.total_time = 0.0

        self.pit_stops = 0
        self.max_pit_stops = 2
        self.compounds_used = set()

        # Race log for strategy tracking and visualisation
        self.race_log: List[Dict[str, Any]] = []
    
    # Observation function
    def make_obs(self) -> np.ndarray:
        lap_fraction = self.current_lap / self.track.laps
        # one hot encoding for compound
        compound_oh = np.array([1.0 if self.compound == c else 0.0 for c in (SOFT, MEDIUM, HARD)], dtype=np.float32)
        stint_age_norm = min(self.stint_age / self.track.max_stint_age, 1.0)
        tyre_wear_norm = np.clip(self.tyre_wear, 0.0, 1.0)
        pit_loss_norm = min(self.track.pit_loss / 30.0, 1.0)

        obs = np.concatenate([
            np.array([lap_fraction], dtype=np.float32),
            compound_oh,
            np.array([stint_age_norm, tyre_wear_norm, pit_loss_norm], dtype=np.float32)
        ]).astype(np.float32)

        return obs

    # Called at the start of each new episode (race)
    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed, options=options)

        # Compound rule update
        self.allowed_tyres = {c: 2 for c in (SOFT, MEDIUM, HARD)}
        self.allowed_tyres[self.starting_compound] -= 1

        # Observations
        self.current_lap = 0
        self.compound = self.starting_compound
        self.stint_age = 0
        self.tyre_wear = 0.0
        self.total_time = 0.0

        self.pit_stops = 0
        self.compounds_used = {self.starting_compound}

        obs = self.make_obs()

        # Logging
        self.race_log = []
        info = {
            "lap": self.current_lap,
            "compound": int(self.compound),
            "stint_age": int(self.stint_age),
            "tyre_wear": float(self.tyre_wear),
            "total_time": float(self.total_time),
            "pitted": False,
            "action": None,
            "lap_time": None, 
        }
        self.race_log.append(info)

        return obs, info
    
    # Advances environment by one time step, given an action from agent
    def step(self, action: int):

        pitted = False
        pit_time = 0.0
        reward_shaping = 0.0
        original_action = action

        # Validation of pit stop action
        # is_valid = True
        # new_compound = None
        # if action in (1, 2, 3):
        #     new_compound = {1: SOFT, 2: MEDIUM, 3: HARD}[action]
        #     if self.pit_stops >= self.max_pit_stops:
        #         is_valid = False
        #     elif self.allowed_tyres[new_compound] <= 0:
        #         is_valid = False
            
        #     if not is_valid:
        #         action = 0
                # reward_shaping -= 5.0
                
        # Action 0: Stay out, no pit penalty
        if action == 0:
            pass

        # Action 1, 2, 3: Pitting
        elif action in (1, 2, 3):
            self.pit_stops += 1
            pitted = True
            pit_time = self.track.pit_loss
            new_compound = {1: SOFT, 2: MEDIUM, 3: HARD}[action]

            self.compounds_used.add(new_compound)
            # self.allowed_tyres[new_compound] -= 1

            self.compound = new_compound
            self.stint_age = 0

        # lap time dynamics
        lap_time = calculate_lap_time(self.compound, self.stint_age) + pit_time
        self.total_time += lap_time
        self.current_lap += 1
        self.stint_age += 1
        self.tyre_wear = min(self.stint_age / self.track.max_stint_age, 1.0)

        terminated = self.current_lap >= self.track.laps

        reward = -lap_time + reward_shaping # reward is negative lap time plus shaping

        # Logging
        info = {
            "lap": int(self.current_lap),
            "compound": int(self.compound),
            "stint_age": int(self.stint_age),
            "tyre_wear": float(self.tyre_wear),
            "total_time": float(self.total_time),
            "pitted": bool(pitted),
            "action": int(action),
            "lap_time": float(lap_time),
        }
        self.race_log.append(info)

        # if race ends, check rules and add full log
        if terminated:
            compounds_used = set(log['compound'] for log in self.race_log)
            if len(compounds_used) < 2:
                # Big penalty due to rule break
                reward -= 100.0

            info["episode_log"] = list(self.race_log)

        return self.make_obs(), reward, terminated, False, info
    
    # Logging Function for seeing used strategy
        # Creates a per-lap output for actions and observations
    def loggeroutput(self):

        if not self.race_log:
            print ("No laps yet")
            return
        
        row = self.race_log[-1]
        compound = {0: "S", 1: "M", 2:"H"}[row["compound"]]
        action = {0: "stay_out", 1: "box_soft", 2:"box_medium", 3:"box_hard"}[row["action"]]
        print(
            f"Lap {row["lap"]:>2} | action: {action:>6} | compound: {compound} | "
            f"stint_age: {row['stint_age']:>2} | lap_time: {row['lap_time']:.2f}s | "
            f"total_time: {row['total_time']:.2f}s | pitted: {row['pitted']}"
        )