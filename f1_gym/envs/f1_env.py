import gymnasium as gym
from gymnasium import spaces
import numpy as np
from f1_gym.components.tracks import compounds, TrackParams, TyreCompound, calculate_lap_time
from f1_gym.components.opponents import Opponent, RandomOpponent
from typing import List, Dict, Any, Tuple, Type
import random

class F1OpponentEnv(gym.Env):

    def __init__(self, track: TrackParams | None = None, starting_compound: TyreCompound = 1, 
                 opponent_class: Type[Opponent] = RandomOpponent):
        
        super().__init__()

        # Environment parameters
        self.track = track or TrackParams()
        self.num_opponents = 19
        self.compounds = compounds
        self.starting_compound = starting_compound
        self.opponent_class = opponent_class
        self.num_pit_stops = 0
        self.lap_time = 0.0
        self.compounds_used = set()

        # Action Space: 0=Stay Out, 1=Soft, 2=Medium, 3=Hard
        self.action_space = spaces.Discrete(4)

        # Observation Space:
            # lap_fraction
            # compound_one_hot
            # tyre_age_norm
            # tyre_wear_norm

            # num_compounds_used (normalized)
            # position
            # time_to_leader
            # time_to_ahead
            # time_to_behind
        self.obs_size = 1 + 3 + 1 + 1 + 1 + 1 + 1 + 1 + 1  # Updated to 11
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.obs_size,), dtype=np.float32
        )

        # Race log for storing lap data
        self.race_log: List[Dict[str, Any]] = []

        # Opponents list
        self.opponents: List[Opponent] = []

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
        self.lap_time = 0.0
        self.compounds_used = {self.starting_compound}  # Track starting compound
        
        # Initialize opponents
        self.opponents = [
            self.opponent_class(i, self.track, starting_compound=random.choice([1, 2, 3]))
            for i in range(self.num_opponents)
        ]

        obs = self.make_obs()

        self.race_log = []
        info = {
            "lap": self.current_lap,
            "compound": int(self.current_compound),
            "tyre_age": int(self.tyre_age),
            "tyre_wear": float(self.tyre_wear),
            "total_time": float(self.total_time),
            "pitted": False,
            "action": None,
            "lap_time": None,
            "position": self.position,
        }
        self.race_log.append(info)
        
        return obs, info
    
    def make_obs(self) -> np.ndarray:
        """Construct observation vector for the current state"""

        # Construct observations (normalisation and one-hot encoding)
        lap_fraction = self.current_lap / self.track.laps
        compound_one_hot = np.array([1.0 if self.current_compound == c else 0.0 for c in (1, 2, 3)], dtype=np.float32)
        tyre_age_norm = min(self.tyre_age / self.track.laps, 1.0)
        tyre_wear_norm = np.clip(self.tyre_wear, 0.0, 1.0)

        # Calculate position and time gaps
        position = self.position
        time_to_leader = self.calculate_time_to_leader()
        time_to_ahead = self.calculate_time_to_ahead()
        time_to_behind = self.calculate_time_to_behind()

        # Normalise times (assuming max gap is 10 seconds)
        time_to_leader_norm = np.clip(time_to_leader / 10.0, 0.0, 1.0)
        time_to_ahead_norm = np.clip(time_to_ahead / 10.0, 0.0, 1.0)
        time_to_behind_norm = np.clip(time_to_behind / 10.0, 0.0, 1.0)
        position_norm = np.clip(position / 20.0, 0.0, 1.0)
        
        # Number of different compounds used so far (normalised to 0-1)
        num_compounds_norm = len(self.compounds_used) / 3.0

        obs = np.concatenate([
            np.array([lap_fraction], dtype=np.float32),
            compound_one_hot,
            np.array([tyre_age_norm, tyre_wear_norm, num_compounds_norm, position_norm, time_to_leader_norm, time_to_ahead_norm, time_to_behind_norm], dtype=np.float32)
        ])
        return obs

    def calculate_time_to_behind(self) -> float:
        """Calculate time gap to the car behind"""

        # Create list of (total_time, is_agent)
        times = [(opp.total_time, opp.opponent_id) for opp in self.opponents]
        times.append((self.total_time, -1))
        times.sort()

        agent_id = next(i for i, (_, id) in enumerate(times) if id == -1)

        if agent_id == len(times) - 1:
            return 0.0
        
        # Returns the time difference to the car behind (or 0 if last)
        return max(0.0, times[agent_id + 1][0] - self.total_time)
    
    def calculate_time_to_leader(self) -> float:
        """Calculate time gap to the race leader"""

        # Find the minimum total time among all cars ahead
        min_time = self.total_time
        for opp in self.opponents:
            if opp.current_lap >= self.current_lap:
                min_time = min(min_time, opp.total_time)
        
        return max(0.0, self.total_time - min_time)
    
    def calculate_time_to_ahead(self) -> float:
        """Calculate time gap to the car ahead"""

        # Create list of (total_time, is_agent)
        times = [(opp.total_time, opp.opponent_id) for opp in self.opponents]
        times.append((self.total_time, -1))
        times.sort()
        
        agent_id = next(i for i, (_, id) in enumerate(times) if id == -1)
        
        if agent_id == 0:
            return 0.0
        
        # Returns the time difference to the car ahead (or 0 if first)
        return max(0.0, self.total_time - times[agent_id - 1][0])
    
    def step(self, action: int):
        """Perform one step in the environment with the given action"""

        # Track previous position for reward shaping
        prev_position = self.position
        pitted = False
        if action in (1, 2, 3):
            pitted = True

        self.update_agent(action)
        self.update_opponents()

        # Lap time reward
        reward = -self.lap_time

        # Position change reward
        position_gain_weight = 10.0
        reward += (prev_position - self.position) * position_gain_weight

        # Pit stop incentive (reward for pitting in strategic window)
        pit_reward = 0.0
        if pitted and 15 <= self.current_lap <= 40:
            pit_reward = 30.0
            # Bonus if pitting to a different compound
            if action != self.current_compound:
                pit_reward += 100.0
        reward += pit_reward

        # Penalty for excessive tyre age/wear
        tyre_penalty = 0.0
        if self.tyre_age > 30 or self.tyre_wear > 0.8:
            # Strong penalty for not pitting on old tyres
            tyre_penalty = -50.0
        reward += tyre_penalty

        # F1 compound rule enforcement: use at least 2 different compounds
        compound_rule_penalty = 0.0
        laps_remaining = self.track.laps - self.current_lap
        
        # Progressive penalty if approaching end of race with only 1 compound
        if len(self.compounds_used) < 2:
            # Progressive penalty as race goes on
            if laps_remaining < 10:
                compound_rule_penalty = -500.0
            elif laps_remaining < 20:
                compound_rule_penalty = -200.0
        
        # Final large penalty for rule violation
        if self.current_lap >= self.track.laps and len(self.compounds_used) < 2:
            compound_rule_penalty = -3000.0
        
        reward += compound_rule_penalty

        terminated = self.current_lap >= self.track.laps

        info = {
            "lap": int(self.current_lap),
            "compound": int(self.current_compound),
            "tyre_age": int(self.tyre_age),
            "tyre_wear": float(self.tyre_wear),
            "total_time": float(self.total_time),
            "pitted": bool(pitted),
            "action": int(action),
            "lap_time": float(self.lap_time),
            "position": self.position,
        }
        self.race_log.append(info)

        if terminated:
            # Final position reward
            final_position_reward = (20 - self.position) * 100
            reward += final_position_reward

            info["episode_log"] = list(self.race_log)
        
        return self.make_obs(), reward, terminated, False, info
    
    def update_agent(self, action: int):
        """Update the agent's state based on the action taken"""

        pit_time = 0.0

        # Stay out action
        if action == 0:
            pass
        
        # Pit stop action
        elif action in (1, 2, 3):
            self.num_pit_stops += 1
            pit_time = self.track.pit_loss
            new_compound = action
            self.current_compound = new_compound
            self.compounds_used.add(new_compound)
            self.tyre_age = 0

        # Lap time calculation and state updates
        compound_obj = compounds[self.current_compound]
        self.lap_time = calculate_lap_time(compound_obj, self.tyre_age) + pit_time
        self.total_time += self.lap_time
        self.current_lap += 1
        self.tyre_age += 1
        self.tyre_wear = min(self.tyre_age / self.track.laps, 1.0)

    def update_opponents(self):
        """Advance all opponents by one lap and update positions"""

        for opp in self.opponents:
            if opp.current_lap < self.track.laps:
                opp.step()

        self.update_positions()

    def update_positions(self):
        """Update the agent's position based on total times"""

        # Create list of (total_time, is_agent)
        times = [(opp.total_time, False) for opp in self.opponents]
        times.append((self.total_time, True))
        
        # Sort by time (ascending - faster times are better)
        times.sort(key=lambda x: x[0])
        
        # Find agent position
        self.position = next(i + 1 for i, (_, is_agent) in enumerate(times) if is_agent)
    
    def logger_output(self):
        """Print the lap information to the console"""

        if not self.race_log:
            print("No laps completed yet")
            return
        
        row = self.race_log[-1]
        compound_names = {1: "S", 2: "M", 3: "H"}
        action_names = {0: "STAY_OUT", 1: "BOX_SOFT", 2: "BOX_MED", 3: "BOX_HARD"}
        
        compound = compound_names.get(row["compound"], "UNKNOWN")
        action = action_names.get(row["action"], "UNKNOWN") if row["action"] is not None else "INITIAL"
        
        print(
            f"Lap {row['lap']:>2} | action: {action:>10} | compound: {compound} | "
            f"tyre_age: {row['tyre_age']:>2} | lap_time: {row['lap_time'] or 0:.2f}s | "
            f"total_time: {row['total_time']:.2f}s | pitted: {row['pitted']} | "
            f"position: {row['position']}"
        )