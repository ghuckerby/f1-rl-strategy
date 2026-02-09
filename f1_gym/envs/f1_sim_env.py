import gymnasium as gym
from gymnasium import spaces
import numpy as np
from f1_gym.components.sim.parameters import compounds, TrackParams, TyreCompound, calculate_lap_time
from f1_gym.components.sim.opponents import (
    Opponent, RandomOpponent, HeuristicOpponent,
    BenchmarkOpponent, HardBenchmarkOpponent, AdaptiveBenchmarkOpponent
)
from f1_gym.components.sim.events import RaceEvents
from f1_gym.reward_config import RewardConfig
from typing import List, Dict, Any, Tuple, Type
import random

class F1OpponentEnv(gym.Env):
    
    # Observation normalisation constants
    MAX_GAP_SECONDS = 60.0
    MAX_POSITION = 20.0
    NUM_COMPOUND_TYPES = 3.0

    def __init__(self, track: TrackParams | None = None, starting_compound: TyreCompound = 1, 
                 opponent_class: Type[Opponent] = AdaptiveBenchmarkOpponent, reward_config: RewardConfig = None):
        
        super().__init__()

        self.reward_config = reward_config or RewardConfig()

        # Environment parameters
        self.track = track or TrackParams()
        self.events = RaceEvents()
        self.num_opponents = 19
        self.compounds = compounds
        self.starting_compound = starting_compound
        self.opponent_class = opponent_class
        self.num_pit_stops = 0
        self.lap_time = 0.0
        self.compounds_used = set()

        # Action Space: 0=Stay Out, 1=Soft, 2=Medium, 3=Hard
        self.action_space = spaces.Discrete(4)

        # Observation Space definition
        # Includes: lap_fraction, compound_one_hot (3), tyre_age_norm, tyre_wear_norm, 
        # num_compounds_norm, position, 3x time gaps (ahead, leader, behind), sc_active
        self.obs_size = 1 + 3 + 1 + 1 + 1 + 1 + 3 + 1
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.obs_size,), dtype=np.float32
        )

        # Race log for storing lap data
        self.race_log: List[Dict[str, Any]] = []

        # Opponents list
        self.opponents: List[Opponent] = []
        
        # Sorted Race Standings
        self.race_standings = []

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
        self.events.reset()
        
        # Initialise opponents
        self.opponents = [
            self.opponent_class(i, self.track, starting_compound=random.choice([1, 2, 3]))
            for i in range(self.num_opponents)
        ]

        # Randomise starting grid positions + add time offset (0.5s per position)
        grid_positions = list(range(self.num_opponents + 1))
        self.np_random.shuffle(grid_positions)
        
        # Assign agent starting position time offset
        agent_grid_pos = grid_positions[0]
        self.total_time = agent_grid_pos * 0.5
        self.position = agent_grid_pos + 1
        
        # Assign opponent starting position time offsets
        for i, opp in enumerate(self.opponents):
            opp_grid_pos = grid_positions[i + 1]
            opp.total_time = opp_grid_pos * 0.5

        self.update_race_standings()
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
            "sc_active": False,
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
        # (Relies on update_race_standings being called)
        time_to_leader = self.calculate_time_to_leader()
        time_to_ahead = self.calculate_time_to_ahead()
        time_to_behind = self.calculate_time_to_behind()

        # Normalise times
        time_to_leader_norm = np.clip(time_to_leader / self.MAX_GAP_SECONDS, 0.0, 1.0)
        time_to_ahead_norm = np.clip(time_to_ahead / self.MAX_GAP_SECONDS, 0.0, 1.0)
        time_to_behind_norm = np.clip(time_to_behind / self.MAX_GAP_SECONDS, 0.0, 1.0)
        position_norm = np.clip(self.position / self.MAX_POSITION, 0.0, 1.0)
        
        # Number of different compounds used so far (normalised)
        num_compounds_norm = len(self.compounds_used) / self.NUM_COMPOUND_TYPES

        # Event flag
        sc_active = 1.0 if self.events.active_event == "safety_car" else 0.0

        obs = np.concatenate([
            np.array([lap_fraction], dtype=np.float32),
            compound_one_hot,
            np.array([tyre_age_norm, tyre_wear_norm, num_compounds_norm, position_norm, 
                      time_to_leader_norm, time_to_ahead_norm, time_to_behind_norm, sc_active], dtype=np.float32)
        ])
        return obs

    def calculate_time_to_behind(self) -> float:
        # Find index of agent
        agent_idx = next(i for i, (_, is_agent, _) in enumerate(self.race_standings) if is_agent)
        if agent_idx == len(self.race_standings) - 1:
            return 0.0
        
        # Next car in list has higher total_time (is behind)
        return max(0.0, self.race_standings[agent_idx + 1][0] - self.total_time)
    
    def calculate_time_to_leader(self) -> float:
        leader_time = self.race_standings[0][0]
        return max(0.0, self.total_time - leader_time)
    
    def calculate_time_to_ahead(self) -> float:
        agent_idx = next(i for i, (_, is_agent, _) in enumerate(self.race_standings) if is_agent)
        if agent_idx == 0:
            return 0.0
        
        # Previous car in list has lower total_time (is ahead)
        return max(0.0, self.total_time - self.race_standings[agent_idx - 1][0])
    
    def step(self, action: int):
        """Perform one step in the environment with the given action"""

        self.events.step()

        # Track previous position for reward shaping
        prev_position = self.position
        pitted = False
        if action in (1, 2, 3):
            pitted = True

        self.update_agent(action)
        self.update_opponents()

        # Calculate reward
        reward = self.calculate_reward(action, prev_position, pitted)

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
            "sc_active": bool(self.events.active_event),
        }
        self.race_log.append(info)

        if terminated:
            # Final position reward
            final_position_reward = (20 - self.position) * 100
            reward += final_position_reward

            info["episode_log"] = list(self.race_log)
        
        return self.make_obs(), reward, terminated, False, info
    
    def calculate_reward(self, action: int, prev_position: int, pitted:bool) -> float:
        config = self.reward_config
        reward = 0.0

        # Lap Time Reward
        reward += (config.time_benchmark - self.lap_time) * config.lap_time_reward_weight

        # Position Reward
        reward += (prev_position - self.position) * config.position_gain_reward

        # Rule Enforcement Penalty (at least 2 compounds used)
        if len(self.compounds_used) < 2:
            if self.current_lap >= self.track.laps:
                reward += config.rule_penalty_violation

        return reward
    
    def update_agent(self, action: int):
        """Update the agent's state based on the action taken"""

        pit_time = 0.0

        lap_speed_multiplier = self.events.get_lap_time_multiplier()
        pit_loss_multiplier = self.events.get_pit_loss_multiplier()

        # Stay out action
        if action == 0:
            pass
        
        # Pit stop action
        elif action in (1, 2, 3):
            self.num_pit_stops += 1
            pit_time = self.track.pit_loss * pit_loss_multiplier
            new_compound = action
            self.current_compound = new_compound
            self.compounds_used.add(new_compound)
            self.tyre_age = 0

        # Lap time calculation and state updates
        compound_obj = compounds[self.current_compound]
        base_lap = calculate_lap_time(compound_obj, self.tyre_age) * lap_speed_multiplier
        self.lap_time = base_lap + pit_time
        self.total_time += self.lap_time
        self.current_lap += 1
        self.tyre_age += 1
        self.tyre_wear = min(self.tyre_age / self.track.laps, 1.0)

    def update_opponents(self):
        """Advance all opponents by one lap and update positions"""

        for opp in self.opponents:
            if opp.current_lap < self.track.laps:
                opp.step(self.events)

        self.update_race_standings()

    def update_race_standings(self):
        """Update the race order and agent position"""
        # Create list of (total_time, is_agent, id)
        times = [(opp.total_time, False, opp.opponent_id) for opp in self.opponents]
        times.append((self.total_time, True, -1))
        
        # Sort by total_time
        times.sort(key=lambda x: x[0])
        self.race_standings = times
        
        # Find agent position
        self.position = next(i + 1 for i, (_, is_agent, _) in enumerate(times) if is_agent)
    
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
        sc = "SC" if row["sc_active"] else "--"
        
        print(
            f"Lap {row['lap']:>2} | {sc} | action: {action:>10} | compound: {compound} | "
            f"tyre_age: {row['tyre_age']:>2} | lap_time: {row['lap_time'] or 0:.2f}s | "
            f"total_time: {row['total_time']:.2f}s | pitted: {row['pitted']} | "
            f"position: {row['position']}"
        )