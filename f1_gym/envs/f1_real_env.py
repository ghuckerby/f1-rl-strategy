import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, List, Optional

from f1_gym.components.real.parameters import RaceParams
from f1_gym.components.real.opponents import RealOpponent
from f1_gym.components.real.events import RealRaceEvents
from f1_gym.reward_config import RewardConfig

class F1RealEnv(gym.Env):

    # Observation normalisation constants
    MAX_GAP_SECONDS = 60.0
    MAX_POSITION = 20.0
    NUM_COMPOUNDS = 3

    def __init__(self, race_data: Dict[str, Any], reward_config: Optional[RewardConfig] = None):
        super().__init__()

        self.race_data = race_data
        self.reward_config = reward_config or RewardConfig()

        # Components from data
        self.params = RaceParams.from_race_data(race_data)
        self.events = RealRaceEvents.from_race_data(race_data)

        # Target driver (agent replaces this driver)
        self.target_driver_data = race_data.get('target_driver_strategy', {})
        self.starting_compound = self.target_driver_data.get('starting_compound', 0)
        self.target_start_position = self.target_driver_data.get('starting_position', 10)

        # Track parameters
        self.total_laps = self.params.track.total_laps
        self.pit_loss = self.params.track.pit_loss_time

        # Action Space: 0=Stay Out, 1=Soft, 2=Medium, 3=Hard
        self.action_space = spaces.Discrete(4)

        # Observation Space definition
        # Includes: lap_fraction, compound_one_hot (3), tyre_age_norm, tyre_wear_norm, 
        # num_compounds_norm, position, 3x time gaps (ahead, leader, behind), sc_active
        self.obs_size = 1 + 3 + 1 + 1 + 1 + 1 + 3 + 1
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.obs_size,), dtype=np.float32
        )

        # Race state
        self.current_lap = 0
        self.current_compound = self.starting_compound
        self.tyre_age = 0
        self.tyre_wear = 0.0
        self.total_time = 0.0
        self.num_pit_stops = 0
        self.position = 1
        self.lap_time = 0.0
        self.compounds_used = set()

        # Opponents list
        self.opponents: List[RealOpponent] = []

        # Race standings for position tracking
        self.race_standings: List[tuple] = []

        # Race log for storing lap data
        self.race_log: List[Dict[str, Any]] = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        # Reset race state
        self.current_lap = 0
        self.current_compound = self.starting_compound
        self.tyre_age = 0
        self.tyre_wear = 0.0
        self.total_time = 0.0
        self.num_pit_stops = 0
        self.lap_time = 0.0
        self.compounds_used = {self.starting_compound}

        # Reset events
        self.events.reset()

        # Load opponents from race data
        opponents_data = self.race_data.get('opponents', [])
        self.opponents = [RealOpponent.from_dict(opp) for opp in opponents_data]
        for opp in self.opponents:
            opp.reset()

        # Set agent's starting position
        self.position = self.target_start_position

        self.update_race_standings()
        obs = self.make_obs()
        self.race_log = []
        info = self.make_info(pitted = False, action = None)
        self.race_log.append(info)

        return obs, info

    def make_obs(self) -> np.ndarray:

        # Create observations (normalisation and one-hot encoding)
        lap_fraction = self.current_lap / self.total_laps
        compound_one_hot = np.array(
            [1.0 if self.current_compound == c else 0.0 for c in (1, 2, 3)],
            dtype=np.float32
        )
        tyre_age_norm = min(self.tyre_age / self.total_laps, 1.0)
        tyre_wear_norm = np.clip(self.tyre_wear, 0.0, 1.0)
        num_compounds_norm = len(self.compounds_used) / self.NUM_COMPOUNDS

        # Position and time gaps
        time_to_leader = self.calculate_time_to_leader()
        time_to_ahead = self.calculate_time_to_ahead()
        time_to_behind = self.calculate_time_to_behind()

        time_to_leader_norm = np.clip(time_to_leader / self.MAX_GAP_SECONDS, 0.0, 1.0)
        time_to_ahead_norm = np.clip(time_to_ahead / self.MAX_GAP_SECONDS, 0.0, 1.0)
        time_to_behind_norm = np.clip(time_to_behind / self.MAX_GAP_SECONDS, 0.0, 1.0)
        position_norm = np.clip(self.position / self.MAX_POSITION, 0.0, 1.0)

        sc_active = 1.0 if self.events.safety_car_active else 0.0

        obs = np.concatenate([
            np.array([lap_fraction], dtype=np.float32),
            compound_one_hot,
            np.array([
                tyre_age_norm,
                tyre_wear_norm,
                num_compounds_norm,
                position_norm,
                time_to_leader_norm,
                time_to_ahead_norm,
                time_to_behind_norm,
                sc_active
            ], dtype=np.float32)
        ])
        return obs
    
    def make_info(self, pitted: bool, action: Optional[int]) -> Dict[str, Any]:
        return {
            "lap": int(self.current_lap),
            "compound": int(self.current_compound),
            "tyre_age": int(self.tyre_age),
            "tyre_wear": float(self.tyre_wear),
            "total_time": float(self.total_time),
            "pitted": bool(pitted),
            "action": int(action) if action is not None else None,
            "lap_time": float(self.lap_time) if self.lap_time else None,
            "position": int(self.position),
            "sc_active": bool(self.events.safety_car_active),
        }
    
    def step(self, action: int):
        
        self.events.step()
        prev_position = self.position
        pitted = action in (1, 2, 3)

        # Update agent, opponents and race standings
        self.update_agent(action)
        self.update_opponents()
        self.update_race_standings()

        # Calculate Reward
        reward = self.calculate_reward(action, prev_position, pitted)
        terminated = self.current_lap >= self.total_laps
        info = self.make_info(pitted=pitted, action=action)
        self.race_log.append(info)

        if terminated:
            # Final position reward
            final_position_reward = (20 - self.position) * 100
            reward += final_position_reward
            info["episode_log"] = list(self.race_log)

        return self.make_obs(), reward, terminated, False, info
    
    def calculate_reward(self, action: int, prev_position: int, pitted: bool) -> float:
        config = self.reward_config
        reward = 0.0

        # Lap Time Reward
        benchmark = self.params.track.average_lap
        reward += (benchmark - self.lap_time) * config.lap_time_reward_weight

        # Position Reward
        reward += (prev_position - self.position) * config.position_gain_reward

        # Rule Enforcement
        if len(self.compounds_used) < 2 and self.current_lap >= self.total_laps:
            reward += config.rule_penalty_violation

        return reward
   
    def update_agent(self, action: int):
        pit_time = 0.0

        lap_speed_multiplier = self.events.get_lap_time_multiplier()
        pit_loss_multiplier = self.events.get_pit_loss_multiplier()

        # Pit Stop
        if action in (1, 2, 3):
            self.num_pit_stops += 1
            pit_time = self.pit_loss * pit_loss_multiplier
            self.current_compound = action
            self.compounds_used.add(action)
            self.tyre_age = 0

        # Calculate lap time
        base_lap = self.params.calculate_lap_time(
            self.current_compound, 
            self.tyre_age, 
            self.current_lap + 1
        )
        if base_lap is None:
             base_lap = self.params.track.average_lap
        
        base_lap *= lap_speed_multiplier
        
        # Lap time and state updates
        self.lap_time = base_lap + pit_time
        self.total_time += self.lap_time
        self.current_lap += 1
        self.tyre_age += 1
        self.tyre_wear = min(self.tyre_age / self.total_laps, 1.0)

    def update_opponents(self):
        for opp in self.opponents:
            opp.step()

    def update_race_standings(self):

        # List of (total_time, is_agent, driver_code)
        times = [
            (opp.cumulative_time, False, opp.driver_code) for opp in self.opponents
            if not (opp.has_finished and opp.dnf)
        ]
        times.append((self.total_time, True, "AGENT"))

        # Sort by cumulative time
        times.sort(key=lambda x: x[0])
        self.race_standings = times

        # Find agent position
        self.position = next(i + 1 for i, (_, is_agent, _) in enumerate(times) if is_agent)
    
    def calculate_time_to_leader(self) -> float:
        leader_time = self.race_standings[0][0]
        return max(0.0, self.total_time - leader_time)
    
    def calculate_time_to_ahead(self) -> float:
        agent_idx = next(
            (i for i, (_, is_agent, _) in enumerate(self.race_standings) if is_agent), 0
        )
        if agent_idx == 0:
            return 0.0
        return max(0.0, self.total_time - self.race_standings[agent_idx - 1][0])
    
    def calculate_time_to_behind(self) -> float:
        agent_idx = next(
            (i for i, (_, is_agent, _) in enumerate(self.race_standings) if is_agent),
            len(self.race_standings) - 1
        )
        if agent_idx >= len(self.race_standings) - 1:
            return 0.0
        return max(0.0, self.race_standings[agent_idx + 1][0] - self.total_time)
    
    def logger_output(self):
        if not self.race_log:
            print("No laps completed yet")
            return
        
        row = self.race_log[-1]
        compound_names = {1: "S", 2: "M", 3: "H"}
        action_names = {0: "STAY_OUT", 1: "BOX_SOFT", 2: "BOX_MED", 3: "BOX_HARD"}
        
        compound = compound_names.get(row["compound"], "?")
        action = action_names.get(row["action"], "UNKNOWN") if row["action"] is not None else "INITIAL"
        sc = "SC" if row["sc_active"] else "--"
        
        print(
            f"Lap {row['lap']:>2} | {sc} | action: {action:>10} | compound: {compound} | "
            f"tyre_age: {row['tyre_age']:>2} | lap_time: {row['lap_time'] or 0:.2f}s | "
            f"total_time: {row['total_time']:.2f}s | pitted: {row['pitted']} | "
            f"position: {row['position']}"
        )
        