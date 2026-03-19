import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, List, Optional

from f1_gym.components.parameters import RaceParams
from f1_gym.components.opponents import RealOpponent
from f1_gym.components.events import RealRaceEvents
from f1_gym.reward_config import RewardConfig

class F1RealEnv(gym.Env):
    """Custom Gym environment for simulating a real F1 race based on historical data, allowing an RL agent to learn optimal pit stop strategies against real opponents and events."""

    # Observation normalisation constants
    MAX_GAP_SECONDS = 60.0
    MAX_POSITION = 20.0
    NUM_COMPOUNDS = 3

    def __init__(self, race_data: Dict[str, Any], reward_config: Optional[RewardConfig] = None, predictor: Optional[Any] = None):
        super().__init__()

        self.race_data = race_data
        self.reward_config = reward_config or RewardConfig()

        # Components from data
        self.params = RaceParams.from_race_data(race_data, predictor=predictor)
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
            opp.total_race_laps = self.total_laps
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
        """Constructs the observation vector for the current state of the race."""

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
        """Constructs the info dictionary for the current state of the race.""" 
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
        """Executes one step of the environment given an action, updating the race state."""
        
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
            final_position_reward = (20 - self.position) * self.reward_config.final_position_weight
            reward += final_position_reward

            # Terminal penalty for not using at least 2 compounds
            if len(self.compounds_used) < 2:
                reward += self.reward_config.terminal_rule_penalty

            info["episode_log"] = list(self.race_log)
            info["final_standings"] = self._build_final_standings()

        return self.make_obs(), reward, terminated, False, info
    
    def calculate_reward(self, action: int, prev_position: int, pitted: bool) -> float:
        """Calculates the reward for the current step based on lap time, position change, pit stops, and progressive rule penalties."""

        config = self.reward_config
        reward = 0.0

        # 1. Lap Time Reward (Scaled with SC multiplier)
        benchmark = self.params.track.average_lap * self.events.get_lap_time_multiplier()
        reward += (benchmark - self.lap_time) * config.lap_time_reward_weight

        # 2. Position Reward
        reward += (prev_position - self.position) * config.position_gain_reward

        # 3. Pit Stop Penalty
        if pitted:
            reward += config.pit_stop_penalty

        # 4. Progressive rule penalty for not using 2 compounds
        if len(self.compounds_used) < 2:
            race_progress = self.current_lap / self.total_laps
            if race_progress >= config.rule_penalty_start_pct:
                penalty_progress = (race_progress - config.rule_penalty_start_pct) / (1.0 - config.rule_penalty_start_pct)
                reward += config.rule_penalty_base * (penalty_progress ** config.rule_penalty_exponent)

        return float(reward)
   
    def update_agent(self, action: int):
        """Updates the agent's state based on the chosen action, including pit stops and lap time calculation."""

        agent_is_pitting = action in (1, 2, 3)
        current_lap_number = self.current_lap + 1

        pit_loss_multiplier = self.events.get_pit_loss_multiplier()

        # Pit Stop — update compound / age BEFORE lap-time calc
        if agent_is_pitting:
            self.num_pit_stops += 1
            self.current_compound = action
            self.compounds_used.add(action)
            self.tyre_age = 0

        # Increment age before calculation so first lap of stint = 1,
        self.tyre_age += 1

        # Calculate lap time anchored to target driver's real pace
        base_lap = self.params.calculate_adjusted_lap_time(
            agent_compound_id=self.current_compound,
            agent_tyre_age=self.tyre_age,
            current_lap=current_lap_number,
            agent_is_pitting=agent_is_pitting,
        )
        if base_lap is None:
            base_lap = self.params.track.average_lap

        # Add pit-loss only when the agent pits on a lap the target did NOT
        # pit (case 3 in calculate_adjusted_lap_time).  When the agent pits
        # on the same lap as the target, the real lap time already includes
        # the pit-stop delay.
        pit_time = 0.0
        if agent_is_pitting:
            same_lap_as_target = current_lap_number in self.params.target_pit_laps
            if not same_lap_as_target:
                pit_time = self.pit_loss * pit_loss_multiplier

        # Lap time and state updates
        self.lap_time = base_lap + pit_time
        self.total_time += self.lap_time
        self.current_lap += 1
        self.tyre_wear = min(self.tyre_age / self.total_laps, 1.0)

    def update_opponents(self):
        """Advances all opponents by one lap, updating their lap times, positions, and compounds based on their individual strategies"""
        for opp in self.opponents:
            opp.step()

    def update_race_standings(self):
        """Updates the race standings based on laps completed then total time"""

        # Agent always completes current_lap laps (full race distance)
        agent_laps = self.current_lap

        # Build entries: (laps_completed, cumulative_time, is_agent, code, dnf)
        entries = []
        for opp in self.opponents:
            if opp.dnf and opp.has_finished:
                continue  # exclude DNFs from active standings
            entries.append((
                opp.laps_completed, opp.cumulative_time, False, opp.driver_code
            ))
        entries.append((agent_laps, self.total_time, True, "AGENT"))
        entries.sort(key=lambda x: (-x[0], x[1]))
        self.race_standings = entries
        self.position = next(i + 1 for i, (_, _, is_agent, _) in enumerate(entries) if is_agent)

    def _build_final_standings(self) -> List[Dict[str, Any]]:
        """Build detailed final standings from simulated race state for display."""
        COMPOUND_SHORT = {1: "S", 2: "M", 3: "H"}
        standings = []

        for opp in self.opponents:
            stints = " -> ".join(COMPOUND_SHORT.get(c, "?") for c in opp.pit_compounds)
            standings.append({
                "code": opp.driver_code,
                "name": opp.driver_name,
                "time": opp.cumulative_time,
                "laps": opp.laps_completed,
                "pit_stops": opp.num_pit_stops,
                "strategy": stints,
                "is_agent": False,
                "dnf": opp.dnf and opp.has_finished,
                "penalty": opp.time_penalty,
            })

        # Agent entry
        stints_agent = []
        if self.race_log:
            stints_agent.append(COMPOUND_SHORT.get(self.race_log[0].get("compound"), "?"))
            for lap in self.race_log:
                if lap.get("pitted"):
                    stints_agent.append(COMPOUND_SHORT.get(lap.get("compound"), "?"))
        standings.append({
            "code": "AGT",
            "name": "Agent",
            "time": self.total_time,
            "laps": self.current_lap,
            "pit_stops": self.num_pit_stops,
            "strategy": " -> ".join(stints_agent),
            "is_agent": True,
            "dnf": False,
            "penalty": 0.0,
        })

        # Sort: DNFs last, then more laps first, then lower time
        standings.sort(key=lambda x: (x["dnf"], -x["laps"], x["time"]))
        return standings
    
    def calculate_time_to_leader(self) -> float:
        """Calculates the time difference between the agent and the race leader."""
        leader_time = self.race_standings[0][1]
        return max(0.0, self.total_time - leader_time)
    
    def calculate_time_to_ahead(self) -> float:
        """Calculates the time difference between the agent and the car immediately ahead in the standings."""
        agent_idx = next(
            (i for i, (_, _, is_agent, _) in enumerate(self.race_standings) if is_agent), 0
        )
        if agent_idx == 0:
            return 0.0
        return max(0.0, self.total_time - self.race_standings[agent_idx - 1][1])
    
    def calculate_time_to_behind(self) -> float:
        """Calculates the time difference between the agent and the car immediately behind in the standings."""
        agent_idx = next(
            (i for i, (_, _, is_agent, _) in enumerate(self.race_standings) if is_agent),
            len(self.race_standings) - 1
        )
        if agent_idx >= len(self.race_standings) - 1:
            return 0.0
        return max(0.0, self.race_standings[agent_idx + 1][1] - self.total_time)
    
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
        