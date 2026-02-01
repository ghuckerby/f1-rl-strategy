
from f1_gym.components.parameters import TrackParams, TyreCompound, calculate_lap_time, compounds
from f1_gym.components.events import RaceEvents
import random
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from abc import ABC, abstractmethod

# Abstract Opponent Class
class Opponent(ABC):
    def __init__(self, opponent_id: int, track: TrackParams, starting_compound: TyreCompound = 1):
        """Initialize Opponent with ID, track parameters, and starting compound"""

        self.opponent_id = opponent_id
        self.track = track
        self.starting_compound = starting_compound
        
        self.current_lap = 0
        self.current_compound = starting_compound
        self.tyre_age = 0
        self.total_time = 0.0
        self.num_pit_stops = 0
        self.lap_time = 0.0
        
        self.strategy = self.generate_strategy()
        self.pit_laps = self.strategy["pit_laps"]
        self.pit_compounds = self.strategy["compounds"]

    @abstractmethod
    def step(self):
        """Advance one lap"""
        pass

    @abstractmethod
    def reset(self):
        """Reset opponent to initial state"""
        pass

# Random Opponent Class
# Generates a random 1 or 2-stop strategy
class RandomOpponent(Opponent):
    def __init__(self, opponent_id: int, track: TrackParams, starting_compound: TyreCompound = 1):
        """Initialize Random Opponent with ID, track parameters, and starting compound"""
        super().__init__(opponent_id, track, starting_compound)
        
    def generate_strategy(self) -> Dict[str, Any]:
        """Generate a random pit stop strategy (1-stop or 2-stop)"""

        start_compound = self.starting_compound
        strategy_type = random.choice([1, 2])
        
        # 1-Stop Strategy
        if strategy_type == 1:
            # Choose pit compound (ensuring it's different from start)
            pit_compound = random.choice([c for c in [1, 2, 3] if c != start_compound])
            max_pit_lap = max(2, self.track.laps - 1)
            pit_lap = random.randint(2, max_pit_lap)
            
            return {
                "type": 1,
                "pit_laps": [pit_lap],
                "compounds": [start_compound, pit_compound]
            }
        
        # 2-Stop Strategy
        else:
            pit1_lap = random.randint(2, self.track.laps - 20)
            pit2_lap = random.randint(pit1_lap + 5, self.track.laps - 1)
            
            pit1_compound = random.choice([c for c in [1, 2, 3] if c != start_compound])
            pit2_compound = random.choice([c for c in [1, 2, 3] if c != pit1_compound])
            
            return {
                "type": 2,
                "pit_laps": [pit1_lap, pit2_lap],
                "compounds": [start_compound, pit1_compound, pit2_compound]
            }
    
    def step(self, events: RaceEvents):
        """Advance one lap"""   
        pit_time = 0.0

        lap_speed_multiplier = events.get_lap_time_multiplier()
        pit_loss_multiplier = events.get_pit_loss_multiplier()
        
        if self.current_lap + 1 in self.pit_laps:
            pit_index = self.pit_laps.index(self.current_lap + 1)
            base_loss = self.track.pit_loss * pit_loss_multiplier
            pit_delay = events.get_pit_delay()
            pit_time = base_loss + pit_delay
            self.current_compound = self.pit_compounds[pit_index + 1]
            self.tyre_age = 0
            self.num_pit_stops += 1
        
        compound_obj = compounds[self.current_compound]
        base_lap = calculate_lap_time(compound_obj, self.tyre_age) * lap_speed_multiplier
        self.lap_time = base_lap + pit_time
        self.total_time += self.lap_time
        self.current_lap += 1
        self.tyre_age += 1
    
    def reset(self):
        """Reset opponent to initial state"""

        self.current_lap = 0
        self.current_compound = self.starting_compound
        self.tyre_age = 0
        self.total_time = 0.0
        self.num_pit_stops = 0
        self.lap_time = 0.0
        self.strategy = self.generate_strategy()
        self.pit_laps = self.strategy["pit_laps"]
        self.pit_compounds = self.strategy["compounds"]

# Heuristic Opponent Class
# Generates a simple heuristic-based 1-stop strategy (pit between 40% and 60% of race)
class HeuristicOpponent(Opponent):
    def __init__(self, opponent_id: int, track: TrackParams, starting_compound: TyreCompound = 1):
        super().__init__(opponent_id, track, starting_compound)

    def generate_strategy(self) -> Dict[str, Any]:
        pit_lap = int(self.track.laps * random.uniform(0.4, 0.6))
        pit_compound = 2 if self.starting_compound == 1 else 1
        return {
            "type": 1,
            "pit_laps": [pit_lap],
            "compounds": [self.starting_compound, pit_compound]
        }
    
    def step(self, events: RaceEvents):
        pit_time = 0.0
        lap_speed_multiplier = events.get_lap_time_multiplier()
        pit_loss_multiplier = events.get_pit_loss_multiplier()

        if self.current_lap + 1 in self.pit_laps:
            pit_index = self.pit_laps.index(self.current_lap + 1)
            base_loss = self.track.pit_loss * pit_loss_multiplier
            pit_delay = events.get_pit_delay()
            pit_time = base_loss + pit_delay
            self.current_compound = self.pit_compounds[pit_index + 1]
            self.tyre_age = 0
            self.num_pit_stops += 1
        
        compound_obj = compounds[self.current_compound]
        base_lap = calculate_lap_time(compound_obj, self.tyre_age) * lap_speed_multiplier
        self.lap_time = base_lap + pit_time
        self.total_time += self.lap_time
        self.current_lap += 1
        self.tyre_age += 1

    def reset(self):
        self.current_lap = 0
        self.current_compound = self.starting_compound
        self.tyre_age = 0
        self.total_time = 0.0
        self.num_pit_stops = 0
        self.lap_time = 0.0
        self.strategy = self.generate_strategy()
        self.pit_laps = self.strategy["pit_laps"]
        self.pit_compounds = self.strategy["compounds"]

# Time Benchmark Opponent
# Generates predefined optimal strategies from benchmarking based on starting compound
class BenchmarkOpponent(Opponent):
    STRATEGIES = {
        1: [  # Soft Start Options
            # 1-Stop: Soft -> Medium
            {"type": 1, "pit_laps": [25], "compounds": [1, 2]},
            {"type": 1, "pit_laps": [24], "compounds": [1, 2]},
            {"type": 1, "pit_laps": [26], "compounds": [1, 2]},
            {"type": 1, "pit_laps": [23], "compounds": [1, 2]},
            {"type": 1, "pit_laps": [27], "compounds": [1, 2]},
            # 2-Stop: Soft -> Medium -> Soft
            {"type": 2, "pit_laps": [17, 33], "compounds": [1, 2, 1]},
            {"type": 2, "pit_laps": [18, 33], "compounds": [1, 2, 1]},
            {"type": 2, "pit_laps": [18, 34], "compounds": [1, 2, 1]},
            {"type": 2, "pit_laps": [17, 34], "compounds": [1, 2, 1]},
            {"type": 2, "pit_laps": [17, 32], "compounds": [1, 2, 1]},
        ],
        2: [  # Medium Start Options
            # 1-Stop: Medium -> Soft
            {"type": 1, "pit_laps": [26], "compounds": [2, 1]},
            {"type": 1, "pit_laps": [27], "compounds": [2, 1]},
            {"type": 1, "pit_laps": [25], "compounds": [2, 1]},
            {"type": 1, "pit_laps": [28], "compounds": [2, 1]},
            # 2-Stop: Medium -> Soft -> Medium
            {"type": 2, "pit_laps": [16, 34], "compounds": [2, 1, 2]},
            {"type": 2, "pit_laps": [17, 35], "compounds": [2, 1, 2]},
            {"type": 2, "pit_laps": [16, 35], "compounds": [2, 1, 2]},
        ],
        3: [  # Hard Start Options
            # 1-Stop: Hard -> Soft
            {"type": 1, "pit_laps": [25], "compounds": [3, 1]},
            {"type": 1, "pit_laps": [26], "compounds": [3, 1]},
            # 2-Stop: Mixed variations
            {"type": 2, "pit_laps": [12, 31], "compounds": [3, 2, 1]},
            {"type": 2, "pit_laps": [12, 32], "compounds": [3, 1, 2]},
        ]
    }

    def __init__(self, opponent_id: int, track: TrackParams, starting_compound: TyreCompound = 1):
        super().__init__(opponent_id, track, starting_compound)

    def generate_strategy(self) -> Dict[str, Any]:
        options = self.STRATEGIES[self.starting_compound]
        return random.choice(options)
    
    def step(self, events: RaceEvents):
        pit_time = 0.0
        lap_speed_multiplier = events.get_lap_time_multiplier()
        pit_loss_multiplier = events.get_pit_loss_multiplier()

        if self.current_lap + 1 in self.pit_laps:
            pit_index = self.pit_laps.index(self.current_lap + 1)
            base_loss = self.track.pit_loss * pit_loss_multiplier
            pit_delay = events.get_pit_delay()
            pit_time = base_loss + pit_delay
            self.current_compound = self.pit_compounds[pit_index + 1]
            self.tyre_age = 0
            self.num_pit_stops += 1
        
        compound_obj = compounds[self.current_compound]
        base_lap = calculate_lap_time(compound_obj, self.tyre_age) * lap_speed_multiplier
        self.lap_time = base_lap + pit_time
        self.total_time += self.lap_time
        self.current_lap += 1
        self.tyre_age += 1

    def reset(self):
        self.current_lap = 0
        self.current_compound = self.starting_compound
        self.tyre_age = 0
        self.total_time = 0.0
        self.num_pit_stops = 0
        self.lap_time = 0.0
        self.strategy = self.generate_strategy()
        self.pit_laps = self.strategy["pit_laps"]
        self.pit_compounds = self.strategy["compounds"]

# Hard Benchmark Opponent
# Generates from the best optimal benchmark strategies
class HardBenchmarkOpponent(Opponent):
    STRATEGIES = {
        1: [
            {"type": 2, "pit_laps": [17, 33], "compounds": [1, 2, 1]},
            {"type": 2, "pit_laps": [18, 33], "compounds": [1, 2, 1]},
            {"type": 2, "pit_laps": [18, 34], "compounds": [1, 2, 1]},
            {"type": 2, "pit_laps": [17, 34], "compounds": [1, 2, 1]},
            {"type": 2, "pit_laps": [17, 32], "compounds": [1, 2, 1]},
        ]
    }

    def __init__(self, opponent_id: int, track: TrackParams, starting_compound: TyreCompound = 1):
        super().__init__(opponent_id, track, starting_compound)

    def generate_strategy(self) -> Dict[str, Any]:
        options = self.STRATEGIES[self.starting_compound]
        return random.choice(options)
    
    def step(self, events: RaceEvents):
        pit_time = 0.0
        lap_speed_multiplier = events.get_lap_time_multiplier()
        pit_loss_multiplier = events.get_pit_loss_multiplier()

        if self.current_lap + 1 in self.pit_laps:
            pit_index = self.pit_laps.index(self.current_lap + 1)
            base_loss = self.track.pit_loss * pit_loss_multiplier
            pit_delay = events.get_pit_delay()
            pit_time = base_loss + pit_delay
            self.current_compound = self.pit_compounds[pit_index + 1]
            self.tyre_age = 0
            self.num_pit_stops += 1
        
        compound_obj = compounds[self.current_compound]
        base_lap = calculate_lap_time(compound_obj, self.tyre_age) * lap_speed_multiplier
        self.lap_time = base_lap + pit_time
        self.total_time += self.lap_time
        self.current_lap += 1
        self.tyre_age += 1

    def reset(self):
        self.current_lap = 0
        self.current_compound = self.starting_compound
        self.tyre_age = 0
        self.total_time = 0.0
        self.num_pit_stops = 0
        self.lap_time = 0.0
        self.strategy = self.generate_strategy()
        self.pit_laps = self.strategy["pit_laps"]
        self.pit_compounds = self.strategy["compounds"]

# Adapative Opponent
# Uses predefined optimal strategies and adapts based on race conditions
class AdaptiveBenchmarkOpponent(Opponent):
    STRATEGIES = {
        1: [
            # 1-Stop: Soft -> Medium
            {"type": 1, "pit_laps": [25], "compounds": [1, 2]},
            {"type": 1, "pit_laps": [24], "compounds": [1, 2]},
            {"type": 1, "pit_laps": [26], "compounds": [1, 2]},
            {"type": 1, "pit_laps": [23], "compounds": [1, 2]},
            {"type": 1, "pit_laps": [27], "compounds": [1, 2]},
            # 2-Stop: Soft -> Medium -> Soft
            {"type": 2, "pit_laps": [17, 33], "compounds": [1, 2, 1]},
            {"type": 2, "pit_laps": [18, 33], "compounds": [1, 2, 1]},
            {"type": 2, "pit_laps": [18, 34], "compounds": [1, 2, 1]},
            {"type": 2, "pit_laps": [17, 34], "compounds": [1, 2, 1]},
            {"type": 2, "pit_laps": [17, 32], "compounds": [1, 2, 1]},
        ],
        2: [
            # 1-Stop: Medium -> Soft
            {"type": 1, "pit_laps": [26], "compounds": [2, 1]},
            {"type": 1, "pit_laps": [27], "compounds": [2, 1]},
            {"type": 1, "pit_laps": [25], "compounds": [2, 1]},
            {"type": 1, "pit_laps": [28], "compounds": [2, 1]},
            # 2-Stop: Medium -> Soft -> Medium
            {"type": 2, "pit_laps": [16, 34], "compounds": [2, 1, 2]},
            {"type": 2, "pit_laps": [17, 35], "compounds": [2, 1, 2]},
            {"type": 2, "pit_laps": [16, 35], "compounds": [2, 1, 2]},
        ],
        3: [
            # 1-Stop: Hard -> Soft
            {"type": 1, "pit_laps": [25], "compounds": [3, 1]},
            {"type": 1, "pit_laps": [26], "compounds": [3, 1]},
            # 2-Stop: Mixed variations
            {"type": 2, "pit_laps": [12, 31], "compounds": [3, 2, 1]},
            {"type": 2, "pit_laps": [12, 32], "compounds": [3, 1, 2]},
        ]
    }

    def __init__(self, opponent_id: int, track: TrackParams, starting_compound: TyreCompound = 1):
        super().__init__(opponent_id, track, starting_compound)
        self.has_reacted = False

    def generate_strategy(self) -> Dict[str, Any]:
        options = self.STRATEGIES[self.starting_compound]
        return random.choice(options)
    
    def sc_pit_decision(self, events: RaceEvents) -> bool:

        # No pit if:
            # Already reacted to this SC
            # Exceed planned stops
            # Early or late
            # Fresh tyres
        if self.has_reacted:
            return False
        if self.num_pit_stops >= self.strategy["type"]:
            return False
        if self.current_lap < 5 or self.current_lap > self.track.laps - 5:
            return False
        if self.tyre_age < 8:
            return False
        
        # Pit if:
            # Safety car is close to planned pit stop
            # Tyres are old
        next_stop = None
        for pit_lap in self.pit_laps:
            if pit_lap > self.current_lap:
                next_stop = pit_lap
                break
        if next_stop and abs(next_stop - self.current_lap) <= 10:
            return True
        if self.tyre_age > 15:
            return True
        
        return False
    
    def get_next_compound(self) -> int:
        if self.num_pit_stops < len(self.pit_compounds) - 1:
            return self.pit_compounds[self.num_pit_stops + 1]
        
        # Fallback
        if self.current_compound == 1:
            return 2
        else:
            return 1
    
    def step(self, events: RaceEvents):
        pit_time = 0.0
        lap_speed_multiplier = events.get_lap_time_multiplier()
        pit_loss_multiplier = events.get_pit_loss_multiplier()
        
        is_safety_car = events.active_event == "safety_car"
        should_pit = False
        
        # Reset SC react flag
        if not is_safety_car:
            self.has_reacted = False
        
        # Pit Schedule
        if self.current_lap + 1 in self.pit_laps:
            should_pit = True
        # Safety Car Opportunistic Pit
        elif is_safety_car and self.sc_pit_decision(events):
            should_pit = True
            self.has_reacted = True
            # Update pit_laps to remove next planned stop
            planned_stops = [p for p in self.pit_laps if p > self.current_lap]
            if planned_stops:
                self.pit_laps.remove(planned_stops[0])
        
        if should_pit:
            base_loss = self.track.pit_loss * pit_loss_multiplier
            pit_delay = events.get_pit_delay()
            pit_time = base_loss + pit_delay
            self.current_compound = self.get_next_compound()
            self.tyre_age = 0
            self.num_pit_stops += 1
        
        compound_obj = compounds[self.current_compound]
        base_lap = calculate_lap_time(compound_obj, self.tyre_age) * lap_speed_multiplier
        self.lap_time = base_lap + pit_time
        self.total_time += self.lap_time
        self.current_lap += 1
        self.tyre_age += 1

    def reset(self):
        self.current_lap = 0
        self.current_compound = self.starting_compound
        self.tyre_age = 0
        self.total_time = 0.0
        self.num_pit_stops = 0
        self.lap_time = 0.0
        self.has_reacted = False
        self.strategy = self.generate_strategy()
        self.pit_laps = self.strategy["pit_laps"].copy()
        self.pit_compounds = self.strategy["compounds"]