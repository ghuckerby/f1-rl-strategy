
from f1_gym.components.tracks import TrackParams, TyreCompound, calculate_lap_time, compounds
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
            # Ensure valid range
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