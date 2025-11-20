
from dataclasses import dataclass
import random
from typing import List, Dict, Any, Tuple

# Tyre Compound Class
@dataclass
class TyreCompound:
    name: str
    base_lap_time: float
    deg_rate: float

# Track Parameter Class
# Laps and pit stop time loss
@dataclass
class TrackParams:
    laps: int = 50
    pit_loss: float = 25.0

# Compound choices
    # 1: Soft, 2: Medium, 3: Hard
    # Base lap times and degradation rates
compounds = {
    1: TyreCompound(
        name="SOFT",
        base_lap_time=90.0,
        deg_rate=0.15,
    ),
    2: TyreCompound(
        name="MEDIUM",
        base_lap_time=91.0,
        deg_rate=0.10,
    ),
    3: TyreCompound(
        name="HARD",
        base_lap_time=92.0,
        deg_rate=0.07,
    )
}

# Current lap time calculation
def calculate_lap_time(compound: TyreCompound, age: int) -> float:
    return compound.base_lap_time + compound.deg_rate * (age - 1)

# Random Opponent Class
# Implements a random pit stop strategy for an opponent
class RandomOpponent:
    def __init__(self, opponent_id: int, track: TrackParams, starting_compound: TyreCompound = 1):
        # Initialize opponent

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
        
    def generate_strategy(self) -> Dict[str, Any]:
        # Generate a random pit stop strategy (1-stop or 2-stop)

        strategy_type = random.choice([1, 2])
        
        # 1-Stop Strategy
        if strategy_type == 1:
            # Choose starting and pit compounds (ensuring they're different)    
            # Random lap for pit stop and compounds
            start_compound = random.choice([1, 2, 3])
            pit_compound = random.choice([c for c in [1, 2, 3] if c != start_compound])
            pit_lap = random.randint(2, self.track.laps - 1)
            
            return {
                "type": 1,
                "pit_laps": [pit_lap],
                "compounds": [start_compound, pit_compound]
            }
        
        # 2-Stop Strategy
        else:
            start_compound = random.choice([1, 2, 3])
            pit1_lap = random.randint(2, self.track.laps - 20)
            pit2_lap = random.randint(pit1_lap + 5, self.track.laps - 1)
            
            pit1_compound = random.choice([c for c in [1, 2, 3] if c != start_compound])
            pit2_compound = random.choice([c for c in [1, 2, 3] if c != pit1_compound])
            
            return {
                "type": 2,
                "pit_laps": [pit1_lap, pit2_lap],
                "compounds": [start_compound, pit1_compound, pit2_compound]
            }
    
    def step(self):
        # Advance one lap
        pit_time = 0.0
        
        if self.current_lap + 1 in self.pit_laps:
            pit_index = self.pit_laps.index(self.current_lap + 1)
            pit_time = self.track.pit_loss
            self.current_compound = self.pit_compounds[pit_index + 1]
            self.tyre_age = 0
            self.num_pit_stops += 1
        
        compound_obj = compounds[self.current_compound]
        self.lap_time = calculate_lap_time(compound_obj, self.tyre_age) + pit_time
        self.total_time += self.lap_time
        self.current_lap += 1
        self.tyre_age += 1
    
    def reset(self):
        # Reset opponent
        self.current_lap = 0
        self.current_compound = self.starting_compound
        self.tyre_age = 0
        self.total_time = 0.0
        self.num_pit_stops = 0
        self.lap_time = 0.0
        self.strategy = self.generate_strategy()
        self.pit_laps = self.strategy["pit_laps"]
        self.pit_compounds = self.strategy["compounds"]