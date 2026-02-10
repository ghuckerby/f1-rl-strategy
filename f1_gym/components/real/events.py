from dataclasses import dataclass
from typing import List

@dataclass
class SafetyCarEvent:
    start_lap: int
    end_lap: int
    duration: int

@dataclass
class RealRaceEvents:
    sc_events: List[SafetyCarEvent]
    sc_speed_factor: float = 1.4 # Need to update this
    sc_pit_factor: float = 0.5 # Need to update this