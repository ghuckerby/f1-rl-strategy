from dataclasses import dataclass
from typing import List

@dataclass
class RealOpponent:

    driver_code: str
    driver_name: str
    starting_compound: int
    starting_position: int
    pit_laps: List[float]
    pit_compounds: List[int]
    lap_times: List[float]
    finishing_position: int
    dnf: bool
    total_time: float
    num_pit_stops: int = 0