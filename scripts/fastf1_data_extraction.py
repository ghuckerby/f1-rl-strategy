
import fastf1 as ff1
from typing import Dict, List, Tuple
from dataclasses import dataclass

ff1.Cache.enable_cache("fastf1_cache/cache")

# Data Classes

@dataclass
class TyreParameters:
    compound: str
    compound_id: int
    base_lap_time: float
    deg_rate: float

@dataclass
class OpponentStrategy:
    driver_code: str
    driver_name: str
    starting_compound: int
    pit_laps: List[int]
    pit_compounds: List[int]
    total_time: float
    finishing_position: int
    num_pit_stops: int
    lap_times: List[float]

@dataclass
class SafetyCarEvent:
    start_lap: int
    end_lap: int
    duration: int

@dataclass
class TrackParameters:
    name: str
    total_laps: int
    pit_loss_time: float
    pit_loss_std: float
    fastest_lap: float
    average_lap: float

# Compound Mapping

COMPOUND_MAP = {
    'SOFT': 1,
    'MEDIUM': 2,
    'HARD': 3,
    'INTERMEDIATE': 4,
    'WET': 5
}

# Main Extractor Class

class FastF1DataExtractor:

    def __init__(self, cache_dir: str = "fastf1_cache/cache"):
        ff1.Cache.enable_cache(cache_dir)
        self.cache_dir = cache_dir

    def load_session(self, year: int, gp: str, session_type: str = 'R') -> ff1.core.Session:
        session = ff1.get_session(year, gp, session_type)
        session.load(laps=True, telemetry=False, weather=True, messages=True)
        return session
    
    def get_tyre_parameters(self, session: ff1.core.Session) -> Dict[int, TyreParameters]:
        return None
    
    def get_pit_loss(self, session: ff1.core.Session) -> Tuple[float, float]:
        return None
    
    def get_safety_car_events(self, session: ff1.core.Session) -> List[SafetyCarEvent]:
        return None
    
    def calculate_sc_probability(self, sc_events: List[SafetyCarEvent], total_laps: int) -> float:
        return None
    
    def get_driver_strategy(self, session: ff1.core.Session, driver_code: str) -> OpponentStrategy:
        return None
    
    def get_tyre_parameters(self, session: ff1.core.Session) -> TrackParameters:
        return None