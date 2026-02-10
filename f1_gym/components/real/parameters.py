from dataclasses import dataclass
from typing import Dict

@dataclass
class TyreCompound:
    compound_id: int
    compound: str
    base_lap_time: float
    deg_rate: float
    avg_stint_length: float
    max_stint_length: float

@dataclass
class TrackParams:
    name: str
    total_laps: int
    pit_loss_time: float
    pit_loss_std: float
    fastest_lap: float
    average_lap: float

@dataclass
class RaceParams:
    track: TrackParams
    compounds: Dict[int, TyreCompound]