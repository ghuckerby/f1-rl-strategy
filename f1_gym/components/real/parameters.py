from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class TyreCompound:
    compound_id: int
    compound: str
    base_lap_time: float
    deg_rate: float
    avg_stint_length: float
    max_stint_length: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TyreCompound':
        return cls(
            compound_id=data['compound_id'],
            compound=data['compound'],
            base_lap_time=data['base_lap_time'],
            deg_rate=data['deg_rate'],
            avg_stint_length=data['avg_stint_length'],
            max_stint_length=data['max_stint_length']
        )
    
    def calculate_lap_time(self, age: int) -> float:
        return self.base_lap_time + self.deg_rate * (age - 1)

@dataclass
class TrackParams:
    name: str
    total_laps: int
    pit_loss_time: float
    pit_loss_std: float
    fastest_lap: float
    average_lap: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrackParams':
        return cls(
            name=data['name'],
            total_laps=data['total_laps'],
            pit_loss_time=data['pit_loss_time'],
            pit_loss_std=data['pit_loss_std'],
            fastest_lap=data['fastest_lap'],
            average_lap=data['average_lap']
        )

@dataclass
class RaceParams:
    track: TrackParams
    compounds: Dict[int, TyreCompound] = field(default_factory=dict)

    def get_compound(self, compound_id: int) -> TyreCompound:
        return self.compounds.get(compound_id)
    
    def calculate_lap_time(self, compound_id: int, age: int) -> float:
        compound = self.get_compound(compound_id)
        if compound:
            return compound.calculate_lap_time(age)
        return None
    
    @classmethod
    def from_race_data(cls, race_data: Dict[str, Any]) -> 'RaceParams':
        track = TrackParams.from_dict(race_data['track'])
        compounds = {compound['compound_id']: TyreCompound.from_dict(compound) for compound in race_data['compounds']}
        return cls(track=track, compounds=compounds)