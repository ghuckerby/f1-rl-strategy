from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import sys
import os
import pandas as pd

@dataclass
class TyreCompound:
    compound_id: int
    compound: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TyreCompound':
        return cls(
            compound_id=data['compound_id'],
            compound=data['compound'],
        )

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
    predictor: Optional[Any] = None

    def get_compound(self, compound_id: int) -> TyreCompound:
        return self.compounds.get(compound_id)
    
    def calculate_lap_time(self, compound_id: int, age: int, current_lap: int = 1) -> float:
        df = pd.DataFrame([{
            'LapNumber': current_lap,
            'TyreAge': age,
            'CompoundID': compound_id
        }])

        return float(self.predictor.predict(df[['LapNumber', 'TyreAge', 'CompoundID']])[0])
    
    @classmethod
    def from_race_data(cls, race_data: Dict[str, Any], predictor: Optional[Any] = None) -> 'RaceParams':
        track = TrackParams.from_dict(race_data['track'])
        tyre_compounds = race_data.get('tyre_compounds', {})
        compounds = {
            int(k): TyreCompound.from_dict(v) for k, v in tyre_compounds.items()
        }
        return cls(track=track, compounds=compounds, predictor=predictor)