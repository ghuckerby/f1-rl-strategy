from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Set
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

    # Target driver lap data for lap time anchoring
    target_lap_times: List[float] = field(default_factory=list)
    target_pit_laps: Set[int] = field(default_factory=set)
    target_compound_at_lap: List[int] = field(default_factory=list)
    target_tyre_age_at_lap: List[int] = field(default_factory=list)

    # Safety car lap numbers
    sc_laps: Set[int] = field(default_factory=set)

    def get_compound(self, compound_id: int) -> TyreCompound:
        return self.compounds.get(compound_id)

    # Lap time prediction using ML model
    def predict(self, compound_id: int, age: int, current_lap: int) -> float:
        """Raw ML predictor lap-time estimate."""
        df = pd.DataFrame([{
            'LapNumber': current_lap,
            'TyreAge': age,
            'CompoundID': compound_id
        }])
        return float(self.predictor.predict(df[['LapNumber', 'TyreAge', 'CompoundID']])[0])

    # Basic lap time calculation without anchoring
    def calculate_lap_time(self, compound_id: int, age: int, current_lap: int = 1) -> float:
        return self.predict(compound_id, age, current_lap)

    # Anchored lap-time calculation
    def calculate_adjusted_lap_time(
        self,
        agent_compound_id: int,
        agent_tyre_age: int,
        current_lap: int,
        agent_is_pitting: bool,
    ) -> float:
        idx = current_lap - 1
        has_target_data = 0 <= idx < len(self.target_lap_times)
        target_real = self.target_lap_times[idx] if has_target_data else None
        is_sc = current_lap in self.sc_laps
        is_target_pit = current_lap in self.target_pit_laps

        # Case 1: safety-car lap -> use target's real lap time directly (already includes SC effect)
        if is_sc and target_real is not None:
            return target_real
        
        # Case 2: agent pits on same lap as target -> use target's real lap time directly (already includes pit loss)
        if agent_is_pitting and is_target_pit and target_real is not None:
            return target_real
        
        # Case 3: agent pits but target didn't -> use predictor for agent's compound/age (clean lap time). Caller adds pit loss separately.
        if agent_is_pitting and not is_target_pit:
            return self.predict(agent_compound_id, agent_tyre_age, current_lap)

        # Case 4: target pitted but agent stays out -> target's real lap time is inflated by pit stop, so fall back to predictor for agent's compound/age (clean lap time)
        if is_target_pit and not agent_is_pitting:
            return self.predict(agent_compound_id, agent_tyre_age, current_lap)

        # Case 5: normal racing lap -> return target's real lap time adjusted by the delta between agent and target predicted times (accounts for compound/age differences)
        if target_real is not None and has_target_data:
            target_compound = self.target_compound_at_lap[idx]
            target_age = self.target_tyre_age_at_lap[idx]

            predicted_agent = self.predict(agent_compound_id, agent_tyre_age, current_lap)
            predicted_target = self.predict(target_compound, target_age, current_lap)
            delta = predicted_agent - predicted_target
            return target_real + delta

        # Fallback: no target data for this lap → raw predictor
        return self.predict(agent_compound_id, agent_tyre_age, current_lap)
    
    @classmethod
    def from_race_data(cls, race_data: Dict[str, Any], predictor: Optional[Any] = None) -> 'RaceParams':
        track = TrackParams.from_dict(race_data['track'])
        tyre_compounds = race_data.get('tyre_compounds', {})
        compounds = {
            int(k): TyreCompound.from_dict(v) for k, v in tyre_compounds.items()
        }

        # Parse target driver lap times and pit stops for anchoring
        td = race_data.get('target_driver_strategy', {})
        target_lap_times: List[float] = td.get('lap_times', [])
        target_pit_laps_list = [int(l) for l in td.get('pit_laps', [])]
        target_pit_laps: Set[int] = set(target_pit_laps_list)

        starting_compound = td.get('starting_compound', 2)
        pit_compounds: List[int] = td.get('pit_compounds', [starting_compound])

        # Pre-compute compound and tyre age at every lap (1-indexed internally)
        total_laps = track.total_laps
        target_compound_at_lap: List[int] = []
        target_tyre_age_at_lap: List[int] = []
        current_compound = starting_compound
        stint_age = 0
        pit_index = 0  # index into pit_laps_list (sorted)
        sorted_pit_laps = sorted(target_pit_laps_list)

        for lap in range(1, total_laps + 1):
            # Check if target pitted on this lap (compound changes on pit lap)
            if pit_index < len(sorted_pit_laps) and lap == sorted_pit_laps[pit_index]:
                pit_index += 1
                if pit_index < len(pit_compounds):
                    current_compound = pit_compounds[pit_index]
                stint_age = 0

            stint_age += 1
            target_compound_at_lap.append(current_compound)
            target_tyre_age_at_lap.append(stint_age)

        # Parse safety car laps
        sc_laps: Set[int] = set()
        for event in race_data.get('sc_events', []):
            start = int(event['start_lap'])
            end = int(event['end_lap'])
            for lap_num in range(start, end + 1):
                sc_laps.add(lap_num)

        return cls(
            track=track,
            compounds=compounds,
            predictor=predictor,
            target_lap_times=target_lap_times,
            target_pit_laps=target_pit_laps,
            target_compound_at_lap=target_compound_at_lap,
            target_tyre_age_at_lap=target_tyre_age_at_lap,
            sc_laps=sc_laps,
        )