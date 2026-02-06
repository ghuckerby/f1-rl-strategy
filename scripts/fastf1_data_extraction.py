
import fastf1 as ff1
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import os
import pandas as pd

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

@dataclass
class RaceConfig:
    year: int
    name: str
    track: TrackParameters
    tyre_params: Dict[int, TyreParameters]
    sc_events: List[SafetyCarEvent]
    sc_probability: float
    opponents: List[OpponentStrategy]
    target_driver: str
    target_driver_strategy: OpponentStrategy

@dataclass
class SeasonConfig:
    year: int
    target_driver: str
    races: List[RaceConfig]
    train_races: List[str]
    test_races: List[str]

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
        sc_starts = len(sc_events)

        total_sc_laps = sum(e.duration for e in sc_events)
        normal_laps = total_laps - total_sc_laps
        
        return sc_starts / normal_laps if normal_laps > 0 else 0.0
    
    def get_driver_strategy(self, session: ff1.core.Session, driver_code: str) -> OpponentStrategy:
        return None
    
    def get_tyre_parameters(self, session: ff1.core.Session) -> Dict[int, TyreParameters]:
        return None
    
    def get_driver_strategy(self, session: ff1.core.Session, driver_code: str) -> OpponentStrategy:
        return None
    
    def get_track_parameters(self, session: ff1.core.Session) -> TrackParameters:
        laps = session.laps.copy()
        total_laps = int(laps['LapNumber'].max())
        track_name = session.event['EventName']
        valid_laps = laps[
            {laps['IsAccurate'] == True} &
            {~laps['LapTime'].isna()}
        ]

        lap_times = valid_laps['LapTime'].dt.total_seconds()
        fastest_lap = lap_times.min()
        average_lap = lap_times.mean()

        pit_loss, pit_loss_std = self.get_pit_loss(session)

        return TrackParameters(
            name=track_name,
            total_laps=total_laps,
            pit_loss_time=pit_loss,
            pit_loss_std=pit_loss_std,
            fastest_lap=fastest_lap,
            average_lap=average_lap
        )
    
    def get_race_config(self, year: int, gp: str, target_driver: str) -> RaceConfig:
        
        print(f"Loading session: {year} {gp}")
        session = self.load_session(year, gp, 'R')

        print(f"Extracting track parameters")
        track_params = self.get_track_parameters(session)
        
        print(f"Extracting tyre parameters")
        tyre_params = self.get_tyre_parameters(session)

        print(f"Extracting safety car events")
        sc_events = self.get_safety_car_events(session)
        sc_prob = self.calculate_sc_probability(sc_events, track_params.total_laps)

        print(f"Extracting target driver strategies")
        target_strategy = self.get_driver_strategy(session, target_driver)

        all_drivers = session.laps['Driver'].unique()
        opponent_strategies = []
        for driver in all_drivers:
            if driver != target_driver:
                try:
                    opponent_strategies.append(self.get_driver_strategy(session, driver))
                except Exception as e:
                    print(f"Error extracting strategy for driver {driver}: {e}")
                    continue

        opponent_strategies.sort(key=lambda x: x.finishing_position)

        return RaceConfig(
            year=year,
            name=gp,
            track=track_params,
            tyre_params=tyre_params,
            sc_events=sc_events,
            sc_probability=sc_prob,
            opponents=opponent_strategies,
            target_driver=target_driver,
            target_driver_strategy=target_strategy
        )
    
    def get_season_config(
            self, year: int, 
            target_driver: str, 
            test_races: Optional[List[str]] = None,
            exclude_wet_races: bool = True
    ) -> SeasonConfig:
        
        schedule = ff1.get_event_schedule(year)
        race_events = schedule[schedule['EventFormat'] == 'Race']

        all_races = []
        completed_gps = []

        for _, event in race_events.iterrows():
            gp_name = event['EventName']

            if pd.isna(event['Session5Date']):
                continue

            try:
                print(f"Processing: {gp_name}")
                race_config = self.get_race_config(year, gp_name, target_driver)

                if exclude_wet_races:
                    wet_used = any(
                        4 in s.pit_compounds or 5 in s.pit_compounds
                        for s in race_config.opponent_strategies
                    )
                    if wet_used:
                        print(f"Excluding {gp_name} - wet race")
                        continue

                all_races.append(race_config)
                completed_gps.append(gp_name)

            except Exception as e:
                print(f"Error processing {gp_name}: {e}")
                continue
        
        train_races = [gp for gp in completed_gps if gp not in test_races]

        return SeasonConfig(
            year=year,
            target_driver=target_driver,
            races = all_races,
            train_races = train_races,
            test_races = test_races
        )

        return None