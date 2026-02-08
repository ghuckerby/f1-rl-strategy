
import fastf1 as ff1
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import os
import pandas as pd
import numpy as np

ff1.Cache.enable_cache("fastf1_cache/cache")

# Issues:
# - Degradation rate is too simplistic (linear regression)
# - Some tracks have excessive DNFS (not taking into account finishing a lap behind leader)

# Data Classes
@dataclass
class TyreParameters:
    compound: str 
    compound_id: int
    base_lap_time: float
    deg_rate: float
    avg_stint_length: float
    max_stint_length: float
    sample_count: int

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
    dnf: bool

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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "year": self.year,
            "name": self.name,
            "track": asdict(self.track),
            "tyre_params": {k: asdict(v) for k, v in self.tyre_params.items()},
            "sc_events": [asdict(e) for e in self.sc_events],
            "sc_probability": self.sc_probability,
            "opponents": [asdict(o) for o in self.opponents],
            "target_driver": self.target_driver,
            "target_driver_strategy": asdict(self.target_driver_strategy)
        }

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

def map_compound(compound_str: str) -> int:
    return COMPOUND_MAP.get(compound_str.upper(), -1)

# Main Extractor Class
class FastF1DataExtractor:

    def __init__(self, cache_dir: str = "fastf1_cache/cache"):
        # Enable FastF1 caching
        ff1.Cache.enable_cache(cache_dir)
        self.cache_dir = cache_dir

    def load_session(self, year: int, gp: str, session_type: str = 'R') -> ff1.core.Session:
        # Load session and data
        session = ff1.get_session(year, gp, session_type)
        session.load(laps=True, telemetry=False, weather=True, messages=True)
        return session
    
    def get_tyre_parameters(self, session: ff1.core.Session) -> Dict[int, TyreParameters]:
        
        laps = session.laps.copy()

        # Filter invalid laps (pit and SC laps)
        valid_laps = laps[
            (laps['IsAccurate'] == True) &
            (laps['PitInTime'].isna()) &
            (laps['PitOutTime'].isna()) &
            (~laps['LapTime'].isna())
        ].copy()
        valid_laps['LaptimeSec'] = valid_laps['LapTime'].dt.total_seconds()

        tyre_params = {}
        for compound_str in ['SOFT', 'MEDIUM', 'HARD']:
            compound_id = COMPOUND_MAP[compound_str]
            compound_laps = valid_laps[valid_laps['Compound'] == compound_str].copy()

            if len(compound_laps) == 0:
                continue

            # Tyre Age
            compound_laps = compound_laps.sort_values(['Driver', 'LapNumber'])
            compound_laps['TyreAge'] = compound_laps.groupby(['Driver', 'Stint']).cumcount() + 1

            if len(compound_laps) < 5:
                continue

            # Fit linear model: time = base + deg_rate * age
            X = compound_laps['TyreAge'].values
            y = compound_laps['LaptimeSec'].values

            n = len(X)
            sum_x = np.sum(X)
            sum_y = np.sum(y)
            sum_xx = np.sum(X * X)
            sum_xy = np.sum(X * y)

            denom = n * sum_xx - sum_x * sum_x
            if abs(denom) < 1e-10:
                continue
            else:
                deg_rate = (n * sum_xy - sum_x * sum_y) / denom
                base_lap_time = (sum_y - deg_rate * sum_x) / n

            # Deg rate
            deg_rate = max(0.01, min(0.5, deg_rate))

            # Stint statistics
            stint_lengths = compound_laps.groupby(['Driver', 'Stint']).size()
            avg_stint_length = float(stint_lengths.mean()) if len(stint_lengths) > 0 else 0
            max_stint_length = float(stint_lengths.max()) if len(stint_lengths) > 0 else 0

            tyre_params[compound_id] = TyreParameters(
                compound=compound_str,
                compound_id=compound_id,
                base_lap_time=base_lap_time,
                deg_rate=deg_rate,
                avg_stint_length=avg_stint_length,
                max_stint_length=max_stint_length,
                sample_count = len(compound_laps)
            )

        return tyre_params
    
    def get_pit_loss(self, session: ff1.core.Session) -> Tuple[float, float]:
        
        laps = session.laps.copy()

        # Get normal laps (exclude pits, SC, invalid laps)
        normal_laps = laps[
            (laps['IsAccurate'] == True) &
            (laps['PitInTime'].isna()) &
            (laps['PitOutTime'].isna()) &
            (~laps['LapTime'].isna())
        ].copy()
        normal_laps['LaptimeSec'] = normal_laps['LapTime'].dt.total_seconds()

        # Median for imputation of lap times
        median_lap = normal_laps['LaptimeSec'].median()

        # Pit stop duration
        pit_losses = []
        pit_in_laps = laps[~laps['PitInTime'].isna()].copy()
        for _, lap in pit_in_laps.iterrows():
            driver = lap['Driver']
            lap_num = lap['LapNumber']
            
            if pd.isna(lap['LapTime']):
                continue

            # Calculate pit loss
            in_lap_time = lap['LapTime'].total_seconds()

            # Skip outliers (caused by issues or crashes)
            if in_lap_time > 3 * median_lap:
                continue

            driver_normal_laps = normal_laps[normal_laps['Driver'] == driver]
            driver_median_lap = driver_normal_laps['LaptimeSec'].median()
            out_lap = laps[
                (laps['Driver'] == driver) &
                (laps['LapNumber'] == lap_num + 1) &
                (~laps['LapTime'].isna())
            ]

            # Calcualte pit loss as difference between pit lap and normal lap
            # some tracks mean the out lap is the longer one, so factor in both
            if len(out_lap) > 0:
                out_lap_time = out_lap.iloc[0]['LapTime'].total_seconds()
                pit_loss = (in_lap_time + out_lap_time) - 2 * driver_median_lap
            else:
                pit_loss = in_lap_time - driver_median_lap
            
            if pit_loss > 0:
                pit_losses.append(pit_loss)
        
        # IQR outlier filtering
        pit_losses = np.array(pit_losses)
        q1, q3 = np.percentile(pit_losses, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        filtered = pit_losses[(pit_losses >= lower) & (pit_losses <= upper)]

        if len(filtered) == 0:
            filtered = pit_losses
        
        return float(np.mean(filtered)), float(np.std(filtered))
    
    def get_safety_car_events(self, session: ff1.core.Session) -> List[SafetyCarEvent]:
        
        events = []
        laps = session.laps.copy()

        if 'TrackStatus' in laps.columns:

            sc_laps = set()

            # Track Status Code: 1=Green, 2=Yellow, 4=SC, 5=Red, 6=VSC, 7=SC Ending
            for _, lap in laps.iterrows():
                status = lap.get('TrackStatus', '')
                if pd.isna(status):
                    continue

                # Check SC status codes
                status_str = str(status)
                if '4' in status_str or '6' in status_str or '7' in status_str:
                    sc_laps.add(lap['LapNumber'])

            if sc_laps:
                # Group SC laps into events
                sc_laps = sorted(sc_laps)
                sc_start = sc_laps[0]
                sc_end = sc_laps[0]

                for lap in sc_laps[1:]:
                    if lap == sc_end + 1:
                        sc_end = lap
                    else:
                        events.append(SafetyCarEvent(
                            start_lap=sc_start, 
                            end_lap=sc_end, 
                            duration=sc_end - sc_start + 1))
                        sc_start = lap
                        sc_end = lap

                events.append(SafetyCarEvent(
                    start_lap=sc_start, 
                    end_lap=sc_end, 
                    duration=sc_end - sc_start + 1))
                
        return events
    
    def calculate_sc_probability(self, sc_events: List[SafetyCarEvent], total_laps: int) -> float:
        # Laps under SC
        sc_starts = len(sc_events)

        # Probability per lap
        total_sc_laps = sum(e.duration for e in sc_events)
        normal_laps = total_laps - total_sc_laps
        
        return sc_starts / normal_laps if normal_laps > 0 else 0.0
    
    def get_driver_strategy(self, session: ff1.core.Session, driver_code: str) -> OpponentStrategy:
        
        laps = session.laps.copy()
        driver_laps = laps[laps['Driver'] == driver_code].sort_values('LapNumber')

        if len(driver_laps) == 0:
            raise ValueError(f"No laps found for driver {driver_code}")
        
        # Driver info
        driver_info = session.get_driver(driver_code)
        driver_name = f"{driver_info.get('FirstName', '')} {driver_info.get('LastName', '')}".strip()
        if not driver_name:
            driver_name = driver_code

        # Starting compound
        first_lap = driver_laps.iloc[0]
        start_compound = map_compound(first_lap['Compound'])

        # Pit laps and compounds
        pit_laps = []
        pit_compounds = [start_compound]

        previous_compound = start_compound
        for _, lap in driver_laps.iterrows():
            current_compound = map_compound(lap['Compound'])
            if current_compound != previous_compound:
                pit_laps.append(lap['LapNumber'])
                pit_compounds.append(current_compound)
                previous_compound = current_compound

        # Lap times
        lap_times = []
        for _, lap in driver_laps.iterrows():
            if pd.isna(lap['LapTime']):
                lap_times.append(0.0)
            else:
                lap_times.append(lap['LapTime'].total_seconds())

        # Replace 0.0 lap times with median
        valid_times = [t for t in lap_times if t > 0]
        if valid_times:
            median_time = float(f"{np.median(valid_times):.3f}")
            lap_times = [t if t > 0 else median_time for t in lap_times]

        # Finishing position and total time
        last_lap = driver_laps.iloc[-1]
        finishing_position = int(last_lap['Position']) if not pd.isna(last_lap['Position']) else 20
        total_time = sum(t for t in lap_times if t > 0)

        # DNF detection: driver completed fewer laps than the race total
        total_race_laps = int(laps['LapNumber'].max())
        driver_completed_laps = int(driver_laps['LapNumber'].max())
        dnf = driver_completed_laps < (total_race_laps - 2) # Allow for finishing a lap behind leader

        return OpponentStrategy(
            driver_code=driver_code,
            driver_name=driver_name,
            starting_compound=start_compound,
            pit_laps=pit_laps,
            pit_compounds=pit_compounds,
            total_time=total_time,
            finishing_position=finishing_position,
            num_pit_stops=len(pit_laps),
            lap_times=lap_times,
            dnf=dnf
        )

    
    def get_track_parameters(self, session: ff1.core.Session) -> TrackParameters:

        laps = session.laps.copy()
        total_laps = int(laps['LapNumber'].max())
        track_name = session.event['EventName']

        # Fastest and average lap times
        valid_laps = laps[
            (laps['IsAccurate'] == True) &
            (~laps['LapTime'].isna())
        ]
        lap_times = valid_laps['LapTime'].dt.total_seconds()
        fastest_lap = lap_times.min()
        average_lap = lap_times.mean()

        # Pit loss and standard deviation
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
        
        # Load session and parameters
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

        # Get all drivers and their strategies
        all_drivers = session.laps['Driver'].unique()
        opponent_strategies = []
        for driver in all_drivers:
            if driver != target_driver:
                try:
                    opponent_strategies.append(self.get_driver_strategy(session, driver))
                except Exception as e:
                    print(f"Error extracting strategy for driver {driver}: {e}")
                    continue

        # sort opponents by finishing position
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
        
        # Get full schedule and filter to race events (exclude testing)
        schedule = ff1.get_event_schedule(year)
        race_events = schedule[schedule['EventFormat'] != 'testing']

        all_races = []
        completed_gps = []

        for _, event in race_events.iterrows():
            gp_name = event['EventName']

            # Skip races that haven't happened yet (no session date) or are missing data
            if pd.isna(event['Session5Date']):
                continue

            try:
                print(f"Processing: {gp_name}")
                race_config = self.get_race_config(year, gp_name, target_driver)

                # Check for wet condition (optional)
                if exclude_wet_races:
                    # Check if any opponent used intermediates or wets in their strategy
                    wet_used = any(
                        4 in s.pit_compounds or 5 in s.pit_compounds
                        for s in race_config.opponents
                    )
                    if wet_used:
                        print(f"Excluding {gp_name} - wet race")
                        continue

                all_races.append(race_config)
                completed_gps.append(gp_name)

            except Exception as e:
                print(f"Error processing {gp_name}: {e}")
                continue
        
        # Determine train/test split
        train_races = [gp for gp in completed_gps if gp not in test_races]

        return SeasonConfig(
            year=year,
            target_driver=target_driver,
            races = all_races,
            train_races = train_races,
            test_races = test_races
        )

    def save_race_config(self, race_config: RaceConfig, output_dir: str = "data/races"):
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{race_config.year}_{race_config.name.replace(' ', '_')}.json"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(race_config.to_dict(), f, indent=2, default=str)

        print(f"Saved race config to {filepath}")
        return filepath
    
    def save_season_config(self, season_config: SeasonConfig, output_dir: str = "data/seasons"):
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{season_config.year}_{season_config.target_driver}_season.json"
        filepath = os.path.join(output_dir, filename)
        
        data = {
            "year": season_config.year,
            "target_driver": season_config.target_driver,
            "train_races": season_config.train_races,
            "test_races": season_config.test_races,
            "races": [race.to_dict() for race in season_config.races]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        print(f"Saved season config to {filepath}")
        return filepath
    
if __name__ == "__main__":
    extractor = FastF1DataExtractor()

    print("\nTest Single Race Config Extraction")
    race_config = extractor.get_race_config(2024, 'Miami Grand Prix', 'HAM')
    extractor.save_race_config(race_config)

    print("\nTest Full Season Config Extraction")
    season_config = extractor.get_season_config(
        year=2024, 
        target_driver='HAM', 
        test_races=['Miami Grand Prix', 'Austrian Grand Prix', 'Singapore Grand Prix',
                    'Abu Dhabi Grand Prix', 'Italian Grand Prix']
    )
    extractor.save_season_config(season_config)