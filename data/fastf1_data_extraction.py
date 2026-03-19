
import fastf1 as ff1
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import os
import pandas as pd
import numpy as np

ff1.Cache.enable_cache("fastf1_cache/cache")

# Data Classes
@dataclass
class TyreCompound:
    compound: str 
    compound_id: int

@dataclass
class OpponentStrategy:
    driver_code: str
    driver_name: str
    starting_compound: int
    starting_position: int
    pit_laps: List[int]
    pit_compounds: List[int]
    total_time: float
    finishing_position: int
    num_pit_stops: int
    lap_times: List[float]
    positions: List[int]
    dnf: bool
    time_penalty: float = 0.0

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
    tyre_compounds: Dict[int, TyreCompound]
    sc_events: List[SafetyCarEvent]
    sc_probability: float
    sc_speed_factor: float
    sc_pit_factor: float
    opponents: List[OpponentStrategy]
    target_driver: str
    target_driver_strategy: OpponentStrategy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "year": self.year,
            "name": self.name,
            "track": asdict(self.track),
            "tyre_compounds": {k: asdict(v) for k, v in self.tyre_compounds.items()},
            "sc_events": [asdict(e) for e in self.sc_events],
            "sc_probability": self.sc_probability,
            "sc_speed_factor": self.sc_speed_factor,
            "sc_pit_factor": self.sc_pit_factor,
            "opponents": [asdict(o) for o in self.opponents],
            "target_driver": self.target_driver,
            "target_driver_strategy": asdict(self.target_driver_strategy),
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
    
    def get_tyre_compounds(self, session: ff1.core.Session) -> Dict[int, TyreCompound]:
        laps = session.laps.copy()

        # Filter invalid laps (pit and SC laps)
        valid_laps = laps[
            (laps['IsAccurate'] == True) &
            (laps['PitInTime'].isna()) &
            (laps['PitOutTime'].isna()) &
            (~laps['LapTime'].isna())
        ].copy()
        
        # Get compounds used in the race
        used_compounds = valid_laps['Compound'].unique()
        tyre_compounds = {}
        for compound_str in used_compounds:
            if pd.isna(compound_str) or compound_str.upper() not in COMPOUND_MAP:
                continue

            compound_id = COMPOUND_MAP[compound_str.upper()]
            tyre_compounds[compound_id] = TyreCompound(
                compound=compound_str,
                compound_id=compound_id,
            )

        return tyre_compounds
    
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

        # Starting compound and grid position
        first_lap = driver_laps.iloc[0]
        start_compound = map_compound(first_lap['Compound'])
        
        # Grid position (starting position)
        if 'GridPosition' in first_lap and not pd.isna(first_lap['GridPosition']):
            starting_position = int(first_lap['GridPosition'])
        else:
            # Fallback: estimate from first lap position
            starting_position = int(first_lap['Position']) if not pd.isna(first_lap['Position']) else 20

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
        driver_laps_sorted = driver_laps.sort_values('LapNumber').copy()

        # Convert Time column to seconds (absolute session time)
        if 'Time' in driver_laps_sorted.columns:
            driver_laps_sorted['TimeSec'] = pd.to_timedelta(
                driver_laps_sorted['Time']
            ).dt.total_seconds()
        else:
            driver_laps_sorted['TimeSec'] = np.nan

        # LapTime for fallback
        driver_laps_sorted['LapTimeSec'] = np.nan
        mask = driver_laps_sorted['LapTime'].notna()
        driver_laps_sorted.loc[mask, 'LapTimeSec'] = pd.to_timedelta(
            driver_laps_sorted.loc[mask, 'LapTime']
        ).dt.total_seconds()

        lap_times: List[float] = []
        prev_time_sec: Optional[float] = None

        for idx_row, (_, lap) in enumerate(driver_laps_sorted.iterrows()):
            time_sec = lap['TimeSec']
            lap_time_sec = lap['LapTimeSec']

            if idx_row == 0:
                if not pd.isna(lap_time_sec) and lap_time_sec > 0:
                    elapsed = lap_time_sec
                elif not pd.isna(time_sec):
                    if 'LapStartTime' in lap and not pd.isna(lap['LapStartTime']):
                        start_sec = pd.to_timedelta(lap['LapStartTime']).total_seconds()
                        elapsed = time_sec - start_sec
                    else:
                        elapsed = time_sec
                else:
                    elapsed = 0.0
            else:
                # Subsequent laps: prefer Time delta, fall back to LapTime
                if not pd.isna(time_sec) and prev_time_sec is not None and not pd.isna(prev_time_sec):
                    elapsed = time_sec - prev_time_sec
                elif not pd.isna(lap_time_sec) and lap_time_sec > 0:
                    elapsed = lap_time_sec
                else:
                    elapsed = 0.0

            lap_times.append(float(f"{elapsed:.3f}"))
            prev_time_sec = time_sec

        # Final fallback: replace any remaining 0.0 with median (very rare)
        valid_times = [t for t in lap_times if t > 0]
        if valid_times:
            median_time = float(f"{np.median(valid_times):.3f}")
            lap_times = [t if t > 0 else median_time for t in lap_times]

        # Positions
        positions: List[int] = []
        for _, lap in driver_laps_sorted.iterrows():
            pos = lap.get('Position', np.nan)
            if pd.isna(pos):
                positions.append(20)  # fallback
            else:
                positions.append(int(pos))

        # Finishing position and total time
        last_lap = driver_laps_sorted.iloc[-1]
        finishing_position = int(last_lap['Position']) if not pd.isna(last_lap['Position']) else 20
        total_time = sum(t for t in lap_times if t > 0)

        # DNF detection: driver completed fewer laps than the race total
        total_race_laps = int(laps['LapNumber'].max())
        driver_completed_laps = int(driver_laps_sorted['LapNumber'].max())
        dnf = driver_completed_laps < (total_race_laps - 2) # Allow for finishing a lap behind leader

        return OpponentStrategy(
            driver_code=driver_code,
            driver_name=driver_name,
            starting_compound=start_compound,
            starting_position=starting_position,
            pit_laps=pit_laps,
            pit_compounds=pit_compounds,
            total_time=total_time,
            finishing_position=finishing_position,
            num_pit_stops=len(pit_laps),
            lap_times=lap_times,
            positions=positions,
            dnf=dnf
        )
    
    def get_sc_factors(self, session: ff1.core.Session, track_params: TrackParameters, sc_events: List[SafetyCarEvent]) -> Tuple[float, float]:

        # No safety car in this race
        if not sc_events:
            return 0.0, 0.0

        laps = session.laps.copy()

        # Build set of all lap numbers under safety car
        sc_lap_numbers = set()
        for event in sc_events:
            for lap_num in range(int(event.start_lap), int(event.end_lap) + 1):
                sc_lap_numbers.add(lap_num)

        # Normal laps: valid laps that are not under SC and not pit laps
        normal_laps = laps[
            (laps['IsAccurate'] == True) &
            (laps['PitInTime'].isna()) &
            (laps['PitOutTime'].isna()) &
            (~laps['LapTime'].isna()) &
            (~laps['LapNumber'].isin(sc_lap_numbers))
        ].copy()
        normal_laps['LaptimeSec'] = normal_laps['LapTime'].dt.total_seconds()
        normal_median = normal_laps['LaptimeSec'].median()

        # SC laps: laps within SC periods with valid timing
        sc_laps = laps[
            (laps['LapNumber'].isin(sc_lap_numbers)) &
            (~laps['LapTime'].isna())
        ].copy()
        sc_laps['LaptimeSec'] = sc_laps['LapTime'].dt.total_seconds()

        # SC Speed Factor: median SC lap time / median normal lap time
        if len(sc_laps) >= 3 and not pd.isna(normal_median):
            sc_median = sc_laps['LaptimeSec'].median()
            sc_speed_factor = sc_median / normal_median
            # Clamp to reasonable range
            sc_speed_factor = max(1.05, min(sc_speed_factor, 2.0))
        else:
            # Insufficient data — fall back to simulation default
            sc_speed_factor = 1.4

        # SC Pit Factor: mean SC pit loss / normal pit loss
        sc_pit_in_laps = laps[
            (~laps['PitInTime'].isna()) &
            (laps['LapNumber'].isin(sc_lap_numbers)) &
            (~laps['LapTime'].isna())
        ].copy()

        # SC non-pit laps: laps under SC, not pit in/out, with valid time
        sc_non_pit_laps = laps[
            (laps['LapNumber'].isin(sc_lap_numbers)) &
            (laps['PitInTime'].isna()) &
            (laps['PitOutTime'].isna()) &
            (~laps['LapTime'].isna())
        ].copy()
        sc_non_pit_laps['LaptimeSec'] = sc_non_pit_laps['LapTime'].dt.total_seconds()

        sc_pit_losses = []
        for _, lap in sc_pit_in_laps.iterrows():
            driver = lap['Driver']
            lap_num = lap['LapNumber']
            in_lap_time = lap['LapTime'].total_seconds()

            driver_sc_laps = sc_non_pit_laps[sc_non_pit_laps['Driver'] == driver]
            if len(driver_sc_laps) > 0:
                sc_baseline = driver_sc_laps['LaptimeSec'].median()
            elif len(sc_non_pit_laps) > 0:
                sc_baseline = sc_non_pit_laps['LaptimeSec'].median()
            else:
                continue

            # Skip extreme outliers (crashes / red flags)
            if in_lap_time > 3 * sc_baseline:
                continue

            pit_loss = in_lap_time - sc_baseline

            if pit_loss > 0:
                sc_pit_losses.append(pit_loss)

        if len(sc_pit_losses) >= 2 and track_params.pit_loss_time > 0:
            sc_pit_factor = float(np.mean(sc_pit_losses)) / track_params.pit_loss_time
            sc_pit_factor = max(0.1, min(sc_pit_factor, 1.0))
        elif len(sc_pit_losses) == 1 and track_params.pit_loss_time > 0:
            sc_pit_factor = float(sc_pit_losses[0]) / track_params.pit_loss_time
            sc_pit_factor = max(0.1, min(sc_pit_factor, 1.0))
        else:
            sc_pit_factor = 0.5

        return float(sc_speed_factor), float(sc_pit_factor)
    
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
    
    def get_time_penalties(self, session: ff1.core.Session) -> Dict[str, float]:
        results = session.results
        laps = session.laps

        if results is None or len(results) == 0:
            return {}

        # Leader's last-lap absolute time
        leader_abbr = results.iloc[0]['Abbreviation']
        leader_laps = laps[laps['Driver'] == leader_abbr].sort_values('LapNumber')
        if len(leader_laps) == 0:
            return {}
        leader_finish = leader_laps['Time'].dt.total_seconds().iloc[-1]

        penalties: Dict[str, float] = {}
        for _, row in results.iterrows():
            abbr = row['Abbreviation']
            if pd.isna(row['Time']):
                continue  # DNF

            # Official gap from classification
            official_gap = row['Time'].total_seconds()
            if row['Position'] == 1:
                official_gap = 0.0

            # Actual on-track gap from lap data
            driver_laps = laps[laps['Driver'] == abbr].sort_values('LapNumber')
            if len(driver_laps) == 0:
                continue
            driver_finish = driver_laps['Time'].dt.total_seconds().iloc[-1]
            actual_gap = driver_finish - leader_finish

            penalty = official_gap - actual_gap
            if penalty >= 1.0:
                penalties[abbr] = round(penalty, 0)
                print(f"Time penalty: {abbr} +{penalties[abbr]:.0f}s")

        return penalties

    def get_race_config(self, year: int, gp: str, target_driver: str) -> RaceConfig:
        
        # Load session and parameters
        print(f"Loading session: {year} {gp}")
        session = self.load_session(year, gp, 'R')

        print(f"Extracting track parameters")
        track_params = self.get_track_parameters(session)
        
        print(f"Extracting tyre compounds")
        tyre_compounds = self.get_tyre_compounds(session)

        print(f"Extracting safety car events")
        sc_events = self.get_safety_car_events(session)
        sc_prob = self.calculate_sc_probability(sc_events, track_params.total_laps)

        print(f"Extracting safety car pit loss factors")
        sc_speed_factor, sc_pit_factor = self.get_sc_factors(session, track_params, sc_events)

        print(f"Extracting time penalties")
        penalties = self.get_time_penalties(session)

        print(f"Extracting target driver strategies")
        target_strategy = self.get_driver_strategy(session, target_driver)

        # Apply penalty to target driver if any
        if target_driver in penalties:
            target_strategy.time_penalty = penalties[target_driver]

        # Get all drivers and their strategies
        all_drivers = session.laps['Driver'].unique()
        opponent_strategies = []
        for driver in all_drivers:
            if driver != target_driver:
                try:
                    strategy = self.get_driver_strategy(session, driver)
                    # Apply penalty if applicable
                    if driver in penalties:
                        strategy.time_penalty = penalties[driver]
                    opponent_strategies.append(strategy)
                except Exception as e:
                    print(f"Error extracting strategy for driver {driver}: {e}")
                    continue

        # sort opponents by finishing position
        opponent_strategies.sort(key=lambda x: x.finishing_position)

        return RaceConfig(
            year=year,
            name=gp,
            track=track_params,
            tyre_compounds=tyre_compounds,
            sc_events=sc_events,
            sc_probability=sc_prob,
            sc_speed_factor=sc_speed_factor,
            sc_pit_factor=sc_pit_factor,
            opponents=opponent_strategies,
            target_driver=target_driver,
            target_driver_strategy=target_strategy,
        )
    
    def get_season_config(
            self, year: int, 
            target_driver: str, 
            test_races: Optional[List[str]] = None,
            skip_races: Optional[List[str]] = None,
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
        train_races = [gp for gp in completed_gps if gp not in test_races and gp not in skip_races]

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

    def save_season_split(self, season_config: SeasonConfig, train_dir: str = "data/training_races", test_dir: str = "data/test_races"):
        train_set = set(season_config.train_races)
        test_set = set(season_config.test_races)

        train_count, test_count = 0, 0
        for race in season_config.races:
            if race.name in train_set:
                self.save_race_config(race, output_dir=train_dir)
                train_count += 1
            elif race.name in test_set:
                self.save_race_config(race, output_dir=test_dir)
                test_count += 1
            else:
                # Race not in either split — skip
                print(f"Skipping {race.name} (not in train or test split)")

        print(f"\nSaved {train_count} training races to {train_dir}/")
        print(f"Saved {test_count} test races to {test_dir}/")
    
if __name__ == "__main__":
    extractor = FastF1DataExtractor()

    print("\nExtracting 2024 season for HAM")
    season_config = extractor.get_season_config(
        year=2024,
        target_driver='HAM',
        test_races=['Chinese Grand Prix', 'Hungarian Grand Prix', 'Singapore Grand Prix',
                    'Bahrain Grand Prix', 'Mexico City Grand Prix'],
        # Skipped races:
        # Monaco: Red flag first lap, hard to predict lap times
        # USA and Australia: Hamilton DNF
        # Japanese GP: Bad Data
        skip_races=['Monaco Grand Prix', 'United States Grand Prix', 'Australian Grand Prix', 'Japanese Grand Prix']
    )

    # Save individual race JSONs into training_races/ and test_races/
    extractor.save_season_split(season_config)

    # Also save the combined season JSON 
    # extractor.save_season_config(season_config)