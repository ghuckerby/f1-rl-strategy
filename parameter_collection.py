
import fastf1 as ff1
import numpy as np
import pandas as pd
from scipy.stats import linregress

# Cache folder for timing data
ff1.Cache.enable_cache("fastf1/cache")

# UPDATE Change to produce per race parameters

session = ff1.get_session(2024, 'Britain', 'R') # Year, Event, Session
session.load(laps=True, telemetry=False, weather=False)
race_dataset = session.laps
print(f"Session Dataset: {race_dataset}")

def get_tyre_parameters(race_dataset):
    deg_rates = { "SOFT": [], "MEDIUM": [], "HARD": []}
    base_times = { "SOFT": [], "MEDIUM": [], "HARD": []}

    for driver in race_dataset['Driver'].unique():
        laps = race_dataset.pick_drivers(driver)
        for stint_num in laps['Stint'].unique():
            stint_laps = laps[laps['Stint'] == stint_num]

            # Stints with at least 5 laps (excluding in and out laps)
            clean_stint = stint_laps.iloc[1:-1]
            if len(clean_stint) < 5:
                continue

            compound = clean_stint['Compound'].iloc[0]
            if compound not in deg_rates:
                continue

            x = clean_stint['TyreLife']
            y = clean_stint['LapTime'].dt.total_seconds()

            tyre_regression = linregress(x, y)
            slope = tyre_regression.slope
            intercept = tyre_regression.intercept

            if 0 < slope < 1.0 and 60 < intercept < 200:
                deg_rates[compound].append(slope)
                base_times[compound].append(intercept)

    average_deg = {}
    average_base = {}

    for compound, rates in deg_rates.items():
        if rates:
            average_deg[compound] = np.mean(rates)
            print(f"Average {compound} degradation: {average_deg[compound]:.3f} sec/lap")
    
    for compound, bases in base_times.items():
        if bases:
            average_base[compound] = np.mean(bases)
            print(f"Average {compound} base lap time: {average_base[compound]:.3f} sec")

    return average_deg, average_base

def get_pit_loss(race_dataset):
    # Calculate average time loss for a pit stop
    print("\nCalculating Pit Loss")

    # Get all laps without a pit stop
    laps_no_pit = race_dataset.pick_wo_box()
    average_lap = laps_no_pit['LapTime'].dt.total_seconds().mean()

    # Get all laps with a pit stop
    laps_with_pit = race_dataset.pick_box_laps()
    average_pit_lap = laps_with_pit['LapTime'].dt.total_seconds().mean()

    pit_loss = average_pit_lap - average_lap

    print(f"Average normal lap: {average_lap:.2f}s")
    print(f"Average lap with pit stop: {average_pit_lap:.2f}s")
    print(f"Pit Loss: {pit_loss:.2f}s")

    return pit_loss

def get_sc_prob(event_dataset):
    total_laps = len(event_dataset)
    if total_laps == 0:
        return 0.0
    
    sc_laps = len(event_dataset[event_dataset['TrackStatus'].isin(['4', '5'])])
    sc_prob = sc_laps / total_laps
    return sc_prob

def get_slow_stop_probability(year, threshold):
    # Calculate probability of a pit stop being 'slow'
    print(f"\nCalculating Slow Stop Probability")
    return 0.001

def main():
    YEAR = 2024
    SLOW_STOP_THRESHOLD = 25.0

    all_race_data = []
    schedule = ff1.get_event_schedule(YEAR)

    races = schedule[schedule['EventFormat'] != 'testing']
    races = races[races['EventName'].str.contains("Grand Prix")]
    races = races[races['RoundNumber'] < 25]

    print(f"Found {len(races)} races in the {YEAR} season.")

    for i, event in races.iterrows():
        event_name = event['EventName']
        round_num = event['RoundNumber']
        print(f"\nProcessing Round {round_num}: {event_name}")

        session = ff1.get_session(YEAR, event_name, 'R')
        session.load(laps=True, telemetry=False, weather=False)
        race_dataset = session.laps

        tyre_deg_param, tyre_base_param = get_tyre_parameters(race_dataset)
        pit_loss_param = get_pit_loss(race_dataset)
        sc_proba_param = get_sc_prob(race_dataset)
        slowstop_param = get_slow_stop_probability(year=YEAR, threshold=SLOW_STOP_THRESHOLD)

        race = {
            'Round': round_num,
            'Event': event_name,
            'PitLoss': pit_loss_param,
            'SC_Probability': sc_proba_param,
            f"SlowStop_Probability_{SLOW_STOP_THRESHOLD}s": slowstop_param,
            'Base_SOFT': tyre_base_param.get("SOFT", None),
            'Base_MEDIUM': tyre_base_param.get("MEDIUM", None),
            'Base_HARD': tyre_base_param.get("HARD", None),
            'Deg_SOFT': tyre_deg_param.get("SOFT", None),
            'Deg_MEDIUM': tyre_deg_param.get("MEDIUM", None),
            'Deg_HARD': tyre_deg_param.get("HARD", None),
        }

        all_race_data.append(race)
        print(f"Completed processing for {event_name}")

    df = pd.DataFrame(all_race_data)
    columns_order = [
        'Round', 'Event', 'PitLoss', 'SC_Probability',
        f"SlowStop_Probability_{SLOW_STOP_THRESHOLD}s",
        'Base_SOFT', 'Deg_SOFT',
        'Base_MEDIUM', 'Deg_MEDIUM',
        'Base_HARD', 'Deg_HARD'
    ]
    df = df[columns_order]
    df = df.sort_values(by='Round').reset_index(drop=True)
    output_file = f"f1_{YEAR}_season_parameters.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved season parameters to {output_file}")

if __name__ == "__main__":
    main()