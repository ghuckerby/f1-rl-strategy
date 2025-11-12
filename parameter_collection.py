
import fastf1 as ff1
import numpy as np
import pandas as pd
from scipy.stats import linregress

# Cache folder for timing data
ff1.Cache.enable_cache("fastf1/cache")

session = ff1.get_session(2024, 'China', 'R') # Year, Event, Session
session.load(laps=True, telemetry=False, weather=False)
race_dataset = session.laps
print(f"Session Dataset: {race_dataset}")

def get_tyre_degradation(race_dataset):
    # Calculate a linear degradation rate (sec/lap) for tyre degradation parameter
    print("\nCalculating Tyre Degradation")

    deg_rates = {
        "SOFT": [],
        "MEDIUM": [],
        "HARD": []
    }

    for driver in race_dataset['Driver'].unique():
        laps = race_dataset.pick_drivers(driver)

        for stint_num in laps['Stint'].unique():
            stint_laps = laps[laps['Stint'] == stint_num]

            clean_stint = stint_laps.iloc[1:-1]
            if len(clean_stint) < 5:
                continue

            compound = clean_stint['Compound'].iloc[0]
            if compound not in deg_rates:
                continue

            x = clean_stint['TyreLife']
            y = clean_stint['LapTime'].dt.total_seconds()

            tyre_regression = linregress(x, y)
            print(f"{compound} Tyre Deg: {tyre_regression.slope}")
            if 0 < tyre_regression.slope < 1.0:
                deg_rates[compound].append(tyre_regression.slope)

    average_deg = {}
    for compound, rates in deg_rates.items():
        if rates:
            average_deg[compound] = np.mean(rates)
            print(f"Average {compound} degradation: {average_deg[compound]:.3f} sec/lap")

    return average_deg

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

def get_sc_probability(year=2024):
    # Calculate per lap probability of a Safety Car across an entire season
    print(f"\nCalculating SC Probability for {year}")

    total_laps = 0
    sc_laps = 0

    schedule = ff1.get_event_schedule(year)

    races = schedule[schedule['EventFormat'] != 'testing']
    races = races[races['RoundNumber'] < 25]

    # Loops over each event to collect total laps and sc laps
    for i, event in races.iterrows():
        print(f"{event['EventName']}:")
        session = ff1.get_session(year, event['EventName'], 'R')
        session.load(laps=True, telemetry=False, weather=False)

        total_laps += len(session.laps)
        sc_laps += len(session.laps[session.laps['TrackStatus'].isin(['4', '5'])])

    if total_laps == 0:
        return 0.0

    # Probability is number of safety car laps / total number of laps
    sc_probability = sc_laps / total_laps

    print(f"Total laps in {year}: {total_laps}")
    print(f"Total laps with safety car in {year}: {sc_laps}")
    print(f"Safety car probability: {sc_probability:.4f}")

    return sc_probability

def get_slow_stop_probability(year, threshold):
    # Calculate probability of a pit stop being 'slow'
    # Defined as time in pits > (threshold seconds) across an entire season
    print(f"\nCalculating Slow Stop Probability for {year}")

    all_stops = []
    schedule = ff1.get_event_schedule(year)
    races = schedule[schedule['EventFormat'] != 'testing']
    races = races[races['RoundNumber'] < 25]

    for i, event in races.iterrows():
        print(f"{event['EventName']}:")
        session = ff1.get_session(year, event['EventName'], 'R')
        session.load(laps=True, telemetry=False, weather=False)

        # All laps with pitintime are pit stop laps
        pit_laps = session.laps[session.laps['PitInTime'].notna()]
        all_stops.append(pit_laps)

    if not all_stops:
        return 0.0
    
    # Combine each pit lap from each race into one 
    full_pit_data = pd.concat(all_stops)

    print(full_pit_data.describe())
    
    # Calculate time spent in pit lane
    full_pit_data['PitLaneTime'] = (full_pit_data['PitOutTime'] - full_pit_data['PitInTime']).dt.total_seconds()

    num_pit_stops = len(full_pit_data)
    num_slow_stops = len(full_pit_data[full_pit_data['PitLaneTime'] > threshold])

    if num_pit_stops == 0:
        return 0.0
    
    slow_stop_probability = num_slow_stops / num_pit_stops

    print(f"Total pit stops in {year}: {num_pit_stops}")
    print(f"Slow stops > {threshold}: {num_slow_stops}")
    print(f"Slow stop probability: {slow_stop_probability:.4f}")

    return slow_stop_probability

tyre_deg_param = get_tyre_degradation(race_dataset)
pit_loss_param = get_pit_loss(race_dataset)
sc_proba_param = get_sc_probability(year=2024)
# slowstop_param = get_slow_stop_probability(year=2024, threshold=10.0)

print(f"Tyre Degs: {tyre_deg_param}, \nPit Loss: {pit_loss_param:.4f}, \nSC Probability: {sc_proba_param:.4f}")