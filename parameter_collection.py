
import fastf1 as ff1
import numpy as np
import pandas as pd
from scipy.stats import linregress

ff1.Cache.enable_cache("fastf1/cache")

print("Loading Session Data")
session = ff1.get_session(2024, 'Bahrain', 'R') # Year, Event, Session
session.load(laps=True, telemetry=False, weather=False)
race_dataset = session.laps
print(f"Session Dataset: {race_dataset}")

def get_tyre_degradation(race_dataset):
    # Calculate a linear degradation rate (sec/lap) for tyre degradation parameter

    print("/nCalculating Tyre Degradation")

def get_pit_loss(race_dataset):
    # Calculate average time loss for a pit stop
    print("/nCalculating Pit Loss")

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
    print(f"/nCalculating SC Probability for {year}")

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
        sc_laps += len(session.laps[session.laps['TrackStatus'] == '4'])

    if total_laps == 0:
        return 0.0

    # Probability is number of safety car laps / total number of laps
    sc_probability = sc_laps / total_laps

    print(f"Total laps in {year}: {total_laps}")
    print(f"Total laps with safety car in {year}: {sc_laps}")
    print(f"Safety car probability: {sc_probability:.4f}")

    return sc_probability

def get_slow_stop_probability(year=2024, threshold_s=25.0):
    # Calculate probability of a pit stop being 'slow'
    # Defined as time in pits > threshold across an entire season

    print(f"\nCalculating Slow Stop Probability for {year}")

# tyre_deg_param = get_tyre_degradation(race_dataset)
pit_loss_param = get_pit_loss(race_dataset)
sc_proba_param = get_sc_probability(year=2024)
# slowstop_param = get_slow_stop_probability(race_dataset)