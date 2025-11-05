
import fastf1 as ff1
import numpy as np
import pandas as pd
from scipy.stats import linregress

ff1.Cache.enable_cache("fastf1/cache")

print("Loading Session Data")
session = ff1.get_session(2024, 'Bahrain', 'R') # Year, Event, Session
session.load(laps=True, telemetry=False, weather=False)
laps_df = session.laps
print(f"Session Loaded, laps: {laps_df}")

def get_tyre_degradation(laps_df):
    # Calculate a linear degradation rate (sec/lap) for tyre degradation parameter

    print("/nCalculating Tyre Degradation")

def get_pit_loss(laps_df):
    # Calculate average time loss for a pit stop

    print("/nCalculating Pit Loss")

def get_sc_probability(year=2024):
    # Calculate per lap probability of a Safety Car across an entire season

    print(f"/nCalculating SC Probability for {year}")

def get_slow_stop_probability(year=2024, threshold_s=25.0):
    # Calculate probability of a pit stop being 'slow'
    # Defined as time in pits > threshold across an entire season

    print(f"/nCalculating Slow Stop Probability for {year}")