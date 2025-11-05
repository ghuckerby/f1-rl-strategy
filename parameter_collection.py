
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