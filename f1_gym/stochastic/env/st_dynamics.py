from dataclasses import dataclass

# Tire encodings
SOFT, MEDIUM, HARD = 0, 1, 2

# Base Lap Times for each compound
BASE_LAP = {

}

# Wear penalty per lap (simple linear degradation)
DEG_RATE = {

}

# Parameters for track
@dataclass
class TrackParams:
    laps: int = 50
    pit_loss: float = 20.0
    max_stint_age: int = 50

# Current lap time calculation
    # Based on degradation rate and age
def calculate_lap_time(compound: int, stint_age: int) -> float:
    return BASE_LAP[compound] + DEG_RATE[compound] * stint_age