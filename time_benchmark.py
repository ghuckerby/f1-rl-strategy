import itertools
from dataclasses import dataclass
from typing import Dict, Tuple

# Script for calculating the optimal strategy for the deterministic race time

SOFT, MEDIUM, HARD = 0, 1, 2

BASE_LAP = {
    SOFT: 88,
    MEDIUM: 90,
    HARD: 91
}

DEG_RATE = {
    SOFT: 0.20,
    MEDIUM: 0.10,
    HARD: 0.06
}

@dataclass
class TrackParams:
    laps: int = 50
    pit_loss: float = 20.0

TRACK = TrackParams()

def calculate_lap_time(compound: int, stint_age: int) -> float:
    return BASE_LAP[compound] + DEG_RATE[compound] * stint_age

def calculate_stint_time(start_lap: int, end_lap: int, compound: int) -> float:
    total_time = 0.0
    for lap in range(start_lap, end_lap + 1):
        stint_age = lap - start_lap
        total_time += calculate_lap_time(compound, stint_age)
    return total_time

def calculate_one_stop(start_compound: int, pit_lap: int, next_compound: int) -> float:
    if start_compound == next_compound:
        return float('inf')
    
    stint_1_time = calculate_stint_time(1, pit_lap - 1, start_compound)
    pit_lap_time = calculate_lap_time(next_compound, 0) + TRACK.pit_loss
    stint_2_time = calculate_stint_time(pit_lap + 1, TRACK.laps, next_compound)

    return stint_1_time + pit_lap_time + stint_2_time

def calculate_two_stop(start_compound: int, pit_lap_1: int, compound_2: int, pit_lap_2: int, compound_3: int) -> float:
    if start_compound == compound_2 or compound_2 == compound_3:
        return float('inf')
        
    stint_1_time = calculate_stint_time(1, pit_lap_1 - 1, start_compound)
    pit_1_time = calculate_lap_time(compound_2, 0) + TRACK.pit_loss

    stint_2_time = calculate_stint_time(pit_lap_1 + 1, pit_lap_2 - 1, compound_2)
    pit_2_time = calculate_lap_time(compound_3, 0) + TRACK.pit_loss
    
    stint_3_time = calculate_stint_time(pit_lap_2 + 1, TRACK.laps, compound_3)
    
    return stint_1_time + pit_1_time + stint_2_time + pit_2_time + stint_3_time


def find_fastest_race():
    compounds = [SOFT, MEDIUM, HARD]
    compound_names = {SOFT: "Soft", MEDIUM: "Medium", HARD: "Hard"}

    best_one_stop = {"strategy": "", "time": float('inf'), "lap": 0}
    for start_compound in compounds:
        for next_compound in compounds:
            if start_compound == next_compound:
                continue 
                
            for pit_lap in range(2, TRACK.laps):
                time = calculate_one_stop(start_compound, pit_lap, next_compound)
                
                if time < best_one_stop["time"]:
                    strategy_name = f"{compound_names[start_compound]} -> {compound_names[next_compound]}"
                    best_one_stop = {"strategy": strategy_name, "time": time, "lap": pit_lap}

    print(f"\nBest 1-Stop: Strategy: {best_one_stop['strategy']}, Pit Lap: {best_one_stop['lap']}, Time: {best_one_stop['time']:.2f}s")

    best_two_stop = {"strategy": "", "time": float('inf'), "laps": (0, 0)}
    for start_compound, compound_2, compound_3 in itertools.permutations(compounds, 3):
        for pit_lap_1, pit_lap_2 in itertools.combinations(range(2, TRACK.laps), 2):
            time = calculate_two_stop(start_compound, pit_lap_1, compound_2, pit_lap_2, compound_3)
            
            if time < best_two_stop["time"]:
                strategy_name = f"{compound_names[start_compound]} -> {compound_names[compound_2]} -> {compound_names[compound_3]}"
                best_two_stop = {"strategy": strategy_name, "time": time, "laps": (pit_lap_1, pit_lap_2)}

    print(f"\nBest 2-Stop: Strategy: {best_two_stop['strategy']}, Pit Laps: {best_two_stop['laps'][0]} & {best_two_stop['laps'][1]}, Time: {best_two_stop['time']:.2f}s")
    
    fastest_race = min(best_one_stop["time"], best_two_stop["time"])
    print(f"\nFastest Race Time: {fastest_race:.2f}s")

if __name__ == "__main__":
    find_fastest_race()