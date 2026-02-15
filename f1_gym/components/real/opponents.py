from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class RealOpponent:

    driver_code: str
    driver_name: str
    starting_compound: int
    starting_position: int
    pit_laps: List[float]
    pit_compounds: List[int]
    lap_times: List[float]
    finishing_position: int
    dnf: bool
    total_time: float
    num_pit_stops: int = 0

    def step(self) -> float:
        if self.has_finished or self.current_lap >= len(self.lap_times):
            self.has_finished = True
            return 0.0
        
        # Lap times and cumulative time
        self.current_lap_time = self.lap_times[self.current_lap]
        self.cumulative_time += self.current_lap_time

        # Update compound if pit occurred (don't need pit delay)
        lap_number = self.current_lap + 1
        if lap_number in self.pit_laps:
            pit_index = self.pit_laps.index(lap_number)
            if pit_index + 1 < len(self.pit_compounds):
                self.current_compound = self.pit_compounds[pit_index + 1]

        self.current_lap += 1

        # Check for dnf
        if self.dnf and self.current_lap >= len(self.lap_times):
            self.has_finished = True

        return self.current_lap_time

    def reset(self):
        self.current_lap = 0
        self.current_compound = self.starting_compound
        self.cumulative_time = 0.0
        self.current_lap_time = 0.0
        self.has_finished = False

    def get_position(self, race_time: float):

        if race_time <= 0:
            return 0.0
        
        cumulative = 0.0
        for i, lap_time in enumerate(self.lap_times):
            if cumulative + lap_time > race_time:
                fraction = (race_time - cumulative) / lap_time
                return i + fraction
            cumulative += lap_time

        return len(self.lap_times)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RealOpponent':
        pit_laps = [int(lap) for lap in data.get('pit_laps', [])]
        return cls(
            driver_code=data['driver_code'],
            driver_name=data['driver_name'],
            starting_compound=data['starting_compound'],
            starting_position=data['starting_position'],
            pit_laps=pit_laps,
            pit_compounds=data.get('pit_compounds', []),
            lap_times=data.get('lap_times', []),
            finishing_position=data.get('finishing_position', 0),
            dnf=data.get('dnf', False),
            total_time=data.get('total_time', 0.0),
            num_pit_stops=len(pit_laps)
        )
