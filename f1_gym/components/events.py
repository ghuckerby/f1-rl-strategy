from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class SafetyCarEvent:
    """Represents a safety car period in a real F1 race"""

    start_lap: int
    end_lap: int
    duration: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SafetyCarEvent':
        return cls(
            start_lap=int(data['start_lap']),
            end_lap=int(data['end_lap']),
            duration=int(data['duration'])
        )

@dataclass
class RealRaceEvents:
    """Manages real race events like safety cars and their impact on lap times and pit stops"""

    sc_events: List[SafetyCarEvent]
    sc_speed_factor: float = 1.4
    sc_pit_factor: float = 0.5

    # State
    current_lap: int = field(default=0, init=False)
    safety_car_active: bool = field(default=False, init=False)

    def reset(self):
        self.current_lap = 0
        self.safety_car_active = False

    def step(self):
        """Advance the race by one lap and update safety car status."""
        self.current_lap += 1
        self.safety_car_active = self.is_sc_lap(self.current_lap)
        return self.safety_car_active
    
    def is_sc_lap(self, lap: int) -> bool:
        """Check if the given lap is under a safety car period."""

        for event in self.sc_events:
            if event.start_lap <= lap <= event.end_lap:
                return True
        return False
    
    def get_lap_time_multiplier(self) -> float:
        """Returns the lap time multiplier based on whether a safety car is active."""

        return self.sc_speed_factor if self.safety_car_active else 1.0
    
    def get_pit_loss_multiplier(self) -> float:
        """Returns the pit stop time multiplier based on whether a safety car is active."""

        return self.sc_pit_factor if self.safety_car_active else 1.0
    
    @classmethod
    def from_race_data(cls, data: Dict[str, Any]) -> 'RealRaceEvents':
        """Creates a RealRaceEvents instance from race data, extracting safety car events and their parameters."""

        sc_events = [SafetyCarEvent.from_dict(event) for event in data.get('sc_events', [])]
        sc_speed_factor = data.get('sc_speed_factor', 1.4)
        sc_pit_factor = data.get('sc_pit_factor', 0.5)
        return cls(sc_events=sc_events, sc_speed_factor=sc_speed_factor, sc_pit_factor=sc_pit_factor)