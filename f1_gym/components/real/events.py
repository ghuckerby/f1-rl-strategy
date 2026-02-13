from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class SafetyCarEvent:
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
    sc_events: List[SafetyCarEvent]
    sc_speed_factor: float = 1.4 # Need to update this
    sc_pit_factor: float = 0.5 # Need to update this

    # State
    current_lap: int = field(default=0, init=False)
    safety_car_active: bool = field(default=False, init=False)

    def reset(self):
        self.current_lap = 0
        self.safety_car_active = False

    def step(self):
        self.current_lap += 1
        self.safety_car_active = self.is_sc_lap(self.current_lap)
        return self.safety_car_active
    
    def is_sc_lap(self, lap: int) -> bool:
        for event in self.sc_events:
            if event.start_lap <= lap <= event.end_lap:
                return True
        return False
    
    def get_lap_time_multiplier(self) -> float:
        return self.sc_speed_factor if self.safety_car_active else 1.0
    
    def get_pit_loss_multiplier(self) -> float:
        return self.sc_pit_factor if self.safety_car_active else 1.0
    
    @classmethod
    def from_race_data(cls, data: Dict[str, Any]) -> 'RealRaceEvents':
        sc_events = [SafetyCarEvent.from_dict(event) for event in data.get('sc_events', [])]
        return cls(sc_events=sc_events)