from dataclasses import dataclass
import random
from typing import Optional

@dataclass
class EventParams:
    # Safety Car
    sc_prob: float = 0.05           # Chance of safety car per lap
    sc_speed_factor: float = 1.5    # Lap times are 1.5x slower under safety car
    sc_pit_factor: float = 0.5      # Time lost pitting is reduced

    # Slow Pit Stop
    slow_stop_prob: float = 0.05    # Chance of slow pit stop per pit
    slow_stop_mean: float = 3.0     # Mean delay in seconds
    slow_stop_std: float = 1.0      # Std deviation of delay

class RaceEvents:
    def __init__(self, params: EventParams = None):
        self.params = params or EventParams()
        self.active_event: Optional[str] = None
        self.event_duration = 0
        self.current_event_lap = 0

    def reset(self):
        self.active_event = None
        self.event_duration = 0
        self.current_event_lap = 0

    def step(self):
        if self.active_event:
            self.current_event_lap += 1
            if self.current_event_lap >= self.event_duration:
                self.active_event = None
        else:
            if random.random() < self.params.sc_prob:
                self.trigger_safety_car()

        return self.active_event
    
    def trigger_safety_car(self):
        self.active_event = "safety_car"
        self.event_duration = random.randint(3, 6)  # Safety car lasts 3-6 laps
        self.current_event_lap = 0

    def get_lap_time_multiplier(self) -> float:
        if self.active_event == "safety_car":
            return self.params.sc_speed_factor
        return 1.0
    
    def get_pit_loss_multiplier(self) -> float:
        if self.active_event == "safety_car":
            return self.params.sc_pit_factor
        return 1.0
    
    def get_pit_delay(self) -> float:
        if random.random() < self.params.slow_stop_prob:
            delay = random.gauss(self.params.slow_stop_mean, self.params.slow_stop_std)
            return max(0.0, delay)
        return 0.0