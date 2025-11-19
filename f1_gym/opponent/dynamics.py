
from dataclasses import dataclass

# Tyre Compound Class
@dataclass
class TyreCompound:
    name: str
    base_lap_time: float
    deg_rate: float

# Track Parameter Class
@dataclass
class TrackParams:
    laps: int = 50
    pit_loss: float = 25.0

# Compound choices
compounds = {
    1: TyreCompound(
        name="SOFT",
        base_lap_time=90.0,
        deg_rate=0.15,
    ),
    2: TyreCompound(
        name="MEDIUM",
        base_lap_time=91.0,
        deg_rate=0.10,
    ),
    3: TyreCompound(
        name="HARD",
        base_lap_time=92.0,
        deg_rate=0.07,
    )
}

# Current lap time calculation
def calculate_lap_time(compound: TyreCompound, age: int) -> float:
    return compound.base_lap_time + compound.deg_rate * (age - 1)