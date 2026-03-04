
from dataclasses import dataclass
import random
from typing import List, Dict, Any, Tuple

# Tyre Compound Class
@dataclass
class TyreCompound:
    """Defines the properties of a tyre compound."""

    name: str
    base_lap_time: float
    deg_rate: float

# Track Parameter Class
# Laps and pit stop time loss
@dataclass
class TrackParams:
    """Defines the parameters of the track"""

    laps: int = 50
    pit_loss: float = 25.0

# Compound choices
    # 1: Soft, 2: Medium, 3: Hard
    # Base lap times and degradation rates
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
        deg_rate=0.05,
    )
}

# Current lap time calculation
def calculate_lap_time(compound: TyreCompound, age: int) -> float:
    """Calculates the lap time based on the compound and its age."""
    return compound.base_lap_time + compound.deg_rate * (age - 1)