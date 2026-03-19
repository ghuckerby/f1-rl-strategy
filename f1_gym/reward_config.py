from dataclasses import dataclass

@dataclass
class RewardConfig:
    """Configuration for reward calculation in the F1 Gym environment."""

    # Base Rewards
    lap_time_reward_weight: float = 0.2
    time_benchmark = 95.0

    # Position Rewards
    position_gain_reward: float = 2.0

    # Pit Stop
    pit_stop_penalty: float = -1.0

    # Final position reward: (20 - position) * final_position_weight
    final_position_weight: float = 3.0

    # Progressive rule penalty (at least 2 compounds must be used)
    rule_penalty_base: float = -5.0
    rule_penalty_start_pct: float = 0.4
    rule_penalty_exponent: float = 2.0

    # Terminal penalty for failing to use 2 compounds by race end
    terminal_rule_penalty: float = -200.0