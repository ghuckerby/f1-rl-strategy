from dataclasses import dataclass

@dataclass
class RewardConfig:
    """Config for F1 Environment Reward Function."""

    # Base Rewards
    lap_time_reward_weight: float = 1.0

    position_gain_reward: float = 10.0
    final_position_reward: float = 150.0

    # Pit Stop Strategy Window
    pit_window_start: int = 15
    pit_window_end: int = 40
    strategic_pit_reward: float = 0.0
    compound_change_reward: float = 0.0

    # Tyre Wear Penalty
    tyre_wear_threshold: float = 0.5
    tyre_wear_penalty: float = -10.0

    # Rule enforcement
    rule_penalty_threshold_one: int = 10
    rule_penalty_one_value: float = -100.0
    rule_penalty_threshold_two: int = 20
    rule_penalty_two_value: float = -50.0

    rule_penalty_violation: float = -1000.0

