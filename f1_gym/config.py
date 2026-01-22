from dataclasses import dataclass

@dataclass
class RewardConfig:
    """Config for F1 Environment Reward Function."""

    # Base Rewards
    lap_time_reward_weight: float = -1.0
    position_gain_reward: float = 10.0
    final_position_reward: float = 100.0

    # Pit Stop Strategy Window
    pit_window_start: int = 15
    pit_window_end: int = 40
    strategic_pit_reward: float = 30.0
    compound_change_reward: float = 100.0

    # Tyre Management Penalties
    tyre_age_limit: int = 30
    tyre_wear_limit: float = 0.8
    tyre_penalty: float = -50.0

    # Rule enforcement
    rule_penalty_threshold_one: int = 10
    rule_penalty_one_value: float = -500.0
    rule_penalty_threshold_two: int = 20
    rule_penalty_two_value: float = -200.0

    rule_penalty_violation: float = -3000.0

