from dataclasses import dataclass

@dataclass
class RewardConfig:
    """Config for F1 Environment Reward Function."""

    # Base Rewards
    lap_time_reward_weight: float = 1.0
    time_benchmark = 95.0

    # Position Rewards
    position_gain_reward: float = 10.0

    # Tyre Wear Penalty
    tyre_wear_threshold: float = 0.5
    tyre_wear_penalty: float = -10.0

    # Rule penalty big enough to offset any gains
    rule_penalty_violation: float = -2000.0

