from dataclasses import dataclass

@dataclass
class RewardConfig:

    # Base Rewards
    lap_time_reward_weight: float = 0.2
    time_benchmark = 95.0

    # Position Rewards
    position_gain_reward: float = 1.0

    # Rule penalty (at least 2 compounds must be used)
    rule_violation_penalty: float = -2000