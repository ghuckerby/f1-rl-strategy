import gymnasium as gym
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from f1_gym.env.f1_real_env import F1RealEnv
from f1_gym.reward_config import RewardConfig


class MultiRaceEnv(gym.Env):
    """Wrapper that samples from multiple race environments
    
    Each episodes is one run on an F1RealEnv selected at each reset
    """

    def __init__( self, race_configs: List[Dict[str, Any]], predictors: List[Any], reward_config: Optional[RewardConfig] = None):
        """Initialise multi-race environment"
        
        Args:
            race_configs: list of race configurations
            predictors: list of lap-time predictors for each race
            reward_config: reward configuration
        """

        super().__init__()

        assert len(race_configs) == len(predictors), (f"Got {len(race_configs)} race configs but {len(predictors)} predictors")
        assert len(race_configs) > 0, "Need at least one race config"

        self._race_configs = race_configs
        self._predictors = predictors
        self._reward_config = reward_config

        # Create all inner F1RealEnv
        self._envs: List[F1RealEnv] = [
            F1RealEnv(rc, reward_config=reward_config, predictor=pred)
            for rc, pred in zip(race_configs, predictors)
        ]

        # Track current environment
        self._current_idx: int = 0
        self._current_env: F1RealEnv = self._envs[0]

        # Shared observation and action space for all environments
        self.observation_space = self._current_env.observation_space
        self.action_space = self._current_env.action_space

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Starts a new episode by randomly selecting a race environment"""

        super().reset(seed=seed, options=options)

        # Pick a random race for this episode
        self._current_idx = int(self.np_random.integers(len(self._envs)))
        self._current_env = self._envs[self._current_idx]

        return self._current_env.reset(seed=seed, options=options)

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        return self._current_env.step(action)

    @property
    def current_race_name(self) -> str:
        """Name of the race currently active."""
        return self._race_configs[self._current_idx].get("name", "Unknown")

    @property
    def num_races(self) -> int:
        return len(self._envs)
