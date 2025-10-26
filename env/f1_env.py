
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dataclasses import dataclass
from .dynamics import (
    SOFT, MEDIUM, HARD, calculate_lap_time, TrackParms
)



class F1PitStopEnv(gym.Env):

    def __init__(self):
        super().__init__()

        # Action Space
        # Discrete actions: stay out, pit soft, pit med, pit hard
        self.action_space = spaces.Discrete(4)

        # Observation Space
        self.observation_space = spaces.Box(
            low=0,              # min value for each element
            high=1,             # max value for each element                    
            shape=(4,),   # 4 values in obs vector
            dtype=np.float32    # use 32-bit floats
        )
        
        self.state = np.zeros(4)    # vector for obs space
        self.current_lap = 0        # track current lap
        self.max_laps = 50          # set race length

    def laptime(self, compound, stint_laps):
        # calculate lap time

        return compound

    # Called at the start of each new episode (race)
    def reset(self, *, seed = None, options = None):

        super().reset(seed=seed, options=options)
        self.current_lap = 0
        self.state = np.zeros(4)
        return self.state, {}
    
    # Advances environment by one time step, given an action from agent
    def step(self, action):

        self.current_lap += 1
        done = self.current_lap >= self.max_laps
        reward = np.random.randn() # Placeholder random reward
        self.state = np.random.rand(4) # Placeholder random state
        return self.state, reward, done, False, {}
