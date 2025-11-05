
from env.f1_env import F1PitStopEnv

def test_env_runs():
    env = F1PitStopEnv()
    obs, _ = env.reset()

    for _ in range (10):
        obs, reward, done, trunc, info = env.step(env.action_space.sample())
        if done:
            break
        
    assert obs is not None