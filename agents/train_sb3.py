
from stable_baselines3 import PPO
from env.f1_env import F1PitStopEnv
import pandas as pd
import numpy as np

def main():
    env = F1PitStopEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000)

    obs, info = env.reset()
    done = False

    print("\nRace Log\n")
    while not done:
        action, _ = model.predict(obs)

        if isinstance(action, np.ndarray):
            action = int(action.squeeze())
        else:
            action = int(action)

        obs, reward, done, truncated, info = env.step(action)
        env.loggeroutput()

    print("\nRace Summary\n")
    for lap in env.race_log:
        print(lap)

    pd.DataFrame(env.race_log).to_csv("race_log.csv", index=False)

    model.save("ppo_f1pitstop")

if __name__ == "__main__":
    main()