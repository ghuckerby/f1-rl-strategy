
from stable_baselines3 import PPO
from env.f1_env import F1PitStopEnv
from env.dynamics import SOFT, MEDIUM, HARD
import pandas as pd
import numpy as np

def train(start_compound, compound_name):
    print(f"\nTraining {compound_name} as Starting Tire")

    env = F1PitStopEnv(starting_compound=start_compound)

    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=10_000_000)
    model.save(f"ppo_f1_{compound_name}_start")

    obs, info = env.reset()
    done = False

    print("\nRace Log\n")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
        env.loggeroutput()

    final_time = env.total_time
    print(f"\nFinal Race time for {compound_name}: {final_time:.2f}s")
    pd.DataFrame(env.race_log).to_csv(f"race_log_{compound_name}.csv", index=False)
    return final_time

def main():
    time_S = train(SOFT, "Soft")
    time_M = train(MEDIUM, "Medium")
    time_H = train(HARD, "Hard")

    print("\nResults Comparison:\n")
    print(f"Soft Start Time: {time_S:.2f}s")
    print(f"Medium Start Time: {time_M:.2f}s")
    print(f"Hard Start Time: {time_H:.2f}s")

    results = {
        "Soft": time_S,
        "Medium": time_M,
        "Hard": time_H
    }
    best_strategy = min(results, key=results.get)
    print(f"\n Best Overall: Start on {best_strategy} (Time : {results[best_strategy]:.2f}s)")

if __name__ == "__main__":
    main()