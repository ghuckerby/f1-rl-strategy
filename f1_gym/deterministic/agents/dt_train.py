
import os
from stable_baselines3 import DQN
from f1_gym.deterministic.env.dt_f1_env import F1PitStopEnv
from f1_gym.deterministic.env.dt_dynamics import SOFT, MEDIUM, HARD
import pandas as pd
import numpy as np

def train(start_compound, compound_name):
    print(f"\nTraining {compound_name} as Starting Tire")

    LOG_DIR = "f1_gym/deterministic/logs"
    MODEL_DIR = "f1_gym/deterministic/dqn_models"
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    env = F1PitStopEnv(starting_compound=start_compound)

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        buffer_size=500_000,
        learning_starts=10_000,
        batch_size=128,
        target_update_interval=1000,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
    )

    model.learn(total_timesteps=10_000_000)

    model_name = f"dqn_f1_{compound_name}_start.zip"
    model_path = os.path.join(MODEL_DIR, model_name)
    model.save(model_path)

    obs, info = env.reset()
    done = False

    print("\nRace Log\n")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
        env.loggeroutput()

    final_time = env.total_time
    print(f"\nFinal Race time for {compound_name}: {final_time:.2f}s")

    log_name = f"race_log_{compound_name}.csv"
    log_path = os.path.join(LOG_DIR, log_name)
    pd.DataFrame(env.race_log).to_csv(log_path, index=False)

    return final_time

def main():
    time_M = train(MEDIUM, "Medium")
    print(f"Medium Time: {time_M:.2f}s")

    time_S = train(SOFT, "Soft")
    print(f"Soft Time: {time_S:.2f}s")
    
    time_H = train(HARD, "Hard")
    print(f"Hard Time: {time_H:.2f}s")

    results = {
        "Soft": time_S,
        "Medium": time_M,
        "Hard": time_H
    }
    best_strategy = min(results, key=results.get)
    print(f"\n Best Overall: Start on {best_strategy} (Time : {results[best_strategy]:.2f}s)")

if __name__ == "__main__":
    main()