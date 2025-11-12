import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from stable_baselines3 import DQN
from stable_baselines3 import PPO
from f1_gym.deterministic.env.dt_f1_env import F1PitStopEnv
from f1_gym.deterministic.env.dt_dynamics import SOFT, MEDIUM, HARD

def plot_results(race_log, save_path, compound_name):
    compound_colours = {
        SOFT: '#D90429',
        MEDIUM: '#FFD100',
        HARD: '#F0F0F0'
    }
    
    colours = race_log['compound'].map(compound_colours)
    pit_laps = race_log[race_log['pitted'] == True]['lap']
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.bar(
        race_log['lap'],
        race_log['lap_time'],
        color=colours,
        edgecolor='black',
        linewidth=0.5
    )

    for i, lap in enumerate(pit_laps):
        ax.axvline(
            x=lap,
            color='blue',
            linestyle='--',
            linewidth=1.5,
            label='Pit Stop' if i == 0 else ""
        )

    ax.set_title(f'Strategy plot for {compound_name} start.', fontsize=16)
    ax.set_xlabel('Lap Number', fontsize=14)
    ax.set_ylabel('Lap Time (s)', fontsize=14)
    ax.set_xticks(range(0, int(race_log['lap'].max()) + 2, 2))
    ax.set_xlim(0.5, race_log['lap'].max() + 0.5)

    legend_elements = [
        Patch(facecolor=compound_colours[SOFT], edgecolor='black', label='Soft Tyre'),
        Patch(facecolor=compound_colours[MEDIUM], edgecolor='black', label='Medium Tyre'),
        Patch(facecolor=compound_colours[HARD], edgecolor='black', label='Hard Tyre'),
    ]

    if not pit_laps.empty:
        handles, labels = ax.get_legend_handles_labels()
        legend_elements.append(handles[0])

    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    ax.grid(axis='y', linestyle=':', color='grey', alpha=0.7)

    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved race plot to {save_path}")

def run_model(start_compound, compound_name):
    print(f"\nEvaluating {compound_name} as Starting Tire")

    # Log output folder
    LOG_DIR = "f1_gym/deterministic/dqn_logs"
    os.makedirs(LOG_DIR, exist_ok=True)
    MODEL_DIR = "f1_gym/deterministic/dqn_models"
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    model_name = f"dqn_f1_{compound_name}_start.zip"
    model_path = os.path.join(MODEL_DIR, model_name)

    if not os.path.exists(model_path):
        print(f"Model file {model_path} does not exist. Train model first.")
        return None
    
    env = F1PitStopEnv(starting_compound=start_compound)
    model = DQN.load(model_path, env=env)
    obs, info = env.reset()
    done = False

    print("\nRace Log\n")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action))
        env.loggeroutput()

    final_time = env.total_time
    print(f"\nFinal race time for {compound_name}: {final_time:.2f}s")

    log_name = f"race_log_{compound_name}.csv"
    log_path = os.path.join(LOG_DIR, log_name)
    plot_name = f"race_plot_{compound_name}.png"
    plot_path = os.path.join(LOG_DIR, plot_name)

    race_df = pd.DataFrame(env.race_log)
    race_df.to_csv(log_path, index=False)
    plot_results(race_df, plot_path, compound_name)

    return final_time

def main():

    time_M = run_model(MEDIUM, "Medium")
    time_S = run_model(SOFT, "Soft")
    time_H = run_model(HARD, "Hard")

    if all(t is not None for t in [time_S, time_M, time_H]):
        results = {
            "Soft": time_S,
            "Medium": time_M,
            "Hard": time_H
        }
        best_strategy = min(results, key=results.get)
        print(f"\n Best Overall: Start on {best_strategy} (Time : {results[best_strategy]:.2f}s)")
    else:
        print("\nEnsure all models are trained before evaluation.")

if __name__ == "__main__":
    main()