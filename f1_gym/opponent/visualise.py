import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
import pandas as pd
import os

from stable_baselines3 import DQN
from f1_env import F1OpponentEnv

PATH = "f1_gym/opponent/models/f1_opponent"
OUTPUT = "f1_gym/opponent/visualisations/race_summary.png"

COMPOUND_COLOURS = {
    1: '#F90429',
    2: '#FFD100',
    3: '#F0F0F0',
}
NAMES = {1: 'Soft', 2: 'Medium', 3: 'Hard'}

def run_race(env, model):
    obs, info = env.reset()
    done = False

    all_drivers_history = {'Agent': []}
    for opp in env.opponents:
        all_drivers_history[f'Opponent {opp.opponent_id}'] = []

    total_reward = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        total_reward += reward

        all_drivers_history['Agent'].append(env.current_compound)
        for opp in env.opponents:
            all_drivers_history[f'Opponent {opp.opponent_id}'].append(opp.current_compound)

    driver_times = [('Agent', env.total_time)]
    for opp in env.opponents:
        driver_times.append((f'Opponent {opp.opponent_id}', opp.total_time))

    driver_times.sort(key=lambda x: x[1])

    return{
        'history': all_drivers_history,
        'standings': driver_times,
        'position': env.position,
        'total_reward': total_reward,
        'laps': env.current_lap
    }

def plot_race(data, output_path):
    history = data['history']
    standings = data['standings']

    total_laps = len(history['Agent'])
    num_drivers = len(history)

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_facecolor('#333333')
    for y_pos, (driver_name, total_time) in enumerate(reversed(standings)):
        compounds_used = history[driver_name]
        xranges = []
        colors = []

        if not compounds_used:
            continue

        current_compound = compounds_used[0]
        current_start = 0

        for lap_id, comp in enumerate(compounds_used):
            if comp != current_compound:
                width = lap_id - current_start
                xranges.append((current_start + 1, width))
                colors.append(COMPOUND_COLOURS.get(current_compound, '#CCCCCC'))

                current_compound = comp
                current_start = lap_id

        width = len(compounds_used) - current_start
        xranges.append((current_start + 1, width))
        colors.append(COMPOUND_COLOURS.get(current_compound, '#CCCCCC'))

        ax.broken_barh(xranges, (y_pos - 0.4, 0.8), facecolors=colors, edgecolor='black', linewidth=0.5)
        ax.text(total_laps + 1, y_pos, f"{total_time:.2f}s", va='center', fontsize=14, color='white')

    ax.set_yticks(range(num_drivers))
    yticklabels = [f"P{i+1}: {d[0]}" for i, d in enumerate(standings)]
    ax.set_yticklabels(reversed(yticklabels))
    ax.set_xlabel("Lap")
    ax.set_title(f"Race Strategy Summary (Winner: {standings[0][0]})", fontsize=16)
    ax.set_xlim(0, total_laps + 5)
    ax.grid(axis='x', color='grey', linestyle=':', alpha=0.5)

    legend_patches = [
        mpatches.Patch(color=COMPOUND_COLOURS[1], label='Soft'),
        mpatches.Patch(color=COMPOUND_COLOURS[2], label='Medium'),
        mpatches.Patch(color=COMPOUND_COLOURS[3], label='Hard'),
    ]
    ax.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved race summary plot to {output_path}")
    plt.close()

def main():
    env = F1OpponentEnv()
    model = DQN.load(PATH)

    race_data = run_race(env, model)
    pos = race_data['position']
    time = race_data['standings'][0][1] if pos == 1 else 0
    print(f"Final Position: {pos}, Total Time: {time:.2f}s")

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    plot_race(race_data, OUTPUT)

if __name__ == "__main__":
    main()