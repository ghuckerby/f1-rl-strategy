import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
import pandas as pd
import os
import random
import time

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, VecEnv
from f1_gym.envs.f1_env import F1OpponentEnv

PATH = "f1_gym/models/f1_rl_dqn.zip"
OUTPUT = "f1_gym/logs/race_summary.png"

COMPOUND_COLOURS = {
    1: '#F90429',
    2: '#FFD100',
    3: '#F0F0F0',
}
NAMES = {1: 'Soft', 2: 'Medium', 3: 'Hard'}

def run_race(env, model):
    base_env = env
    normalizer = None
    
    # Handle VecNormalize wrapper
    if isinstance(env, VecNormalize):
        normalizer = env
        env = env.venv

    # Seeding for random safety car events
    if isinstance(env, VecEnv):
        base_env = env.envs[0]
        seed = int(time.time() * 1000) % 2**32
        random.seed(seed)
        np.random.seed(seed)
        
        obs, info = base_env.reset()
    else:
        seed = int(time.time() * 1000) % 2**32
        random.seed(seed)
        np.random.seed(seed)
        
        obs, info = env.reset()
        base_env = env

    done = False
    
    all_drivers_history = {'Agent': []}
    pit_stops = {'Agent': []}
    for opp in base_env.opponents:
        all_drivers_history[f'Opponent {opp.opponent_id}'] = []
        pit_stops[f'Opponent {opp.opponent_id}'] = []
    
    sc_history = []

    total_reward = 0
    while not done:

        model_obs = obs
        
        if normalizer:
            model_obs = normalizer.normalize_obs(obs)
            
        action, _states = model.predict(model_obs, deterministic=True)
        obs, reward, terminated, truncated, info = base_env.step(int(action))
        done = terminated or truncated

        total_reward += reward

        all_drivers_history['Agent'].append(base_env.current_compound)
        if base_env.tyre_age == 1 and base_env.current_lap > 1:
            pit_stops['Agent'].append(len(all_drivers_history['Agent']) - 1)

        for opp in base_env.opponents:
            all_drivers_history[f'Opponent {opp.opponent_id}'].append(opp.current_compound)
            if opp.tyre_age == 1 and opp.current_lap > 1:
                pit_stops[f'Opponent {opp.opponent_id}'].append(len(all_drivers_history[f'Opponent {opp.opponent_id}']) - 1)
        
        sc_history.append(info.get('sc_active', False))

    driver_times = [('Agent', base_env.total_time)]
    for opp in base_env.opponents:
        driver_times.append((f'Opponent {opp.opponent_id}', opp.total_time))

    driver_times.sort(key=lambda x: x[1])

    return{
        'history': all_drivers_history,
        'pit_stops': pit_stops,
        'sc_history': sc_history,
        'standings': driver_times,
        'position': base_env.position,
        'total_reward': total_reward,
        'laps': base_env.current_lap
    }

def plot_race(data, output_path):
    history = data['history']
    pit_stops = data.get('pit_stops', {})
    sc_history = data.get('sc_history', [])
    standings = data['standings']

    total_laps = len(history['Agent'])
    num_drivers = len(history)

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_facecolor('#333333')

    # Safety Car windows
    if sc_history:
        current_start = -1
        for i, is_sc in enumerate(sc_history):
            if is_sc and current_start == -1:
                current_start = i
            elif not is_sc and current_start != -1:
                ax.axvspan(current_start + 1, i + 1, color='#cf5a00', alpha=1, lw=0)
                current_start = -1
        if current_start != -1:
            ax.axvspan(current_start + 1, len(sc_history) + 1, color='#cf5a00', alpha=1, lw=0)

    for y_pos, (driver_name, total_time) in enumerate(reversed(standings)):
        compounds_used = history[driver_name]
        driver_pits = pit_stops.get(driver_name, [])
        xranges = []
        colors = []

        if not compounds_used:
            continue

        current_compound = compounds_used[0]
        current_start = 0

        for lap_id, comp in enumerate(compounds_used):
            is_pit = lap_id in driver_pits
            if comp != current_compound or is_pit:
                width = lap_id - current_start
                xranges.append((current_start + 1, width))
                colors.append(COMPOUND_COLOURS.get(current_compound, '#CCCCCC'))

                current_compound = comp
                current_start = lap_id

        width = len(compounds_used) - current_start
        xranges.append((current_start + 1, width))
        colors.append(COMPOUND_COLOURS.get(current_compound, '#CCCCCC'))

        ax.broken_barh(xranges, (y_pos - 0.4, 0.8), facecolors=colors, edgecolor='black', linewidth=1.0)
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
        mpatches.Patch(color='#cf5a00', alpha=1, label='Safety Car'),
    ]
    ax.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(0.5, -0.05), ncol=4, frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved race summary plot to {output_path}")
    plt.close()

def visualise_model(model_path, output_path, algo="dqn", vecnormalize_path=None):

    random.seed()
    np.random.seed()

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    base_env = F1OpponentEnv()
    
    if algo.lower() == "ppo":
        model = PPO.load(model_path)
        env = DummyVecEnv([lambda: base_env])
        if vecnormalize_path and os.path.exists(vecnormalize_path):
            env = VecNormalize.load(vecnormalize_path, env)
            env.training = False
            env.norm_reward = False
        else:
            print("Warning: Running without normalisation.")
    elif algo.lower() == "dqn":
        model = DQN.load(model_path)
        env = base_env
    else:
        print(f"Error: Unknown algorithm {algo}")
        return

    race_data = run_race(env, model)
    pos = race_data['position']
    time = race_data['standings'][0][1] if pos == 1 else 0
    print(f"Final Position: {pos}, Total Time: {time:.2f}s")
    
    if isinstance(env, VecEnv):
        env.close()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plot_race(race_data, output_path)

if __name__ == "__main__":
    visualise_model("f1_gym/models/f1_rl_dqn.zip", "f1_gym/logs/race_summary.png")