import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation

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

    standings_history = []
    
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

        current_lap_standings = [('Agent', base_env.total_time)]
        for opp in base_env.opponents:
            current_lap_standings.append((f'Opponent {opp.opponent_id}', opp.total_time))

        current_lap_standings.sort(key=lambda x: x[1])
        standings_history.append(current_lap_standings)

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
        'laps': base_env.current_lap,
        'standings_history': standings_history
    }

def animate_race(data, output_path):
    history = data['history']
    pit_stops = data['pit_stops']
    sc_history = data['sc_history']
    standings_history = data['standings_history']
    total_laps = data['laps']

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_facecolor('#333333')

    def update(frame):
        ax.clear()
        ax.set_facecolor('#333333')
        current_lap = frame + 1
        current_standings = standings_history[frame]
        leader_time = current_standings[0][1]

        # Safety Car Background
        for i in range(current_lap):
            if sc_history[i]:
                ax.axvspan(i + 1, i + 2, color='#cf5a00', alpha=0.3, lw=0)

        # Driver Bars
        for y_pos, (driver_name, total_time) in enumerate(reversed(current_standings)):
            compounds_used = history[driver_name][:current_lap]
            driver_pits = [p for p in pit_stops.get(driver_name, []) if p < current_lap]

            current_start = 0
            for lap_id in range(len(compounds_used)):
                if lap_id in driver_pits or (lap_id > 0 and compounds_used[lap_id] != compounds_used[lap_id - 1]):
                    width = lap_id - current_start
                    color = COMPOUND_COLOURS.get(compounds_used[current_start], '#CCCCCC')
                    ax.broken_barh([(current_start + 1, width)], (y_pos - 0.4, 0.8), facecolors=color, edgecolor='black')
                    current_start = lap_id

            width = len(compounds_used) - current_start
            color = COMPOUND_COLOURS.get(compounds_used[current_start], '#CCCCCC')
            ax.broken_barh([(current_start + 1, width)], (y_pos - 0.4, 0.8), facecolors=color, edgecolor='black')

            gap = total_time - leader_time
            label = f"{total_time:.2f}s" if y_pos == len(current_standings) - 1 else f"+{gap:.2f}s"
            ax.text(current_lap + 1, y_pos, label, va='center', fontsize=14, color='white')

        ax.set_yticks(range(len(current_standings)))
        ax.set_yticklabels([f"P{i+1}: {d[0]}" for i, d in enumerate(current_standings)][::-1])
        ax.set_title(f"Race Replay - Lap {current_lap}/{total_laps}", fontsize=16, color='white')
        ax.set_xlim(0, total_laps + 5)
    
    ani = FuncAnimation(fig, update, frames=total_laps, repeat=False)
    ani.save(output_path, writer='pillow', fps=2)
    plt.close()

def animate_model(model_path, output_path, vecnormalize_path=None):
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        return
    
    base_env = F1OpponentEnv()

    model = PPO.load(model_path)
    env = DummyVecEnv([lambda: base_env])
    if vecnormalize_path and os.path.exists(vecnormalize_path):
        env = VecNormalize.load(vecnormalize_path, env)
        env.training = False
        env.norm_reward = False
    else:
        print("Warning: Running without normalisation.")

    race_data = run_race(env, model)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    animate_race(race_data, output_path)