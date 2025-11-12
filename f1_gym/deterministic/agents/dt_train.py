
import os
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from f1_gym.deterministic.env.dt_f1_env import F1PitStopEnv
from f1_gym.deterministic.env.dt_dynamics import SOFT, MEDIUM, HARD
import pandas as pd
import numpy as np

def train(start_compound, compound_name):
    print(f"\nTraining {compound_name} as Starting Tire")

    # Log output folder
    LOG_DIR = "f1_gym/deterministic/dqn_logs"
    MODEL_DIR = "f1_gym/deterministic/dqn_models"
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    env = F1PitStopEnv(starting_compound=start_compound)

    # Model: Deep Q Network
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,  
        buffer_size=200_000,            # Replay buffer size
        learning_starts=50_000,         # Steps before learning starts
        batch_size=128,                 # Minibatch size
        target_update_interval=1000,    # Target network update frequency
        exploration_fraction=0.05,      # Fraction of total timesteps for exploration
        exploration_final_eps=0.05,     # Final epsilon for exploration
        learning_rate=0.0003,           # Learning rate
    )

    # Model learning
        # Time steps / Laps = Number of races (episodes)
    model.learn(total_timesteps=20_000_000)

    # Model output folder
    model_name = f"dqn_f1_{compound_name}_start.zip"
    model_path = os.path.join(MODEL_DIR, model_name)
    model.save(model_path)
    print(f"Saved model to {model_path}")

    return model_path

def main():
    # Train using different starting tyres
    # train(SOFT, "Soft")
    train(MEDIUM, "Medium")
    # train(HARD, "Hard")

    print("\nTraining complete.")

if __name__ == "__main__":
    main()