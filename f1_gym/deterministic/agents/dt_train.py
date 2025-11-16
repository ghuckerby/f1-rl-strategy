
import os
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from f1_gym.deterministic.env.dt_f1_env import F1PitStopEnv
from f1_gym.deterministic.env.dt_dynamics import SOFT, MEDIUM, HARD
import pandas as pd
import numpy as np

import wandb
from wandb.integration.sb3 import WandbCallback

def train(start_compound, compound_name):

    run = wandb.init(
        project="f1-deterministic-dqn",
        name=f"dqn_f1_{compound_name}_start",
        sync_tensorboard=True,
        reinit=True, # Allow multiple runs in same script
    )

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
        buffer_size=500_000,            # Replay buffer size
        learning_starts=50_000,         # Steps before learning starts
        batch_size=256,                 # Samples from buffer before each update
        target_update_interval=1000,    # Target network update frequency
        exploration_fraction=0.05,      # Fraction of total timesteps for exploration
        exploration_final_eps=0.05,     # Final epsilon for exploration
        learning_rate=0.0002,           # Learning rate
        tensorboard_log=f"f1_gym/deterministic/dqn_logs/tensorboard/{run.id}"
    )

    # WandB Callback
    callback = WandbCallback(
        model_save_path="f1_gym/deterministic/dqn_models/wandb/{run.id}",
        verbose=2
    )

    # Model learning
        # Time steps / Laps = Number of races (episodes)
    model.learn(total_timesteps=1_000_000, callback=callback)

    # Model output folder
    model_name = f"dqn_f1_{compound_name}_start.zip"
    model_path = os.path.join(MODEL_DIR, model_name)
    model.save(model_path)
    print(f"Saved model to {model_path}")

    run.finish() # finish wandb run

    return model_path

def main():
    # Train using different starting tyres
    train(SOFT, "Soft")
    # train(MEDIUM, "Medium")
    # train(HARD, "Hard")

    print("\nTraining complete.")

if __name__ == "__main__":
    main()