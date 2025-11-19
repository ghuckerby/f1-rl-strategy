
import numpy as np
import os
from stable_baselines3 import PPO
from f1_env import F1OpponentEnv
from dynamics import TyreCompound, compounds

import wandb
from wandb.integration.sb3 import WandbCallback

def train_f1_agent(
        total_timesteps = 1_000_000,
        learning_rate=3e-4,
        n_steps = 2048,
        batch_size = 64,
        n_epochs = 10,
        gae_lambda = 0.95,
):
    
    run = wandb.init(
        project="f1-opponent",
        name=f"f1_opponent_rl",
        sync_tensorboard=True,
        reinit=True,
    )

    LOG_DIR = "f1_gym/opponent/logs"
    MODEL_DIR = "f1_gym/opponent/models"
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    env = F1OpponentEnv()
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gae_lambda=gae_lambda,
        verbose=1,
        tensorboard_log=LOG_DIR,
    )

    callback = WandbCallback(
        model_save_path="f1_gym/opponent/models/wandb/{run.id}",
        verbose=2
    )

    print("Starting Training")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )

    model_name = f"f1_opponent.zip"
    model_path = os.path.join(MODEL_DIR, model_name)
    model.save(model_path)
    print(f"Saved model to {model_path}")
    
    run.finish()

    return model_path

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            print("Training Agent")
            model = train_f1_agent(
                total_timesteps=1_000_000,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10
            )

        elif sys.argv[1] == "evaluate":
            print("Evaluating Model")

        elif sys.argv[2] == "visualise":
            print("Visualising Results")
    
    else:
        print("Usage:")
        print("python train.py train               - Train the agent")
        print("python train.py evaluate [model]    - Evaluate the agent")
        print("python train.py visualise [model]   - Visualise strategy (pit stops & tyres)")
        print("python train.py                     - Test environment")
