
import numpy as np
import os
from stable_baselines3 import DQN
from env.f1_opponent_env import F1OpponentEnv

import wandb
from wandb.integration.sb3 import WandbCallback

def train_f1_agent(
        total_timesteps=1_000_000,
        buffer_size=200_000,
        learning_starts=100_000,
        batch_size=128,
        tau=0.005,
        target_update_interval=1000,
        gamma=0.995,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.4,
        exploration_final_eps=0.05,
        learning_rate=1e-4,
):
    """Train a DQN agent to compete in the F1 opponent environment."""
    
    # Training Logging
    run = wandb.init(
        project="f1-rl-opponent",
        name=f"f1_rl_opponent_dqn_{total_timesteps//1_000_000}M",
        sync_tensorboard=True,
        reinit=True,
    )
    LOG_DIR = "logs"
    MODEL_DIR = "models"
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Environment and Model Setup
    env = F1OpponentEnv()
    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs={"net_arch": [256, 256]},
        verbose=1,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        learning_rate=learning_rate,
        tau=tau,
        gamma=gamma,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        tensorboard_log=f"logs/tensorboard/{run.id}"
    )

    # WandB Callback for logging
    callback = WandbCallback(
        model_save_path=f"models/wandb/{run.id}",
        verbose=2
    )

    # Model training
    print("Starting Training")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback
    )

    # Save model
    model_name = f"f1_rl_opponent_dqn.zip"
    model_path = os.path.join(MODEL_DIR, model_name)
    model.save(model_path)
    print(f"Saved model to {model_path}")
    
    run.finish()

    return model_path

def evaluate_model(model_path: str="models/f1_rl_opponent_dqn.zip", num_episodes: int = 5):
    """Evaluate a trained DQN agent in the F1 opponent environment."""

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    print(f"Loading model from {model_path}")
    model = DQN.load(model_path)
    env = F1OpponentEnv()
    
    # Metrics for evaluation
    total_rewards = []
    total_positions = []
    
    # Evaluation loop
    print(f"\nEvaluating model over {num_episodes} episodes...\n")
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False

        step = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            episode_reward += reward
            done = terminated or truncated
            env.logger_output()
            step += 1
        
        total_rewards.append(episode_reward)
        total_positions.append(env.position)

        print(f"\nEpisode {episode + 1} completed. Reward: {episode_reward:.2f}, Final Position: {env.position}/20\n")

    # Summary of results
    print(f"\nReward Summary:\n")
    print(f"Average Reward: {np.mean(total_rewards):.2f}")
    print(f"Std Dev Reward: {np.std(total_rewards):.2f}")
    print(f"Min Reward: {np.min(total_rewards):.2f}")
    print(f"Max Reward: {np.max(total_rewards):.2f}")

    print(f"\nPosition Summary:\n")
    print(f"Average Position: {np.mean(total_positions):.2f}")
    print(f"Std Dev Position: {np.std(total_positions):.2f}")
    print(f"Min Position: {np.min(total_positions):.2f}")
    print(f"Max Position: {np.max(total_positions):.2f}")
    
    env.close()