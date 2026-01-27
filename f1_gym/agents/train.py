
import numpy as np
import os
from stable_baselines3 import DQN
from f1_gym.envs.f1_env import F1OpponentEnv

import wandb
from wandb.integration.sb3 import WandbCallback

def train_f1_agent(
        # Time steps for training
        total_timesteps=1_000_000,
        # Buffer size, size of training memory
        buffer_size=200_000,
        # Steps before learning starts
        learning_starts=100_000,
        # Size of each training batch
        batch_size=32,
        # Soft update coefficient
        tau=0.005,
        # How often to update the target network
        target_update_interval=1000,
        # Discount factor
        gamma=0.995,
        # Frequency of training
        train_freq=4,
        # Number of gradient steps
        gradient_steps=1,
        # Fraction of exploration
        exploration_fraction=0.4,
        # Final epsilon for exploration
        exploration_final_eps=0.05,
        # Learning rate
        learning_rate=1e-4,
):
    """Train an RL agent to compete in the F1 opponent environment using a DQN algorithm"""
    
    # WandB Initialization for tracking training
    run = wandb.init(
        project="f1-rl",
        name=f"f1_rl_dqn_{total_timesteps//1_000_000}M",
        sync_tensorboard=True,
        reinit=True,
    )

    LOG_DIR = "f1_gym/logs"
    MODEL_DIR = "f1_gym/models"
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

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
        tensorboard_log=f"f1_gym/logs/tensorboard/{run.id}"
    )

    callback = WandbCallback(
        model_save_path=f"f1_gym/models/wandb/{run.id}",
        verbose=2
    )

    print("Starting Training")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback
    )

    model_name = f"f1_rl_dqn.zip"
    model_path = os.path.join(MODEL_DIR, model_name)
    model.save(model_path)
    print(f"Saved model to {model_path}")
    
    run.finish()

    return model_path

def evaluate_model(model_path: str = "f1_gym/models/f1_rl_dqn.zip", num_episodes: int = 5):
    """Evaluate a trained model over a number of episodes"""

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

        print(f"\nEpisode {episode + 1} completed.")
    
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

def test_env():
    """Simple test of the F1OpponentEnv environment"""
    
    print("Testing F1 Environment")
    env = F1OpponentEnv()
    obs, _ = env.reset()

    print(f"Initial obs shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    print("Running with 50 Random Actions")
    total_reward = 0
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if (step + 1) % 10 == 0:
            print(f"Step {step + 1}: Position {env.position}, "
                  f"Lap {env.current_lap}, Accumulated Reward: {total_reward:.2f}")
            
        if terminated:
            print(f"Race finished, Final Position: {env.position}/20")
            print(f"Final Cumulative reward: {total_reward:.2f}")
            break