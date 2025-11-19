
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from f1_env import F1OpponentEnv
from dynamics import TyreCompound, compounds

import wandb
from wandb.integration.sb3 import WandbCallback

def train_f1_agent(
        total_timesteps=1_000_000,
        buffer_size=300_000,
        learning_starts=100_000,
        batch_size=256,
        tau=0.005,
        target_update_interval=1000,
        gamma=0.995,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.4,
        exploration_final_eps=0.05,
        learning_rate=3e-4,
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
    model = DQN(
        "MlpPolicy",
        env,
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
        tensorboard_log=f"f1_gym/opponent/logs/tensorboard/{run.id}"
    )

    callback = WandbCallback(
        model_save_path=f"f1_gym/opponent/models/wandb/{run.id}",
        verbose=2
    )

    print("Starting Training")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback
    )

    model_name = f"f1_opponent.zip"
    model_path = os.path.join(MODEL_DIR, model_name)
    model.save(model_path)
    print(f"Saved model to {model_path}")
    
    run.finish()

    return model_path

def evaluate_model(model_path: str = "f1_gym/opponent/models/f1_opponent.zip", num_episodes: int = 5):
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    print(f"Loading model from {model_path}")
    model = DQN.load(model_path)
    env = F1OpponentEnv()
    
    print(f"\nEvaluating model over {num_episodes} episodes...\n")
    
    total_rewards = []
    total_positions = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False

        print(f"Episode {episode + 1}/{num_episodes}")
        
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
        print(f"\nEpisode {episode + 1} Summary:")
        print(f"  - Final Lap: {env.current_lap}")
        print(f"  - Total Time: {env.total_time:.2f}s")
        print(f"  - Total Pit Stops: {env.num_pit_stops}")
        print(f"  - Episode Reward: {episode_reward:.2f}")
        print(f"  - Final Position: {env.position}/20")
    
    print(f"Reward Summary:\n")
    print(f"Average Reward: {np.mean(total_rewards):.2f}")
    print(f"Std Dev Reward: {np.std(total_rewards):.2f}")
    print(f"Min Reward: {np.min(total_rewards):.2f}")
    print(f"Max Reward: {np.max(total_rewards):.2f}")

    print(f"Position Summary:\n")
    print(f"Average Position: {np.mean(total_positions):.2f}")
    print(f"Std Dev Position: {np.std(total_positions):.2f}")
    print(f"Min Position: {np.min(total_positions):.2f}")
    print(f"Max Position: {np.max(total_positions):.2f}")
    
    env.close()

def test_env():
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

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            print("Training Agent")
            model = train_f1_agent()

        elif sys.argv[1] == "evaluate":
            print("Evaluating Model")
            if len(sys.argv) > 2:
                model_path = sys.argv[2]
            else:
                model_path = "f1_gym/opponent/models/f1_opponent.zip"
            
            num_episodes = 10
            if len(sys.argv) > 3:
                num_episodes = int(sys.argv[3])
            
            evaluate_model(model_path=model_path, num_episodes=num_episodes)

        elif sys.argv[1] == "visualise":
            print("Visualising Results")
    
    else:
        print("Testing environment...\n")
        test_env()
        print("\nUsage:")
        print("python train.py train                          - Train the agent")
        print("python train.py evaluate [model] [episodes]    - Evaluate the agent")
        print("python train.py visualise                      - Visualise strategy (pit stops & tyres)")
        print("python train.py                                - Test environment")