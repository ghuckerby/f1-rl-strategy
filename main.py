import argparse
import sys
import os

from f1_gym.agents.train_dqn import evaluate_model, train_f1_agent, test_env
from f1_gym.agents.train_ppo import train_f1_ppo, evaluate_ppo_model
from f1_gym.visualisation.visualise import visualise_model
from f1_gym.visualisation.animate import animate_model

def main():
    parser = argparse.ArgumentParser(description="F1 RL Strategy Main Controller")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train DQN command
    train_parser = subparsers.add_parser("train-dqn", help="Train the RL agent with DQN")
    train_parser.add_argument("--timesteps", type=int, default=1_000_000, help="Total training timesteps")

    # Train PPO command
    train_ppo_parser = subparsers.add_parser("train-ppo", help="Train the RL agent with PPO")
    train_ppo_parser.add_argument("--timesteps", type=int, default=2_000_000, help="Total training timesteps")
    train_ppo_parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel environments")
    train_ppo_parser.add_argument("--seed", type=int, default=6, help="Random seed")

    # Evaluate DQN command
    eval_parser = subparsers.add_parser("evaluate-dqn", help="Evaluate a trained DQN model")
    eval_parser.add_argument("--model", type=str, default="f1_gym/models/f1_rl_dqn.zip", help="Path to the trained model")
    eval_parser.add_argument("--episodes", type=int, default=1000, help="Number of evaluation episodes")

    # Evaluate PPO command
    eval_ppo_parser = subparsers.add_parser("evaluate-ppo", help="Evaluate a trained PPO model")
    eval_ppo_parser.add_argument("--model", type=str, default="f1_gym/models/f1_rl_ppo.zip", help="Path to the trained model")
    eval_ppo_parser.add_argument("--vecnormalize", type=str, default="f1_gym/models/f1_rl_ppo_vecnormalize.pkl", 
                                  help="Path to VecNormalize stats")
    eval_ppo_parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")

    # Visualise DQN command
    vis_parser = subparsers.add_parser("visualise-dqn", help="Run a race and plot results (DQN)")
    vis_parser.add_argument("--model", type=str, default="f1_gym/models/f1_rl_dqn.zip", help="Path to the trained model")
    vis_parser.add_argument("--output", type=str, default="f1_gym/logs/race_summary.png", help="Output path for the plot")

    # Visualise PPO command
    vis_ppo_parser = subparsers.add_parser("visualise-ppo", help="Run a race and plot results (PPO)")
    vis_ppo_parser.add_argument("--model", type=str, default="f1_gym/models/f1_rl_ppo.zip", help="Path to the trained model")
    vis_ppo_parser.add_argument("--output", type=str, default="f1_gym/logs/race_summary_ppo.png", help="Output path for the plot")
    vis_ppo_parser.add_argument("--vecnormalize", type=str, default="f1_gym/models/f1_rl_ppo_vecnormalize.pkl", 
                                  help="Path to VecNormalize stats")
    
    # Animate PPO command
    animate_ppo_parser = subparsers.add_parser("animate-ppo", help="Animate a race (PPO)")
    animate_ppo_parser.add_argument("--model", type=str, default="f1_gym/models/f1_rl_ppo.zip", help="Path to the trained model")
    animate_ppo_parser.add_argument("--output", type=str, default="f1_gym/logs/race_animation_ppo.gif", help="Output path for the animation")
    animate_ppo_parser.add_argument("--vecnormalize", type=str, default="f1_gym/models/f1_rl_ppo_vecnormalize.pkl", 
                                  help="Path to VecNormalize stats")

    # Test environment command
    test_parser = subparsers.add_parser("test", help="Test the F1 environment")

    args = parser.parse_args()

    if args.command == "train-dqn":
        print(f"Training DQN model for {args.timesteps} timesteps...")
        train_f1_agent(total_timesteps=args.timesteps)

    elif args.command == "train-ppo":
        print(f"Training PPO model for {args.timesteps} timesteps...")
        train_f1_ppo(total_timesteps=args.timesteps)

    elif args.command == "evaluate-dqn":
        evaluate_model(model_path=args.model, num_episodes=args.episodes)

    elif args.command == "evaluate-ppo":
        evaluate_ppo_model(
            model_path=args.model,
            vecnormalize_path=args.vecnormalize,
            num_episodes=args.episodes,
        )

    elif args.command == "visualise-dqn":
        visualise_model(model_path=args.model, output_path=args.output, algo="dqn")

    elif args.command == "test":
        test_env()

    elif args.command == "visualise-ppo":
        visualise_model(model_path=args.model, output_path=args.output, algo="ppo", vecnormalize_path=args.vecnormalize)

    elif args.command == "animate-ppo":
        animate_model(model_path=args.model, output_path=args.output, vecnormalize_path=args.vecnormalize)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()