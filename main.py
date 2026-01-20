import argparse
import sys
import os

from f1_gym.agents.train import evaluate_model, train_f1_agent, test_env
from f1_gym.visualisation.visualise import visualise_model

def main():
    parser = argparse.ArgumentParser(description="F1 RL Strategy Main Controller")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the RL agent")
    train_parser.add_argument("--timesteps", type=int, default=1_000_000, help="Total training timesteps")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument("--model", type=str, default="f1_gym/models/f1_rl_dqn.zip", help="Path to the trained model")
    eval_parser.add_argument("--episodes", type=int, default=1000, help="Number of evaluation episodes")

    # Visualise command
    vis_parser = subparsers.add_parser("visualise", help="Run a race and plot results")
    vis_parser.add_argument("--model", type=str, default="f1_gym/models/f1_rl_dqn.zip", help="Path to the trained model")
    vis_parser.add_argument("--output", type=str, default="f1_gym/logs/race_summary.png", help="Output path for the plot")

    # Test environment command
    test_parser = subparsers.add_parser("test", help="Test the F1 environment")

    args = parser.parse_args()

    if args.command == "train":
        print(f"Training model for {args.timesteps} timesteps...")
        train_f1_agent(total_timesteps=args.timesteps)

    elif args.command == "evaluate":
        evaluate_model(model_path=args.model, num_episodes=args.episodes)

    elif args.command == "visualise":
        visualise_model(model_path=args.model, output_path=args.output)

    elif args.command == "test":
        test_env()

    else:
        parser.print_help()

if __name__ == "__main__":
    main()