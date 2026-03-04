import argparse
import json
import os
import joblib
from f1_gym.agents.train_ppo import train_f1_ppo, evaluate_ppo_model

def load_race_data(json_path: str) -> dict:
    with open(json_path, "r") as f:
        return json.load(f)

def load_predictor(race_name: str):
    filename = race_name.strip().replace(" ", "_").lower() + ".pkl"
    model_path = os.path.join("f1_gym", "models", "lap_predictors", filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No lap predictor found at {model_path}. "
            f"Train one first with: python f1_gym/lap_predictor.py"
        )
    return joblib.load(model_path)

def main():
    parser = argparse.ArgumentParser(description="F1 RL Real Race Environment")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train
    train_parser = subparsers.add_parser("train", help="Train PPO on a real race")
    train_parser.add_argument("--timesteps", type=int, default=1_500_000)
    train_parser.add_argument("--data", type=str, default="data/races/2024_Miami_Grand_Prix.json")
    train_parser.add_argument("--race", type=str, default="Miami Grand Prix")
    train_parser.add_argument("--n-envs", type=int, default=8)
    train_parser.add_argument("--seed", type=int, default=6)

    # Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate PPO on a real race")
    eval_parser.add_argument("--model", type=str, default="f1_gym/models/f1_rl_ppo_miami.zip")
    eval_parser.add_argument("--vecnormalize", type=str, default="f1_gym/models/f1_rl_ppo_miami_vecnormalize.pkl")
    eval_parser.add_argument("--episodes", type=int, default=100)
    eval_parser.add_argument("--data", type=str, default="data/races/2024_Miami_Grand_Prix.json")
    eval_parser.add_argument("--race", type=str, default="Miami Grand Prix")

    args = parser.parse_args()

    if args.command == "train":
        race_data = load_race_data(args.data)
        predictor = load_predictor(args.race)
        print(f"Training PPO (real) for {args.timesteps} timesteps on {args.race}...")
        train_f1_ppo(
            total_timesteps=args.timesteps,
            race_data=race_data,
            predictor=predictor,
        )

    elif args.command == "evaluate":
        race_data = load_race_data(args.data)
        predictor = load_predictor(args.race)
        evaluate_ppo_model(
            model_path=args.model,
            vecnormalize_path=args.vecnormalize,
            num_episodes=args.episodes,
            race_data=race_data,
            predictor=predictor,
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
