import argparse
import glob
import json
import os
import sys
import joblib
import numpy as np

from f1_gym.agents.train_ppo import train_f1_ppo, evaluate_ppo_model

MODEL_DIR = "f1_gym/models"
PREDICTOR_DIR = os.path.join(MODEL_DIR, "lap_predictors")
TRAIN_DIR = "data/training_races"
TEST_DIR = "data/test_races"

# Helper for loading races
def load_races(directory: str):
    if not os.path.isdir(directory):
        print(f"ERROR: directory not found: {directory}")
        sys.exit(1)

    race_files = sorted(glob.glob(os.path.join(directory, "*.json")))
    if not race_files:
        print(f"ERROR: no JSON files found in {directory}")
        sys.exit(1)

    race_configs, predictors = [], []
    for race in race_files:
        with open(race, "r") as f:
            race_data = json.load(f)

        race_name = race_data.get("name", "")
        file_name = race_name.strip().replace(" ", "_").lower() + ".pkl"
        pred_path = os.path.join(PREDICTOR_DIR, file_name)

        if not os.path.exists(pred_path):
            print(f"Skipping {race_name} - no lap predictor at {pred_path}")
            continue

        predictor = joblib.load(pred_path)
        race_configs.append(race_data)
        predictors.append(predictor)
        print(f"Loaded {race_name} ({race_data['track']['total_laps']} laps)")

    return race_configs, predictors

# Train
def train(args):
    train_configs, train_predictors = load_races(args.train_dir)
    train_f1_ppo(
        total_timesteps=args.timesteps, 
        race_configs=train_configs, 
        predictors=train_predictors,
        model_name=args.model_name,
        n_envs=args.n_envs,
        seed=args.seed,
    )

# Evaluate
def evaluate(args):
    test_configs, test_predictors = load_races(args.test_dir)

    all_results = []
    target_stats = []

    for race_data, predictor in zip(test_configs, test_predictors):

        # Per Race Evaluation
        print(f"\nEvaluating on {race_data['name']}.")
        race_results = evaluate_ppo_model(
            model_path=args.model,
            vecnormalize_path=args.vecnormalize,
            num_episodes=args.episodes,
            race_data=race_data,
            predictor=predictor,
        )
        all_results.append((race_data["name"], race_results))

        # Target driver stats for comparison
        target = race_data.get("target_driver_strategy", {})
        target_stats.append({
            "race": race_data["name"],
            "code": target.get("driver_code", "TGT"),
            "position": target.get("finishing_position", 20),
            "total_time": target.get("total_time", 0),
            "pit_stops": target.get("num_pit_stops", 0)
        })

    # Summary across test races
    target_code = target_stats[0]["code"] if target_stats else "TGT"

# Command line interface
def main():
    parser = argparse.ArgumentParser(description="Reinforcement Learning for F1 Pit Stop strategy - Season Train & Test.")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands: train or evaluate")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a PPO agent on the specified training races.")
    train_parser.add_argument("--train-dir", type=str, default=TRAIN_DIR, help="Directory containing training race JSON files.")
    train_parser.add_argument("--timesteps", type=int, default=3_000_000)
    train_parser.add_argument("--n-envs", type=int, default=8)
    train_parser.add_argument("--model-name", type=str, default="f1_rl_ppo_2024_season_1")
    train_parser.add_argument("--seed", type=int, default=6)

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained PPO model on the specified test races.")
    eval_parser.add_argument("--test-dir", type=str, default=TEST_DIR, help="Directory containing test race JSON files.")
    eval_parser.add_argument("--model", type=str, default="f1_gym/models/f1_rl_ppo_2024_season_1.zip", help="Path to the trained PPO model file.")
    eval_parser.add_argument("--vecnormalize", type=str, default="f1_gym/models/f1_rl_ppo_2024_season_1_vecnormalize.pkl", help="Path to the VecNormalize statistics file.")
    eval_parser.add_argument("--episodes", type=int, default=1)
    eval_parser.add_argument("--seed", type=int, default=6)

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()