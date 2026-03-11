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

    agent_positions = [r["positions"][0] for _, r in all_results]
    agent_rewards = [r["rewards"][0] for _, r in all_results]
    agent_times = [r["total_times"][0] for _, r in all_results]
    agent_pit_stops = [r["pit_stops"][0] for _, r in all_results]
    agent_compounds = [r["compounds_used"][0] for _, r in all_results]
    agent_lap_times = [r["lap_times"][0] for _, r in all_results]

    target_positions = [t["position"] for t in target_stats]
    target_times = [t["total_time"] for t in target_stats]
    target_pit_stops = [t["pit_stops"] for t in target_stats]

    print("\n" + "=" * 74)
    print(f"  Overall Evaluation Summary - From {len(all_results)} Test Races")
    print("=" * 74)

    # Race breakdown table
    print(f"\n  {'Race':<30} | {'Agent':>6} | {target_code:>6} | {'Delta Pos':>6} | {'Delta Time':>10}")
    print(f"  {'─' * 76}")
    position_deltas = []
    time_deltas = []
    for i, (name, _) in enumerate(all_results):

        # Position Comparison
        agent_position = agent_positions[i]
        target_position = target_positions[i]
        delta_position = agent_position - target_position
        position_deltas.append(delta_position)

        # Time Comparison
        agent_time = agent_times[i]
        target_time = target_times[i]
        delta_time = agent_time - target_time
        time_deltas.append(delta_time)

        print(f"  {name:<30} | P{agent_position:<5} | P{target_position:<5} | {delta_position:>+9} | {delta_time:>+10.2f}s")
    print(f"  {'─' * 76}")

    # Agent Summary
    a_positions = np.array(agent_positions)
    print(f"\n  Agent Statistics (across {len(all_results)} races):")
    print(f"    Mean Position:   {np.mean(a_positions):.2f}")
    print(f"    Best / Worst:    P{np.min(a_positions)} / P{np.max(a_positions)}")
    print(f"    Wins (P1):       {np.sum(a_positions == 1)} ({100*np.mean(a_positions == 1):.1f}%)")
    print(f"    Podiums (P1-3):  {np.sum(a_positions <= 3)} ({100*np.mean(a_positions <= 3):.1f}%)")
    print(f"    Points (P1-10):  {np.sum(a_positions <= 10)} ({100*np.mean(a_positions <= 10):.1f}%)")
    print(f"    Mean Reward:     {np.mean(agent_rewards):.2f}")
    print(f"    Mean Pit Stops:  {np.mean(agent_pit_stops):.2f}")
    print(f"    Mean Compounds:  {np.mean(agent_compounds):.2f}")
    print(f"    Mean Lap Time:   {np.mean(agent_lap_times):.2f}s")

    # Target Driver Summary
    t_positions = np.array(target_positions)
    print(f"\n  {target_code} Statistics (across {len(all_results)} races):")
    print(f"    Mean Position:   {np.mean(t_positions):.2f}")
    print(f"    Best / Worst:    P{np.min(t_positions)} / P{np.max(t_positions)}")
    print(f"    Wins (P1):       {np.sum(t_positions == 1)} ({100*np.mean(t_positions == 1):.1f}%)")
    print(f"    Podiums (P1-3):  {np.sum(t_positions <= 3)} ({100*np.mean(t_positions <= 3):.1f}%)")
    print(f"    Points (P1-10):  {np.sum(t_positions <= 10)} ({100*np.mean(t_positions <= 10):.1f}%)")
    print(f"    Mean Pit Stops:  {np.mean(target_pit_stops):.2f}")

    # Comparison
    print(f"\n  Agent vs {target_code} Comparison:")
    print(f"    Position delta average:  {np.mean(position_deltas):+.2f}")
    print(f"    Time delta average:      {np.mean(time_deltas):+.2f}s")
    agent_wins = sum(1 for d in position_deltas if d < 0)
    ties = sum(1 for d in position_deltas if d == 0)
    target_wins = sum(1 for d in position_deltas if d > 0)
    print(f"    Head-to-head:      Agent won {agent_wins} times, tied {ties} times, {target_code} won {target_wins} times")
    print("=" * 74)

# Command line interface
def main():
    parser = argparse.ArgumentParser(description="Reinforcement Learning for F1 Pit Stop strategy - Season Train & Test.")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands: train or evaluate")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a PPO agent on the specified training races.")
    train_parser.add_argument("--train-dir", type=str, default=TRAIN_DIR, help="Directory containing training race JSON files.")
    train_parser.add_argument("--timesteps", type=int, default=3_000_000)
    train_parser.add_argument("--n-envs", type=int, default=8)
    train_parser.add_argument("--model-name", type=str, default="f1_rl_ppo_2024_season")
    train_parser.add_argument("--seed", type=int, default=6)

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained PPO model on the specified test races.")
    eval_parser.add_argument("--test-dir", type=str, default=TEST_DIR, help="Directory containing test race JSON files.")
    eval_parser.add_argument("--model", type=str, default="f1_gym/models/f1_rl_ppo_2024_season.zip", help="Path to the trained PPO model file.")
    eval_parser.add_argument("--vecnormalize", type=str, default="f1_gym/models/f1_rl_ppo_2024_season_vecnormalize.pkl", help="Path to the VecNormalize statistics file.")
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