import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from f1_gym.lap_predictor import LapPredictor
from f1_gym.env.f1_real_env import F1RealEnv
from data.fastf1_data_extraction import COMPOUND_MAP

TEST_RACES = [
    "Abu Dhabi Grand Prix",
    "Austrian Grand Prix",
    "Italian Grand Prix",
    "Miami Grand Prix",
    "Singapore Grand Prix",
]
YEAR = 2024
TEST_DIR = "data/test_races"
OUTPUT_DIR = os.path.dirname(__file__)
COMPOUND_NAMES = {1: "Soft", 2: "Medium", 3: "Hard"}

# Checks how well the lap time model predicts lap times

# Predicts on the same data the model was trained on
# And leave-one-driver_out validation (train on all drivers, test on held out, repeat for each driver)
def validate_lap_prediction(predictor: LapPredictor):

    rows = []

    for race in TEST_RACES:
        model = LapPredictor.load_model(race)
        session = predictor.extractor.load_session(YEAR, race)
        laps = session.laps.copy()
        valid = laps[
            (laps['IsAccurate'] == True) &
            (laps['PitInTime'].isna()) &
            (laps['PitOutTime'].isna())
        ].copy()
        valid["LapTimeSec"] = valid["LapTime"].dt.total_seconds()
        valid["CompoundID"] = valid["Compound"].map(COMPOUND_MAP)
        valid = valid.sort_values(['Driver', 'LapNumber'])
        valid["TyreAge"] = valid.groupby(['Driver', 'Stint']).cumcount() + 1
        valid = valid[["Driver", "LapNumber", "TyreAge", "CompoundID", "LapTimeSec"]].dropna()

        features = ["LapNumber", "TyreAge", "CompoundID"]
        X, y = valid[features], valid["LapTimeSec"]

        # Predict on dataset used to train the model
        predictions = model.predict(X)
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        r2 = r2_score(y, predictions)

        # Leave one out driver cross validation
        drivers = valid["Driver"].unique()
        loo_errors = []
        for held_out in drivers:
            train_mask = valid["Driver"] != held_out
            test_mask = valid["Driver"] == held_out
            if test_mask.sum() < 5: # Skip drivers with too few laps
                continue
            model = RandomForestRegressor(n_estimators=100, random_state=6)
            model.fit(valid.loc[train_mask, features], valid.loc[train_mask, "LapTimeSec"])
            predictions = model.predict(valid.loc[test_mask, features])
            loo_errors.append(mean_absolute_error(valid.loc[test_mask, "LapTimeSec"], predictions))

        loo_mae = np.mean(loo_errors)
        rows.append({
            "Race": race,
            "Samples": len(y),
            "MAE (s)": round(mae, 3),
            "RMSE (s)": round(rmse, 3),
            "R^2": round(r2, 3),
            "LOO MAE (s)": round(loo_mae, 3),
        })

    df = pd.DataFrame(rows)
    return df

# Simulation consistency / replay check
# Loads the recorded target strategy and runs the env lap by lap
# Checks the difference to validate the anchoring
def validate_anchoring():

    rows = []

    for race in TEST_RACES:
        race_file = f"2024_{race.replace(' ', '_')}.json"
        with open(os.path.join(TEST_DIR, race_file)) as f:
            race_data = json.load(f)
        model = LapPredictor.load_model(race)
        env = F1RealEnv(race_data, predictor=model)
        obs, info = env.reset()

        target_strategy = race_data["target_driver_strategy"]
        pit_laps = set(int(lap) for lap in target_strategy["pit_laps"])
        pit_compounds = target_strategy["pit_compounds"]
        pit_idx = 0

        for lap in range(1, env.total_laps + 1):
            if lap in pit_laps:
                pit_idx += 1
                compound = pit_compounds[pit_idx] if pit_idx < len(pit_compounds) else 0
                action = compound
            else:
                action = 0
            obs, reward, done, truncated, info = env.step(action)

        agent_time = env.total_time
        target_time = target_strategy["total_time"]
        time_diff = agent_time - target_time

        rows.append({
            "Race": race,
            "Hamilton Time (s)": round(target_time, 2),
            "Agent Time (s)": round(agent_time, 2),
            "Time Difference (s)": round(time_diff, 2),
        })

    df = pd.DataFrame(rows)
    return df

# Removes target from testing
# Trains random forest on all other drivers, test on target
# Plots actual vs predicted across each stint
def plot_stint_traces(predictor: LapPredictor):
    plot_dir = os.path.join(OUTPUT_DIR, "stint_traces")
    os.makedirs(plot_dir, exist_ok=True)

    for race in TEST_RACES:
        session = predictor.extractor.load_session(YEAR, race)
        laps = session.laps.copy()
        valid = laps[
            (laps['IsAccurate'] == True) &
            (laps['PitInTime'].isna()) &
            (laps['PitOutTime'].isna())
        ].copy()
        valid["LapTimeSec"] = valid["LapTime"].dt.total_seconds()
        valid["CompoundID"] = valid["Compound"].map(COMPOUND_MAP)
        valid = valid.sort_values(["Driver", "LapNumber"])
        valid["TyreAge"] = valid.groupby(["Driver", "Stint"]).cumcount() + 1
        valid = valid[["Driver", "LapNumber", "TyreAge", "CompoundID", "LapTimeSec", "Stint"]].dropna()
        features = ["LapNumber", "TyreAge", "CompoundID"]

        # Train without Target Driver
        train = valid[valid["Driver"] != "HAM"]
        held_out = valid[valid["Driver"] == "HAM"]
        model = RandomForestRegressor(n_estimators=100, random_state=6)
        model.fit(train[features], train["LapTimeSec"])

        predictions = model.predict(held_out[features])
        mae = mean_absolute_error(held_out["LapTimeSec"], predictions)

        fig, ax = plt.subplots(figsize=(10, 6))
        for stint_id, group in held_out.groupby("Stint"):
            prediction = model.predict(group[features])
            compoundid = int(group["CompoundID"].iloc[0])
            label = f"Actual ({COMPOUND_NAMES.get(compoundid, "?")})"
            ax.plot(group["LapNumber"].values, group["LapTimeSec"].values, label=label, marker="o")
            ax.plot(group["LapNumber"].values, prediction, marker="x", label="Predicted")

        ax.set_title(f"Hamilton Stint Traces - {race.replace(' Grand Prix', ' GP')} (MAE: {mae:.2f}s)")
        ax.set_xlabel("Lap Number")
        ax.set_ylabel("Lap Time (s)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        file_name = f"stint_trace_{LapPredictor.race_to_filename(race)}.png"
        file_path = os.path.join(plot_dir, file_name)
        fig.savefig(file_path)
        plt.close(fig)
        print(f"Saved stint trace plot for {race} to {file_path}")

if __name__ == "__main__":
    predictor = LapPredictor()
    lap_prediction_results = validate_lap_prediction(predictor)
    anchoring_results = validate_anchoring()
    plot_stint_traces(predictor)

    # Save to CSV
    csv_path = os.path.join(OUTPUT_DIR, "lap_prediction_validation.csv")
    with open(csv_path, "w") as f:
        lap_prediction_results.to_csv(f, index=False)
        anchoring_results.to_csv(f, index=False)
    print(f"Validation results saved to {csv_path}")
