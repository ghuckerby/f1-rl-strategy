# Reinforcement Learning for F1 Pit Stop Strategy

A PPO-based reinforcement learning agent that learns Formula 1 pit-stop strategy using a real-race replay environment built from historical season data from the FastF1 API. The agent replaces a driver (Lewis Hamilton) in each race while the remaining 19 drivers follow their historical strategies.

## Project Structure

```
f1-rl-strategy/
├── main.py                          # Entry point — training and evaluation
├── requirements.txt                 # Dependencies
│
├── data/                            # Data Extraction and Extracted race configurations (JSON)
│   │
│   ├── fastf1_data_extraction.py    # FastF1DataExtractor: builds JSON race configs
│   ├── training_races/              # Training races (12 races)
│   └── test_races/                  # Held-out test races (5 races)
│
├── f1_gym/                          # Main package
│   │
│   ├── lap_predictor.py             # Lap prediction model
│   ├── reward_config.py             # Reward configuration for the environment
│   │
│   ├── agents/
│   │   └── train_ppo.py             # Main PPO training and evaluation
│   │
│   ├── components/
│   │   ├── parameters.py            # RaceParams: track info, lap-time prediction, anchoring
│   │   ├── events.py                # RealRaceEvents: Safety Car periods
│   │   └── opponents.py             # RealOpponent: historical driver replay
│   │
│   ├── env/
│   │   ├── f1_real_env.py           # F1RealEnv: main Gymnasium environment
│   │   └── multi_race_env.py        # MultiRaceEnv: wrapper for multi-race training
│   │
│   ├── models/                      # Saved model weights
│   │   ├── final_model/             # Final PPO and vecnormalize model weights
│   │   └── lap_predictors/          # Lap prediction models for each track used
│   │
│   ├── logs/                        # Training logs and W&B outputs
│   ├── visualisation/
│   │   └── evaluation_plots/        # Cumulative time-delta and position trace plots
│   │
│   └── validation/
│       ├── stint_traces/            # Environment validation stint traces for each test track
│       └── validation.py            # Script for environment validation
│
└── opponent(legacy)/                # Earlier synthetic version (not used in final system)
```

## How it works

1. **Data extraction** — `FastF1DataExtractor` pulls lap times, tyre usage, pit stops, Safety Car events, and race results from the FastF1 API and saves them as JSON race configs.
2. **Lap-time prediction** — A `RandomForestRegressor` trained per race predicts lap times from lap number, tyre age, and compound. A per-race pace offset then adjusts predictions closer to the target driver.
3. **Replay environment** — `F1RealEnv` reconstructs a race lap by lap. The agent controls pit stop timing and compound selection whilst all other drivers replay their historical strategies.
4. **Training** — A tuned PPO agent from Stable-Baselines3 is trained across 3M timesteps with `MultiRaceEnv` randomly sampling from the 12 training races each episode.
5. **Evaluation** — The saved policy is run on 5 held-out test races and compared against Hamilton's historical finishing position and total race time.

## Installation

```bash
git clone <repository_url>
cd f1-rl-strategy
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

(See CLI commands in main.py for further usage)

### Train the agent

Trains a PPO agent across the 12 training races for 3M timesteps

```bash
python main.py train
```

### Evaluate on test races

Runs the saved agent on the 5 held-out test races and prints a results summary

```bash
python main.py evaluate
```

## Results

Evaluated on 5 held-out 2024 races against Lewis Hamilton's historical performance:

| Race           | Agent    | Hamilton | Δ Position | Δ Time (s) |
| -------------- | -------- | -------- | ---------- | ---------- |
| Bahrain GP     | P5       | P7       | +2         | −6.46      |
| Chinese GP     | P8       | P9       | +1         | −7.53      |
| Hungarian GP   | **P1**   | P3       | +2         | −32.85     |
| Mexico City GP | P4       | P4       | 0          | 0.00       |
| Miami GP       | P6       | P6       | 0          | +1.42      |
| **Average**    | **4.80** | **5.80** | **+1.00**  | **−9.08**  |
