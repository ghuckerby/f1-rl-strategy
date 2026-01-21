# Reinforcement Learning for F1 Pit Stop Strategy

This project implements a Reinforcement Learning (RL) agent to optimize Formula 1 pit stop strategy. It uses a custom Gymnasium environment to simulate races, where the agent makes decisions on tyre compounds and pit stops to minimize race time and beat opponents.

## Project Structure

```

f1-rl-strategy/
├── main.py # Main entry point for CLI commands
├── requirements.txt # Dependencies
├── f1_gym/ # Main package
│ ├── agents/ # RL agent implementations and training logic
│ │ └── train.py # Training and evaluation functions
│ ├── components/ # Simulation components
│ │ ├── envs/ # Gymnasium environment definitions
│ │ │ └── f1_env.py # F1OpponentEnv class
│ │ ├── tracks.py # Track and tyre parameters
│ │ ├── opponents.py # Opponent AIs
│ │ └── events.py # Race event handling (Safety Cars etc.)
│ ├── models/ # Saved RL models
│ └── visualisation/ # Visualisation tools
│ └── visualise.py # Plotting race results
├── fastf1_cache/ # Cache for FastF1 data
└── scripts/
└── fastf1_loader.py # Script to load real F1 data for calibration

```

## Installation

1.  **Clone the repository**:
    ```
    git clone <repository_url>
    cd f1-rl-strategy
    ```
2.  **Create a virtual environment**:
    ```
    python -m venv .venv
    source .venv/bin/activate # On Windows: .venv\Scripts\activate
    ```
3.  **Install dependencies**:
    ```
    pip install -r requirements.txt
    ```

## Usage

The project is controlled with `main.py`.

### Training the Agent

To train the RL agent, use the `train` command. You can specify the total number of timesteps for training.

python main.py train --timesteps 1000000

### Evaluating the Agent

To evaluate a trained model, use the `evaluate` command.
This executes the trained policy on a number of races and outputs a quantitative analysis.

python main.py evaluate --model f1_gym/models/f1_rl_dqn.zip --episodes 100

### Visualising a Race

To run a single race with the trained agent and generate a custom summary plot, use the `visualise` command.

python main.py visualise --model f1_gym/models/f1_rl_dqn.zip --output f1_gym/logs/race_summary.png

### Testing the Environment

To run a basic test of the environment loop:

python main.py test

## Key Components

### F1 Gymnasium Environment (`f1_gym/envs/f1_env.py`)

A custom Gymnasium environment `F1OpponentEnv` that simulates an F1 race.

- **Action Space**: Discrete(4) - 0: Stay Out, 1: Soft, 2: Medium, 3: Hard.
- **Observation Space**: Includes lap fraction, current compound (one-hot), tyre age, tyre wear, position, and time gaps to other drivers.
- **Reward**: Based on minimizing race time and improving position.

### Data Loading (`scripts/fastf1_loader.py`)

Uses the `fastf1` library to load real Formula 1 session data. It calculates tyre degradation parameters (base lap time and degradation rate) from actual race laps to calibrate the simulation.

### Opponents

The environment includes `RandomOpponent` agents to simulate other drivers on the track, providing a competitive context for the RL agent.
