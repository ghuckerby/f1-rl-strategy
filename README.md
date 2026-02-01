# Reinforcement Learning for F1 Pit Stop Strategy

This project implements a Reinforcement Learning (RL) agent to optimize Formula 1 pit stop strategy. It uses a custom Gymnasium environment to simulate races, where the agent makes decisions on tyre compounds and pit stops to minimize race time and beat opponents.

## Project Structure

```
f1-rl-strategy/
├── main.py                     # Main entry point for CLI commands
├── requirements.txt            # Dependencies
├── f1_gym/                     # Main package
│   ├── agents/                 # RL agent implementations
│   │   └── train_dqn.py        # Training and evaluation using a DQN algorithm
│   │   └── train_ppo.py        # Training and evaluation using a PPO algorithm
│   ├── components/             # Simulation components
│   │   ├── events.py           # Race event handling (Safety Cars, slow stops, etc.)
│   │   ├── opponents.py        # Opponent AIs
│   │   └── parameters.py       # Tyre and track parameters
│   ├── envs/                   # Gymnasium environments
│   │   └── f1_env.py           # F1OpponentEnv class (main environment)
│   ├── logs/                   # Saved Training Logs + Visualisations
│   ├── models/                 # Saved RL models
│   └── visualisation/          # Visualisation tools
│       └── visualise.py        # Plotting race results
├── fastf1_cache/               # Cache for FastF1 data
└── scripts/
    └── time_benchmark.py       # Script to calculate strategies for opponents
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

To train the RL agent, use the `train-dqn` or `train-ppo` commands. There are also additional parameter commands listed in `main.py`.

**DQN:**

```
python main.py train-dqn
```

**PPO:**

```
python main.py train-ppo
```

### Evaluating the Agent

To evaluate a trained model, use the `evaluate-dqn` or `evaluate-ppo` commands.
This executes the trained policy on a number of races and outputs a quantitative analysis.

**DQN:**

```
python main.py evaluate-dqn --episodes 1000
```

**PPO:**

```
python main.py evaluate-ppo --episodes 1000
```

### Visualising a Race

To run a single race with the trained agent and generate a custom summary plot, use the `visualise-dqn` or `visualise-ppo` commands.

**DQN:**

```
python main.py visualise-dqn --output f1_gym/logs/race_summary.png
```

**PPO:**

```
python main.py visualise-ppo --output f1_gym/logs/race_summary_ppo.png
```

### Testing the Environment

To run a basic test of the environment loop:

```
python main.py test
```

## Key Components

### F1 Gymnasium Environment (`f1_gym/envs/f1_env.py`)

A custom Gymnasium environment `F1OpponentEnv` that simulates an F1 race.

- **Action Space**: Discrete(4) - 0: Stay Out, 1: Soft, 2: Medium, 3: Hard.
- **Observation Space**: Includes lap fraction, current compound (one-hot), tyre age, tyre wear, number of compounds used, position, time gaps (leader, ahead, behind), and safety car status.
- **Reward**: Based on minimizing race time and improving position.

### Opponents

The environment supports various opponent types to simulate other drivers on the track. The default is `AdaptiveBenchmarkOpponent`. The available opponents are:

- **RandomOpponent**: Makes random pit stop decisions.
- **HeuristicOpponent**: Uses a simple heuristic strategy.
- **BenchmarkOpponent**: Follows a pre-calculated optimal strategy.
- **HardBenchmarkOpponent**: A more challenging version of the benchmark opponent.
- **AdaptiveBenchmarkOpponent**: Adapts its strategy based on race conditions.
