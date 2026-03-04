# Reinforcement Learning for F1 Pit Stop Strategy

This project implements a Reinforcement Learning (RL) agent to optimize Formula 1 pit stop strategy. It uses a custom Gymnasium environment to simulate races, where the agent makes decisions on tyre compounds and pit stops to minimize race time and beat opponents.

## Project Structure (Outdated)

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
│   ├── env/                    # Gymnasium environment
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
