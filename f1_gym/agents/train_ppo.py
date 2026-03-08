import numpy as np
import os
from stable_baselines3 import PPO
from typing import Callable, Optional, Any, Dict, List
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
import torch.nn as nn

from f1_gym.env.f1_real_env import F1RealEnv
from f1_gym.env.multi_race_env import MultiRaceEnv

import wandb
from wandb.integration.sb3 import WandbCallback

COMPOUND_NAMES = {0: "?", 1: "SOFT", 2: "MED", 3: "HARD"}
COMPOUND_SHORT = {0: "?", 1: "S", 2: "M", 3: "H"}
ACTION_NAMES = {0: "STAY OUT", 1: "BOX SOFT", 2: "BOX MED", 3: "BOX HARD"}

# Helper for creating single-race environments
def make_env(rank: int, seed: int = 0, race_data: Optional[Dict] = None, predictor: Optional[Any] = None) -> Callable:
    """Utility function for multiple training environments."""

    def init() -> F1RealEnv:
        env = F1RealEnv(race_data=race_data, predictor=predictor)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return init

# Helper for creating multi-race environments
def make_multi_env(rank: int, seed: int = 0, race_configs: Optional[List[Dict]] = None, predictors: Optional[List[Any]] = None) -> Callable:
    """Utility function for multi-race training environments."""

    def init() -> MultiRaceEnv:
        env = MultiRaceEnv(race_configs=race_configs, predictors=predictors)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return init

# Custom callback for F1 Metrics during training in wandb
class F1MetricsCallback(BaseCallback):
    """Custom callback to log F1-specific metrics to WandB during training."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_positions = []
        self.episode_pit_stops = []
        self.episode_compound_counts = []
        self.episode_rewards = []
        
    def _on_step(self) -> bool:
        # Check for episode completion in any environment
        for idx, done in enumerate(self.locals.get("dones", [])):
            if done:
                info = self.locals.get("infos", [{}])[idx]
                if "episode_log" in info:
                    episode_log = info["episode_log"]
                    if episode_log:
                        # Final position
                        final_position = episode_log[-1].get("position", 20)
                        self.episode_positions.append(final_position)
                        
                        # Pit Stop Count
                        pit_stops = sum(1 for lap in episode_log if lap.get("pitted", False))
                        self.episode_pit_stops.append(pit_stops)
                        
                        # Compound Count
                        compounds = set(lap.get("compound") for lap in episode_log)
                        self.episode_compound_counts.append(len(compounds))
        
        # Log F1 metrics every 1000 steps
        if self.n_calls % 1000 == 0 and self.episode_positions:
            # Last 100 episodes stats
            self.logger.record("f1/mean_position", np.mean(self.episode_positions[-100:]))
            self.logger.record("f1/mean_pit_stops", np.mean(self.episode_pit_stops[-100:]))
            self.logger.record("f1/mean_compounds_used", np.mean(self.episode_compound_counts[-100:]))
            
            # Position distribution
            recent_positions = self.episode_positions[-100:]
            self.logger.record("f1/podium_rate", sum(1 for p in recent_positions if p <= 3) / len(recent_positions))
            self.logger.record("f1/points_rate", sum(1 for p in recent_positions if p <= 10) / len(recent_positions))
            
        return True

# Linear learning rate schedule
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return schedule

# Main training function
def train_f1_ppo(
    total_timesteps: int = 2_000_000,

    # Single-race data
    race_data: Optional[Dict] = None,
    predictor: Optional[Any] = None,

    # Multi-race data (takes priority)
    race_configs: Optional[List[Dict]] = None,
    predictors: Optional[List[Any]] = None,

    # Model naming
    model_name: str = "f1_rl_ppo",

    # Environment settings
    n_envs: int = 8,
    normalise_obs: bool = False,
    normalise_reward: bool = True,

    # PPO Parameters
    learning_rate: float = 3e-4,
    use_lr_schedule: bool = True,
    lr_schedule_type: str = "linear",

    n_steps: int = 2048,
    batch_size: int = 256,
    n_epochs: int = 5,
    
    gamma: float = 0.95,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    clip_range_vf: Optional[float] = None,
    ent_coef: float = 0.02,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    
    # Network
    net_arch: Dict[str, Any] = None,
    activation_fn: str = "tanh",  # change
    ortho_init: bool = True,
    
    # Evaluation and checkpoints
    eval_freq: int = 10000,
    n_eval_episodes: int = 10,
    save_freq: int = 50000,
    
    # Seed for Reproducibility
    seed: int = 6,
    
    # WandB settings
    use_wandb: bool = True,
    wandb_project: str = "f1-rl",
    run_name: Optional[str] = None,
) -> str:
    """Train a PPO agent on the F1 Real Environment with WandB logging."""
    
    # Logging and model directories
    LOG_DIR = "f1_gym/logs"
    MODEL_DIR = "f1_gym/models"
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # WandB Initialisation
    run = None
    if use_wandb:
        run_name = run_name or f"f1_ppo_{total_timesteps // 1_000_000}M"
        run = wandb.init(
            project=wandb_project,
            name=run_name,
            config={
                "algorithm": "PPO",
                "total_timesteps": total_timesteps,
                "n_envs": n_envs,
                "learning_rate": learning_rate,
                "lr_schedule_type": lr_schedule_type if use_lr_schedule else "constant",
                "n_steps": n_steps,
                "batch_size": batch_size,
                "n_epochs": n_epochs,
                "gamma": gamma,
                "gae_lambda": gae_lambda,
                "clip_range": clip_range,
                "ent_coef": ent_coef,
                "vf_coef": vf_coef,
                "max_grad_norm": max_grad_norm,
                "normalise_obs": normalise_obs,
                "normalise_reward": normalise_reward,
                "seed": seed,
            },
            sync_tensorboard=True,
            reinit=True,
        )

    # Decide whether to use multi-race or single-race environments
    multi_race = race_configs is not None and predictors is not None

    # Create vectorized environment
    if multi_race:
        print(f"Creating {n_envs} parallel multi-race environments ({len(race_configs)} races)...")
        env = DummyVecEnv([make_multi_env(i, seed, race_configs, predictors) for i in range(n_envs)])
    else:
        print(f"Creating {n_envs} parallel environments...")
        env = DummyVecEnv([make_env(i, seed, race_data, predictor) for i in range(n_envs)])
    
    # Observation and reward normalisation
    if normalise_obs or normalise_reward:
        env = VecNormalize(
            env,
            norm_obs=normalise_obs,
            norm_reward=normalise_reward,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=gamma,
        )
    
    # Create evaluation environment
    if multi_race:
        eval_env = DummyVecEnv([make_multi_env(0, seed + 100, race_configs, predictors)])
    else:
        eval_env = DummyVecEnv([make_env(0, seed + 100, race_data, predictor)])
    if normalise_obs or normalise_reward:
        eval_env = VecNormalize(
            eval_env,
            norm_obs=normalise_obs,
            norm_reward=False,
            training=False,
            clip_obs=10.0,
        )
    
    # Setup learning rate schedule
    if use_lr_schedule:
        lr = linear_schedule(learning_rate)
    else:
        lr = learning_rate
    
    # PPO network architecture
    if net_arch is None:
        net_arch = dict(
            pi=[64, 64],
            vf=[64, 64],
        )
    
    # Map activation function string to torch module
    activation_map = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
    }
    activation = activation_map.get(activation_fn, nn.Tanh)
    policy_kwargs = {
        "net_arch": net_arch,
        "activation_fn": activation,
        "ortho_init": ortho_init,
    }
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        clip_range_vf=clip_range_vf,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        tensorboard_log=f"{LOG_DIR}/tensorboard/{run.id if run else 'local'}",
    )
    
    # Setup callbacks
    callbacks = []
    
    # F1 Metrics callback
    f1_callback = F1MetricsCallback(verbose=1)
    callbacks.append(f1_callback)
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{MODEL_DIR}/best_ppo",
        log_path=f"{LOG_DIR}/eval",
        eval_freq=eval_freq // n_envs,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )
    callbacks.append(eval_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // n_envs,
        save_path=f"{MODEL_DIR}/checkpoints_ppo",
        name_prefix="f1_ppo",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)
    
    # WandB callback
    if use_wandb and run:
        wandb_callback = WandbCallback(
            model_save_path=f"{MODEL_DIR}/wandb/{run.id}",
            verbose=2,
        )
        callbacks.append(wandb_callback)
    
    callback_list = CallbackList(callbacks)
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        progress_bar=True,
    )
    
    # Save final model
    model_path = os.path.join(MODEL_DIR, model_name)
    model.save(model_path)
    print(f"Saved model to {model_path}.zip")
    
    # Save VecNormalize statistics
    if normalise_obs or normalise_reward:
        env.save(f"{model_path}_vecnormalize.pkl")
        print(f"Saved VecNormalize stats to {model_path}_vecnormalize.pkl")
    
    # Cleanup
    env.close()
    eval_env.close()
    
    if run:
        run.finish()
    
    return f"{model_path}.zip"
    
def evaluate_ppo_model(
    model_path: str = "f1_gym/models/f1_rl_ppo.zip",
    vecnormalize_path: Optional[str] = "f1_gym/models/f1_rl_ppo_vecnormalize.pkl",
    num_episodes: int = 1000,
    deterministic: bool = True,
    verbose: bool = True,
    race_data: Optional[Dict] = None,
    predictor: Optional[Any] = None,
) -> Dict[str, Any]:
    """Evaluate a trained PPO model on the F1 Real Environment and log detailed metrics."""

    print("\n" + "=" * 70) 
    print("  EVALUATION  —  F1RealEnv (PPO)")
    print(f"  Model: {model_path}")
    print(f"  Episodes: {num_episodes}")
    print("=" * 70)
    
    # Load the model
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return {}
    model = PPO.load(model_path)
    base_env = F1RealEnv(race_data=race_data, predictor=predictor)
    env = DummyVecEnv([lambda: base_env])
    
    # Load VecNormalize stats if available
    if vecnormalize_path and os.path.exists(vecnormalize_path):
        env = VecNormalize.load(vecnormalize_path, env)
        env.training = False
        env.norm_reward = False
    
    # Metrics
    race_name = race_data.get("name", "Unknown Race")
    total_laps = race_data.get("track", {}).get("total_laps", 0)
    results = {
        "rewards": [],
        "positions": [],
        "pit_stops": [],
        "compounds_used": [],
        "lap_times": [],
        "total_times": [],
    }
    
    # Evaluate the model over multiple episodes
    for episode in range(num_episodes):
        print(f"\n{'─' * 70}")
        print(f"  EPISODE {episode + 1}/{num_episodes}  —  {race_name}  ({total_laps} laps)")
        print(f"{'─' * 70}")

        obs = env.reset()
        episode_reward = 0
        done = False
        terminal_info = None
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, dones, infos = env.step(action)
            episode_reward += reward[0]
            done = dones[0]
            
            # Print lap-by-lap output
            if verbose and infos[0].get("lap", 0) > 0:
                row = infos[0]
                compound = COMPOUND_NAMES.get(row.get("compound"), "?")
                action_str = ACTION_NAMES.get(row.get("action"), "?")
                sc = "SC" if row.get("sc_active") else "--"
                print(
                    f"Lap {row.get('lap', 0):>2} | {sc} | action: {action_str:>10} | compound: {compound} | "
                    f"tyre_age: {row.get('tyre_age', 0):>2} | lap_time: {row.get('lap_time', 0) or 0:.2f}s | "
                    f"total_time: {row.get('total_time', 0):.2f}s | pitted: {row.get('pitted', False)} | "
                    f"position: {row.get('position', 0)}"
                )

            if done:
                terminal_info = infos[0]
        
        # Episode summary
        episode_log = terminal_info["episode_log"]
        final_position = episode_log[-1].get("position", 20)
        total_time = episode_log[-1].get("total_time", 0)
        pit_stops = sum(1 for lap in episode_log if lap.get("pitted", False))
        compounds = len(set(lap.get("compound") for lap in episode_log))
        lap_times = [lap.get("lap_time", 0) for lap in episode_log if lap.get("lap_time")]
        avg_lap_time = np.mean(lap_times) if lap_times else 0

        # Stint order
        stints = []
        if episode_log:
            stints.append(COMPOUND_SHORT.get(episode_log[0].get("compound"), "?"))
            for lap in episode_log:
                if lap.get("pitted"):
                    stints.append(COMPOUND_SHORT.get(lap.get("compound"), "?"))
        compounds_str = " -> ".join(stints)

        results["rewards"].append(episode_reward)
        results["positions"].append(final_position)
        results["pit_stops"].append(pit_stops)
        results["compounds_used"].append(compounds)
        results["lap_times"].append(avg_lap_time)
        results["total_times"].append(total_time)

        print(f"\nEpisode {episode + 1}/{num_episodes} completed.")
        print(f"Agent finished P{final_position}  |  "
              f"Total Time: {total_time:.2f}s  |  Pit Stops: {pit_stops}  |  "
              f"Compounds Used: {compounds} ({compounds_str})  |  Reward: {episode_reward:.2f}\n")
        
        # Comparison to target driver
        target = race_data.get("target_driver_strategy", {})
        target_lap_times = target.get("lap_times", [])
        target_total_time = target.get("total_time", 0)
        target_code = target.get("driver_code", "TARGET")
        target_position = target.get("finishing_position", "?")
        if episode_log and target_lap_times:
            # Lap 1 comparison
            agent_lap1 = next((l["lap_time"] for l in episode_log if l.get("lap") == 1 and l.get("lap_time")), None)
            target_lap1 = target_lap_times[0] if target_lap_times else None

            if agent_lap1 and target_lap1:
                delta1 = agent_lap1 - target_lap1
                print(f"\n  ANCHORING VALIDATION")
                print(f"  {'─' * 52}")
                print(f"  Lap 1  →  Agent: {agent_lap1:.2f}s  |  {target_code}: {target_lap1:.2f}s  |  Δ {delta1:+.2f}s")

            # Overall comparison
            print(f"  Total  →  Agent: {total_time:.2f}s  |  {target_code}: {target_total_time:.2f}s  |  "
                  f"Δ {total_time - target_total_time:+.2f}s")
            print(f"  {target_code} finished P{target_position} (real)  |  Agent finished P{final_position}")
            print(f"  {'─' * 52}")

        # Standings table — (handles lapped drivers classified as +N Lap(s))
        print(f"\n  {'RACE STANDINGS':^74}")
        print(f"  {'─' * 74}")
        print(f"  {'Pos':>3} | {'Driver':<21} | {'Total Time':>11} | {'Gap':>8} | {'Pen':>5} | {'Stops':>5} | Strategy")
        print(f"  {'─' * 74}")

        standings = terminal_info.get("final_standings", [])

        if standings:
            leader_laps = standings[0].get("laps", total_laps) if standings else total_laps
            leader_time = standings[0]["time"] if standings and not standings[0]["dnf"] else 0
            for position, entry in enumerate(standings, 1):
                marker = "   << AGENT" if entry["is_agent"] else ""
                penalty = f"+{entry['penalty']:.0f}s" if entry.get("penalty") else "     "

                if entry["dnf"]:
                    gap_str = "DNF"
                    time_str = "DNF"
                elif entry.get("laps", leader_laps) < leader_laps:
                    laps_behind = leader_laps - entry["laps"]
                    gap_str = f"+{laps_behind} Lap{'s' if laps_behind > 1 else ''}"
                    time_str = f"{entry['time']:.2f}s"
                elif entry["time"] == leader_time:
                    gap_str = "LEADER"
                    time_str = f"{entry['time']:.2f}s"
                else:
                    gap_str = f"+{entry['time'] - leader_time:.2f}s"
                    time_str = f"{entry['time']:.2f}s"

                print(
                    f"  {position:>3} | {entry['code']:<4} {entry['name']:<16} | "
                    f"{time_str:>11} | {gap_str:>8} | {penalty:>5} | {entry['pit_stops']:>5} | "
                    f"{entry['strategy']}{marker}"
                )

        print(f"  {'─' * 74}")
    
    # Print summary statistics for each race
    print("\nReward Statistics:")
    print(f"Mean: {np.mean(results['rewards']):.2f}, "
          f"Std: {np.std(results['rewards']):.2f}, "
          f"Min: {np.min(results['rewards']):.2f}, "
          f"Max: {np.max(results['rewards']):.2f}")
    
    print("\nPosition Statistics:")
    print(f"Mean: {np.mean(results['positions']):.2f}, "
          f"Std: {np.std(results['positions']):.2f}, "
          f"Best: {np.min(results['positions'])}, "
          f"Worst: {np.max(results['positions'])}")
    
    print("\nRace Time Statistics:")
    print(f"Mean: {np.mean(results['total_times']):.2f}s, "
          f"Std: {np.std(results['total_times']):.2f}s, "
          f"Best: {np.min(results['total_times']):.2f}s, "
          f"Worst: {np.max(results['total_times']):.2f}s")
    
    positions = np.array(results['positions'])
    print("\nPosition Distribution:")
    print(f"Wins (P1): {np.sum(positions == 1)} ({100*np.mean(positions == 1):.1f}%), "
          f"Podiums (P1-3): {np.sum(positions <= 3)} ({100*np.mean(positions <= 3):.1f}%), "
          f"Points (P1-10): {np.sum(positions <= 10)} ({100*np.mean(positions <= 10):.1f}%)")
    
    print("\nStrategy Statistics:")
    print(f"Mean Pit Stops: {np.mean(results['pit_stops']):.2f}, "
          f"Mean Compounds: {np.mean(results['compounds_used']):.2f}, "
          f"Mean Lap Time: {np.mean(results['lap_times']):.2f}s")
    
    env.close()
    
    return results  