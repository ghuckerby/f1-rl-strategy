import numpy as np
import os
from stable_baselines3 import PPO
from typing import Callable, Optional
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
import torch.nn as nn

from f1_gym.envs.f1_env import F1OpponentEnv

import wandb
from wandb.integration.sb3 import WandbCallback

def make_env(rank: int, seed: int = 0) -> Callable:
    def _init():
        env = F1OpponentEnv()
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

# Linear learning rate schedule
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return schedule

def train_f1_ppo(
    total_timesteps: int = 2_000_000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    clip_range_vf: Optional[float] = None,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,

    n_envs: int = 8,
    use_subprocess: bool = True,
    normalise_obs: bool = True,
    normalise_reward: bool = True,

    seed: int = 42,
):
    # Directories
    LOG_DIR = "f1_gym/logs"
    MODEL_DIR = "f1_gym/models"
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # WandB Initialisation
    run = None
    run = wandb.init(
        project="f1-rl",
        name=f"f1_rl_ppo_{total_timesteps//1_000_000}M",
        config={
            "algorithm": "PPO",
            "total_timesteps": total_timesteps,
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_range": clip_range,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm,
            "n_envs": n_envs,
            "normalise_obs": normalise_obs,
            "normalise_reward": normalise_reward,
            "seed": seed,
        },
        sync_tensorboard=True,
        reinit=True,
    )

    # Vectorised Environment
    if use_subprocess and n_envs > 1:
        env = SubprocVecEnv([make_env(i, seed) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(i, seed) for i in range(n_envs)])

    # Observation and Reward Normalisation
    if normalise_obs or normalise_reward:
        env = VecNormalize(
            env,
            norm_obs=normalise_obs,
            norm_reward=normalise_reward,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=gamma,
        )

    # Evaluation Environment
    eval_env = DummyVecEnv([make_env(0, seed + 100)])
    if normalise_obs or normalise_reward:
        eval_env = VecNormalize(
            eval_env,
            norm_obs=normalise_obs,
            norm_reward=False,
            training=False,
            clip_obs=10.0,
        )

    # Network Architecture
    policy_kwargs = {
        "net_arch": [dict(pi=[256, 256], vf=[256, 256])],
        "activation_fn":  nn.Tanh,
        "ortho_init": True,
    }

    # PPO Model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=linear_schedule(learning_rate),
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
        tensorboard_log=f"{LOG_DIR}/tensorboard/{run.id}",
    )

    # Callbacks
    callback = WandbCallback(
        model_save_path=f"{MODEL_DIR}/wandb/{run.id}",
        verbose=2,
    )

    # Train Model
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )

    # Save Final Model
    model_name = f"f1_rl_ppo.zip"
    model_path = os.path.join(MODEL_DIR, model_name)
    model.save(model_path)
    print(f"Saved model to {model_path}")

    return model_path
    
def evaluate_ppo_model():
    pass