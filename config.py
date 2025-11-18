# Change variable for environment
ENVIRONMENT = 'deterministic'

def get_environment_config():
    if ENVIRONMENT == 'deterministic':
        from f1_gym.deterministic.env.dt_f1_env import F1PitStopEnv
        from f1_gym.deterministic.agents import dt_train
        from f1_gym.deterministic.env import dt_dynamics
        
        return {
            'env_class': F1PitStopEnv,
            'train_module': dt_train,
            'dynamics_module': dt_dynamics,
            'base_path': 'f1_gym/deterministic',
            'project_name': 'f1-deterministic-dqn',
            'env_name': 'Deterministic',
        }
    
    elif ENVIRONMENT == 'stochastic':
        from f1_gym.stochastic.env.st_f1_env import F1PitStopEnv
        from f1_gym.stochastic.agents import st_train
        from f1_gym.stochastic.env import st_dynamics
        
        return {
            'env_class': F1PitStopEnv,
            'train_module': st_train,
            'dynamics_module': st_dynamics,
            'base_path': 'f1_gym/stochastic',
            'project_name': 'f1-stochastic-dqn',
            'env_name': 'Stochastic',
        }
    else:
        raise ValueError(f"Unknown environment: {ENVIRONMENT}. Use 'deterministic' or 'stochastic'.")


def get_constants():
    config = get_environment_config()
    return (
        config['dynamics_module'].SOFT,
        config['dynamics_module'].MEDIUM,
        config['dynamics_module'].HARD,
    )
