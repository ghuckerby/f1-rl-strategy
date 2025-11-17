
from config import get_environment_config, ENVIRONMENT

if __name__ == "__main__":
    config = get_environment_config()
    print(f"Training RL Agent on F1 pit-stop environment ({config['env_name']} mode).")
    config['train_module'].main()