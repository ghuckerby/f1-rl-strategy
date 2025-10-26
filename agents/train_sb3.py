
from stable_baselines3 import PPO
from env.f1_env import F1PitStopEnv

def main():
    env = F1PitStopEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("ppo_f1pitstop")

if __name__ == "__main__":
    main()