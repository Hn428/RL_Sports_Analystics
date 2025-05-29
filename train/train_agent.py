import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from environments.predict_env import NBAPredictEnv

print("Loading data...")
games_df = pd.read_csv("nbaData/enhanced_games.csv", low_memory=False)
team_stats_df = pd.read_csv("nbaData/TeamStatistics.csv", low_memory=False)

print("Creating environment...")
env = NBAPredictEnv(games_df, team_stats_df)
check_env(env, warn=True)


print("Training model with tuned hyperparameters...")

policy_kwargs = dict(
    net_arch=[dict(pi=[128, 128], vf=[128, 128])]  # Bigger policy/value networks
)

model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.0005419523748796766,
    n_steps=4096,
    batch_size=128,
    gamma=0.9612808782925225,
    gae_lambda=0.8947172932455036,
    ent_coef=0.013327564081616292,
    clip_range=0.19685680083627766,
    n_epochs=8,
    policy_kwargs=policy_kwargs,
    verbose=1
)
model.learn(total_timesteps=300_000)

print("Model trained with tuned hyperparameters. Saving...")
model.save("models/nba_rl_predictor")
print("Model saved")
