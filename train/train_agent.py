import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from environments.predict_env import NBAPredictEnv

print("Loading data...")
games_df = pd.read_csv("nbaData/PreprocessedGames.csv", low_memory=False)
team_stats_df = pd.read_csv("nbaData/TeamStatistics.csv", low_memory=False)

print("Creating environment...")
env = NBAPredictEnv(games_df, team_stats_df)
check_env(env, warn=True)

print("Training model...")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

print("✅ Model trained. Saving...")
model.save("models/nba_rl_predictor")
print("✅ Model saved")
