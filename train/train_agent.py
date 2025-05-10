import sys
import os

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from environments.predict_env import NBAPredictEnv  
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Load preprocessed game data
games_df = pd.read_csv('nbaData/PreprocessedGames.csv')

# Create and check environment
env = NBAPredictEnv(games_df)
check_env(env)

# Train the model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Save model
model.save("models/nba_rl_predictor")
print("✅ Model saved")
