from stable_baselines3 import PPO
from environments.predict_env import NBAPredictEnv
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd
import numpy as np

# Load preprocessed game data
print("Loading data...")
games_df = pd.read_csv("nbaData/PreprocessedGames.csv")

# Create and wrap the environment
print("Creating environment...")
env = DummyVecEnv([lambda: NBAPredictEnv(games_df)])

# Load the trained model
print("Loading model...")
model = PPO.load("models/nba_rl_predictor")

# Evaluate model
correct_predictions = 0
total_predictions = 0

NUM_EVAL_EPISODES = 100  # Back to original number

print(f"Starting evaluation with {NUM_EVAL_EPISODES} episodes...")
for episode in range(NUM_EVAL_EPISODES):
    print(f"Episode {episode + 1}/{NUM_EVAL_EPISODES}")
    obs = env.reset()

    done = False
    while not done:
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            print("🚫 Skipping NaN/Inf observation:", obs)
            obs = env.reset()
            continue

        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)

    if reward > 0:
        correct_predictions += 1
    total_predictions += 1

# Print evaluation results
accuracy = correct_predictions / total_predictions
print(f"✅ Evaluation Complete")
print(f"Correct Predictions: {correct_predictions}/{total_predictions}")
print(f"Accuracy: {accuracy:.2%}")
