import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from environments.predict_env import NBAPredictEnv
from gymnasium.wrappers import RecordEpisodeStatistics

# Load data
games_df = pd.read_csv("nbaData/enhanced_games.csv")
team_stats_df = pd.read_csv("nbaData/TeamStatistics.csv")

# Create environment and load trained model
env = RecordEpisodeStatistics(NBAPredictEnv(games_df, team_stats_df))
model = PPO.load("models/nba_rl_predictor")

print("Starting evaluation...\n")
NUM_EVAL_EPISODES = 1
total_abs_error = 0
total_games = 0

# Evaluate model
for ep in range(NUM_EVAL_EPISODES):
    print(f"Episode {ep + 1}/{NUM_EVAL_EPISODES}")
    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        row = env.env.games_df.iloc[env.env.current_step - 1]
        actual_spread = row["homeScore"] - row["awayScore"]
        predicted_spread = float(action[0])
        abs_error = abs(predicted_spread - actual_spread)

        print(f"{row['hometeamName']} vs {row['awayteamName']} | Predicted: {predicted_spread:.2f}, "
              f"Actual: {actual_spread}, Absolute Error: {abs_error:.2f}")

        total_abs_error += abs_error
        total_games += 1

# Report model performance
model_aae = total_abs_error / total_games
print("\nEvaluation Complete")
print(f"Total Games Evaluated: {total_games}")
print(f"Model Average Absolute Error (AAE): {model_aae:.2f}")

# Mean Spread Baseline Comparison
print("\nEvaluating Mean Spread Baseline...")

spread_series = games_df["homeScore"] - games_df["awayScore"]
mean_spread_value = spread_series.mean()
mean_baseline_aae = (spread_series - mean_spread_value).abs().mean()

print(f"Mean Spread Value: {mean_spread_value:.2f}")
print(f"Baseline Average Absolute Error: {mean_baseline_aae:.2f}")
print(f"Model Average Absolute Error:    {model_aae:.2f}")

improvement = mean_baseline_aae - model_aae
print(f"Improvement Over Baseline: {improvement:.2f} points")
