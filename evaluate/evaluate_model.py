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
# Load trained model
model = PPO.load("models/nba_rl_predictor")

NUM_EVAL_EPISODES = 1
spread_error_total = 0
total_points_error_total = 0
game_total = 0

env = RecordEpisodeStatistics(NBAPredictEnv(games_df, team_stats_df))

for ep in range(NUM_EVAL_EPISODES):
    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        row = env.env.games_df.iloc[env.env.current_step - 1]
        actual_spread = row["homeScore"] - row["awayScore"]
        actual_total = row["homeScore"] + row["awayScore"]

        pred_spread = -25 + ((action[0] + 1) / 2.0) * 50
        pred_total = 175 + ((action[1] + 1) / 2.0) * 100

        spread_error = abs(pred_spread - actual_spread)
        total_points_error = abs(pred_total - actual_total)

        print(f"{row['hometeamName']} vs {row['awayteamName']} | "
              f"Spread: Pred {pred_spread:.2f} vs Act {actual_spread}, "
              f"Total: Pred {pred_total:.2f} vs Act {actual_total}")

        spread_error_total += spread_error
        total_points_error_total += total_points_error
        game_total += 1

print("\nEvaluation Complete")
print(f"Games Evaluated: {game_total}")
print(f"Spread AAE: {spread_error_total / game_total:.2f}")
print(f"Total Points AAE: {total_points_error_total / game_total:.2f}")


