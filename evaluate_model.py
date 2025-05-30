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
spread_error_total = 0
total_points_error_total = 0
win_correct_total = 0
game_total = 0

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
        actual_total = row["homeScore"] + row["awayScore"]
        actual_win = 1 if actual_spread > 0 else 0

        # === Unscale Actions ===
        pred_spread = -25 + ((action[0] + 1) / 2.0) * 50       # [-1,1] -> [-25,25]
        pred_total = 175 + ((action[1] + 1) / 2.0) * 100        # [-1,1] -> [175,275]
        pred_win = 1 if action[2] > 0 else 0                    # classification


        spread_error = abs(pred_spread - actual_spread)
        total_points_error = abs(pred_total - actual_total)
        win_correct = int(pred_win == actual_win)

        print(f"{row['hometeamName']} vs {row['awayteamName']} | "
              f"Spread: Pred {pred_spread:.2f} vs Act {actual_spread}, "
              f"Total: Pred {pred_total:.2f} vs Act {actual_total}, "
              f"Win: Pred {'W' if pred_win else 'L'} vs Act {'W' if actual_win else 'L'}")

        spread_error_total += spread_error
        total_points_error_total += total_points_error
        win_correct_total += win_correct
        game_total += 1

print("\nEvaluation Complete")
print(f"Games Evaluated: {game_total}")
print(f"Spread AAE: {spread_error_total / game_total:.2f}")
print(f"Total Points AAE: {total_points_error_total / game_total:.2f}")
print(f"Win Prediction Accuracy: {win_correct_total / game_total:.2%}")

