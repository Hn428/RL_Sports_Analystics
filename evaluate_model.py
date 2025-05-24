import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from environments.predict_env import NBAPredictEnv
from gymnasium.wrappers import RecordEpisodeStatistics

# ========== CONFIG ==========
NUM_EVAL_EPISODES = 1
REWARD_SHAPING = True  # Set to False for strict match only
# ============================

# Label map
spread_labels = {
    0: "Home Win >10",
    1: "Home Win ≤10",
    2: "Close Game (±5)",
    3: "Away Win ≤10",
    4: "Away Win >10"
}

print("Loading data...")
games_df = pd.read_csv("nbaData/enhanced_games.csv", low_memory=False)
team_stats_df = pd.read_csv("nbaData/TeamStatistics.csv", low_memory=False)

print("Creating environment...")
raw_env = NBAPredictEnv(games_df, team_stats_df)
env = RecordEpisodeStatistics(raw_env)

print("Loading model...")
model = PPO.load("models/nba_rl_predictor")

correct_predictions = 0
total_games = 0
all_preds = []
all_actuals = []

print(f"Starting evaluation with {NUM_EVAL_EPISODES} episodes...")

for ep in range(NUM_EVAL_EPISODES):
    print(f"Episode {ep + 1}/{NUM_EVAL_EPISODES}")
    obs, _ = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        pred_class = int(action)

        obs, reward, terminated, truncated, info = env.step(pred_class)
        done = terminated or truncated

        row = env.env.games_df.iloc[env.env.current_step - 1]
        actual_spread = row["homeScore"] - row["awayScore"]

        actual_class = (
            0 if actual_spread > 10 else
            1 if actual_spread > 0 else
            2 if abs(actual_spread) <= 5 else
            3 if actual_spread < 0 and abs(actual_spread) <= 10 else
            4
        )

        # Reward shaping
        if REWARD_SHAPING:
            if pred_class == actual_class:
                reward = 2.0
            elif abs(pred_class - actual_class) == 1:
                reward = 1.0
            else:
                reward = -1.0
        else:
            reward = 2.0 if pred_class == actual_class else -1.0

        # Track accuracy
        if pred_class == actual_class:
            correct_predictions += 1
        total_games += 1

        all_preds.append(pred_class)
        all_actuals.append(actual_class)

        print(f"{row['hometeamName']} vs {row['awayteamName']} | 🧠 Predicted: {spread_labels[pred_class]}, Actual: {spread_labels[actual_class]}, Reward: {reward:.1f}")

print("\n✅ Evaluation Complete")
print(f"Total Games Evaluated: {total_games}")
print(f"Correct Predictions: {correct_predictions}/{total_games}")
print(f"Accuracy: {(correct_predictions / total_games):.2%}")
