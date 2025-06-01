import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from environments.predict_env import NBAPredictEnv
from gymnasium.wrappers import RecordEpisodeStatistics

# Load data
games_df = pd.read_csv("nbaData/enhanced_games.csv")
team_stats_df = pd.read_csv("nbaData/TeamStatistics.csv")

CHECKPOINTS = [100_000, 200_000, 300_000, 400_000, 500_000]
results = []

# Compute baselines
spread_baseline = games_df["homeScore"].mean() - games_df["awayScore"].mean()
total_baseline = games_df["homeScore"].mean() + games_df["awayScore"].mean()

# Calculate baseline AAE (static error from predicting mean every time)
baseline_spread_aae = abs(spread_baseline - (games_df["homeScore"] - games_df["awayScore"])).mean()
baseline_total_aae = abs(total_baseline - (games_df["homeScore"] + games_df["awayScore"])).mean()

for step in CHECKPOINTS:
    print(f"\nEvaluating PPO model at {step} steps...")
    model_path = f"models/nba_rl_{step}"
    model = PPO.load(model_path)

    env = RecordEpisodeStatistics(NBAPredictEnv(games_df, team_stats_df))

    spread_errors, total_errors, rewards = [], [], []

    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        rewards.append(reward)

        row = env.env.games_df.iloc[env.env.current_step - 1]
        actual_spread = row["homeScore"] - row["awayScore"]
        actual_total = row["homeScore"] + row["awayScore"]

        pred_spread = -25 + ((action[0] + 1) / 2.0) * 50
        pred_total = 175 + ((action[1] + 1) / 2.0) * 100

        spread_errors.append(abs(pred_spread - actual_spread))
        total_errors.append(abs(pred_total - actual_total))

    mean_spread_aae = np.mean(spread_errors)
    mean_total_aae = np.mean(total_errors)
    mean_reward = np.mean(rewards)

    results.append({
        "checkpoint": step,
        "mean_reward": mean_reward,
        "spread_aae": mean_spread_aae,
        "total_aae": mean_total_aae,
        "baseline_spread_aae": baseline_spread_aae,
        "baseline_total_aae": baseline_total_aae
    })

    # Console log
    print(f"Step {step} Evaluation Summary:")
    print(f"  Mean Reward          : {mean_reward:.2f}")
    print(f"  Spread AAE           : {mean_spread_aae:.2f}")
    print(f"  Total Points AAE     : {mean_total_aae:.2f}")
    print(f"  Baseline Spread AAE  : {baseline_spread_aae:.2f}")
    print(f"  Baseline Total AAE   : {baseline_total_aae:.2f}")

# Save results
df = pd.DataFrame(results)
df.to_csv("ppo_eval_results.csv", index=False)
print("\n✅ Evaluation complete. Saved to ppo_eval_results.csv")
