import os
import pandas as pd
from stable_baselines3 import PPO
from environments.predict_env import NBAPredictEnv

# === Load Data ===
games_df = pd.read_csv("nbaData/enhanced_games.csv", low_memory=False)
team_stats_df = pd.read_csv("nbaData/TeamStatistics.csv", low_memory=False)

# === Environment Setup ===
env = NBAPredictEnv(games_df, team_stats_df)

# === PPO Setup ===
policy_kwargs = dict(
    net_arch=[dict(pi=[128, 128], vf=[128, 128])]
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

# === Train and Save Checkpoints ===
TOTAL_STEPS = 500_000
SAVE_INTERVAL = 100_000

for step in range(SAVE_INTERVAL, TOTAL_STEPS + 1, SAVE_INTERVAL):
    print(f"\nTraining PPO up to {step} timesteps...")
    model.learn(total_timesteps=SAVE_INTERVAL, reset_num_timesteps=False)
    save_path = f"models/nba_rl_{step}"
    model.save(save_path)
    print(f"✅ Saved model at: {save_path}.zip")
