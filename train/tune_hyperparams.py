import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import optuna
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from environments.predict_env import NBAPredictEnv

def optimize_ppo(trial):
    # Load data
    games_df = pd.read_csv("nbaData/enhanced_games.csv", low_memory=False)
    team_stats_df = pd.read_csv("nbaData/TeamStatistics.csv", low_memory=False)
    env = NBAPredictEnv(games_df, team_stats_df)

    # Define hyperparameter search space
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99)
    ent_coef = trial.suggest_float("ent_coef", 0.001, 0.02)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    n_epochs = trial.suggest_int("n_epochs", 5, 15)

    policy_kwargs = dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        clip_range=clip_range,
        n_epochs=n_epochs,
        policy_kwargs=policy_kwargs,
        verbose=0,
    )

    model.learn(total_timesteps=50_000)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
    return mean_reward

# Run Optuna
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(optimize_ppo, n_trials=20)

    print("✅ Best hyperparameters found:")
    print(study.best_params)
