import numpy as np
import gymnasium as gym
from gymnasium import spaces

def normalize_obs(obs):
    return np.nan_to_num(
        obs / np.array([1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 20.0, 20.0, 1.0, 1.0, 150.0, 150.0, 1.0, 1.0, 1.0]),
        nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)

class NBAPredictEnv(gym.Env):
    def __init__(self, games_df, team_stats_df):
        super().__init__()
        self.games_df = games_df
        self.team_stats_df = team_stats_df
        self.current_step = 0

        # Action space: Spread and Total Points predictions only
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)

    def _get_team_stats(self, game_id, home_id, away_id):
        game_stats = self.team_stats_df[self.team_stats_df['gameId'] == game_id]
        home_stats = game_stats[game_stats['teamId'] == home_id]
        away_stats = game_stats[game_stats['teamId'] == away_id]

        def safe(df, col): return df[col].values[0] if col in df and not df.empty else 0.0

        return (
            safe(home_stats, 'fieldGoalsPercentage') - safe(away_stats, 'fieldGoalsPercentage'),
            safe(home_stats, 'reboundsTotal') - safe(away_stats, 'reboundsTotal'),
            safe(home_stats, 'turnovers') - safe(away_stats, 'turnovers')
        )

    def _get_obs(self):
        if self.current_step >= len(self.games_df):
            return np.zeros(15, dtype=np.float32)

        row = self.games_df.iloc[self.current_step]
        fg_diff, reb_diff, tov_diff = self._get_team_stats(row["gameId"], row["hometeamId"], row["awayteamId"])

        obs = np.array([
            row["home_rolling_win_rate_5"],
            row["home_rolling_win_rate_10"],
            row["away_rolling_win_rate_5"],
            row["away_rolling_win_rate_10"],
            row["home_win_streak_overall"],
            row["away_win_streak_overall"],
            row["home_rolling_point_diff_10"],
            row["away_rolling_point_diff_10"],
            row["home_team_h2h_win_rate_10"],
            row["away_team_h2h_win_rate_10"],
            row["home_rolling_points_scored_10"],
            row["home_rolling_points_allowed_10"],
            fg_diff, reb_diff, tov_diff
        ])
        return normalize_obs(obs)

    def reset(self, seed=None, options=None):
        self.current_step = np.random.randint(0, len(self.games_df))
        return self._get_obs(), {}

    def step(self, action):
        if self.current_step >= len(self.games_df):
            return self._get_obs(), 0.0, True, False, {}

        row = self.games_df.iloc[self.current_step]
        actual_spread = row["homeScore"] - row["awayScore"]
        actual_total = row["homeScore"] + row["awayScore"]

        # Action scaling
        pred_spread = -25 + ((action[0] + 1) / 2.0) * 50
        pred_total = 175 + ((action[1] + 1) / 2.0) * 100

        spread_reward = -abs(pred_spread - actual_spread)
        total_reward = -abs(pred_total - actual_total)

        reward = (
            0.3 * spread_reward +
            0.7 * total_reward
        )

        self.current_step += 1
        done = self.current_step >= len(self.games_df)
        return self._get_obs(), reward, done, False, {}
