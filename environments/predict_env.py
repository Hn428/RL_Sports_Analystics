import numpy as np
import gymnasium as gym
from gymnasium import spaces

def normalize_obs(obs):
    return np.nan_to_num(
        obs / np.array([1.0, 10.0, 1.0, 10.0, 1.0, 1.0, 1.0]),  # empirical scaling
        nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)

class NBAPredictEnv(gym.Env):
    def __init__(self, games_df, team_stats_df):
        super(NBAPredictEnv, self).__init__()
        self.games_df = games_df
        self.team_stats_df = team_stats_df
        self.current_step = 0

        # Action space: predict outcome and spread
        self.action_space = spaces.Discrete(5)  # 0–4 categories

        # Observation: team win rates, streaks, stat differentials
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )

    def _get_team_stats(self, game_id, home_id, away_id):
        """Returns fg%, rebounds, and turnover diff between teams."""
        game_stats = self.team_stats_df[self.team_stats_df['gameId'] == game_id]

        home_stats = game_stats[game_stats['teamId'] == home_id]
        away_stats = game_stats[game_stats['teamId'] == away_id]

        def safe_extract(df, col):
            return df[col].values[0] if col in df and not df.empty else 0.0

        fg_pct_diff = safe_extract(home_stats, 'fieldGoalsPercentage') - safe_extract(away_stats, 'fieldGoalsPercentage')
        rebound_diff = safe_extract(home_stats, 'reboundsTotal') - safe_extract(away_stats, 'reboundsTotal')
        turnover_diff = safe_extract(home_stats, 'turnovers') - safe_extract(away_stats, 'turnovers')

        return fg_pct_diff, rebound_diff, turnover_diff

    def _get_obs(self):
        if self.current_step >= len(self.games_df):
            return np.zeros(7, dtype=np.float32)

        row = self.games_df.iloc[self.current_step]
        fg_pct_diff, rebound_diff, turnover_diff = self._get_team_stats(
            row["gameId"], row["hometeamId"], row["awayteamId"]
        )

        obs = np.array([
            row["rolling_home_win_rate"],
            1.0 - row["rolling_home_win_rate"],  # approximate away win rate
            row["home_win_streak"],
            -row["home_win_streak"],  # assume inverse for away
            fg_pct_diff,
            rebound_diff,
            turnover_diff
        ])
        return normalize_obs(obs)

    def reset(self, seed=None, options=None):
        self.current_step = np.random.randint(0, len(self.games_df))  # random game start
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        if self.current_step >= len(self.games_df):
            return self._get_obs(), 0.0, True, False, {}

        row = self.games_df.iloc[self.current_step]
        home_score = row["homeScore"]
        away_score = row["awayScore"]
        actual_spread = home_score - away_score

        actual_class = (
            0 if actual_spread > 10 else
            1 if actual_spread > 0 else
            2 if abs(actual_spread) <= 5 else
            3 if actual_spread < 0 and abs(actual_spread) <= 10 else
            4
        )

        reward = 2.0 if int(action) == actual_class else -1.0

        self.current_step += 1
        terminated = self.current_step >= len(self.games_df)
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}
