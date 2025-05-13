import numpy as np
import gymnasium as gym
from gymnasium import spaces

def normalize_obs(obs):
    return np.nan_to_num(
        obs / np.array([1.0, 10.0]), nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)

class NBAPredictEnv(gym.Env):
    def __init__(self, games_df):
        super(NBAPredictEnv, self).__init__()
        self.games_df = games_df
        self.current_step = 0

        # Define action space: 0 = predict loss, 1 = predict win
        self.action_space = spaces.Discrete(2)

        # Observation space: [rolling_home_win_rate, home_win_streak]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )

    def _get_obs(self):
        if self.current_step >= len(self.games_df):
            return np.array([0.0, 0.0], dtype=np.float32)

        row = self.games_df.iloc[self.current_step]
        obs = np.array([
            row["rolling_home_win_rate"],
            row["home_win_streak"]
        ])
        return normalize_obs(obs)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        done = False
        reward = 0.0

        if self.current_step >= len(self.games_df):
            done = True
            return self._get_obs(), reward, done, False, {}

        row = self.games_df.iloc[self.current_step]
        predicted = int(action)
        actual = int(row["home_team_win"])

        reward = 1.0 if predicted == actual else -1.0

        self.current_step += 1
        done = self.current_step >= len(self.games_df)

        return self._get_obs(), reward, done, False, {}

    def render(self):
        print(f"Step: {self.current_step}")
