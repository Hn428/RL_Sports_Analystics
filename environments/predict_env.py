import numpy as np
import gymnasium as gym
from gymnasium import spaces


class NBAPredictEnv(gym.Env):
    def __init__(self, games_df):
        super(NBAPredictEnv, self).__init__()
        self.games_df = games_df
        self.current_step = 0

        self.state_cols = ['rolling_home_win_rate', 'home_win_streak']
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.state_cols),),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(2)  # 0 = home win, 1 = away win

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        obs = self._get_obs()
        return obs, {}  # gymnasium expects (obs, info)


    def _get_obs(self):
        row = self.games_df.iloc[self.current_step]
        return np.array(row[self.state_cols], dtype=np.float32)

    def step(self, action):
        row = self.games_df.iloc[self.current_step]
        actual_home_win = row['home_win']
        correct = (action == 0 and actual_home_win == 1) or (action == 1 and actual_home_win == 0)

        reward = 1.0 if correct else -1.0

        self.current_step += 1
        done = self.current_step >= len(self.games_df)

        obs = self._get_obs()
        terminated = done
        truncated = False  
        info = {}

        return obs, reward, terminated, truncated, info

