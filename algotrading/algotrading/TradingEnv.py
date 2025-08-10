# filepath: /algotrading/algotrading/TradingEnv.py
import gym
import numpy as np
import pandas as pd
from gym import spaces

class TradingEnv(gym.Env):
    def __init__(self, parquet_path, gamma=0.9):
        super(TradingEnv, self).__init__()

        # Load your Parquet trading data
        self.df = pd.read_parquet(parquet_path)
        self.n_steps = len(self.df)

        # Define action space: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)

        # Define observation space: e.g., [close, RSI, MACD, ...]
        n_features = self.df.drop(columns=["date"]).shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )

        self.gamma = gamma  # discount factor for future reward
        self.current_step = 0
        self.cash = 10000
        self.position = 0
        self.entry_price = 0

    def reset(self):
        self.current_step = 0
        self.cash = 10000
        self.position = 0
        self.entry_price = 0
        return self._get_observation()

    def step(self, action):
        done = False
        immediate_reward = 0
        future_reward = 0

        current_price = self.df.iloc[self.current_step]["close"]

        if action == 1:  # Buy
            if self.position == 0:
                self.entry_price = current_price
                self.position = 1

        elif action == 2:  # Sell
            if self.position == 1:
                immediate_reward = current_price - self.entry_price
                self.cash += immediate_reward
                self.position = 0
                self.entry_price = 0

        # Estimate future reward if holding a position
        if self.position == 1:
            lookahead_window = 10
            end_idx = min(self.current_step + 1 + lookahead_window, self.n_steps)
            max_future_price = self.df.iloc[self.current_step + 1:end_idx]["close"].max()
            future_reward = self.gamma * (max_future_price - self.entry_price)

        reward = immediate_reward + future_reward

        self.current_step += 1
        if self.current_step >= self.n_steps - 1:
            done = True

        obs = self._get_observation()
        return obs, reward, done, {}

    def _get_observation(self):
        obs = self.df.drop(columns=["date"]).iloc[self.current_step].values
        return obs.astype(np.float32)

    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Cash: {self.cash}, Position: {self.position}")