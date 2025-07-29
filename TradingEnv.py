import gym
import numpy as np
import pandas as pd
from gym import spaces
import random
import glob
import os

class TradingEnv(gym.Env):
    def __init__(self, parquet_path):
        super(TradingEnv, self).__init__()

        # Load all parquet files
        parquet_files = glob.glob(os.path.join(parquet_path, "*.parquet"))
        
        # Create list of dataframes from all parquet files
        self.envs = []
        for file in parquet_files:
            df = pd.read_parquet(file)
            df['timestamp'] = pd.to_datetime(df.index)
            df['date'] = df['timestamp'].dt.date
            self.envs.append(df)
        
        # Use the first dataframe for initialization (or you could combine them)
        self.df = self.envs[0] if self.envs else pd.DataFrame()
        self.current_env_idx = 0
        
        # Define action space: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)

        # Define observation space: e.g., [close, RSI, MACD, ...]
        print(len(self.envs))
        feature_columns = [col for col in self.df.columns if col not in ['timestamp', 'date']]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(feature_columns),), dtype=np.float32
        )

        self.cash = 10000
        self.position = 0
        self.entry_price = 0
        self.current_step = 0

    def reset(self):

        # Select a random date
        self.current_env_idx = random.randint(0, len(self.envs) - 1)
        self.df = self.envs[self.current_env_idx]
        
        # Find the first row of that date

        self.cash = 10000
        self.position = 0
        self.entry_price = 0
        self.count = 0  
        self.current_step = 0 
        return self._get_observation()

    def step(self, action):
        done = False
        reward = 0
 
        current_price = self.df.iloc[self.current_step]["spx"]

        if action == 1:  # Buy
            if self.position == 0:
                self.entry_price = current_price
                self.position = 1

        elif action == 2:  # Sell
            if self.position == 1:
                reward = current_price - self.entry_price
                self.cash += reward
                self.position = 0
                self.entry_price = 0


        self.current_step += 1
        if self.current_step >= self.df.shape[0] - 2:
            done = True

        obs = self._get_observation()
        return obs, reward, done, {}

    def _get_observation(self):
        # Get all columns except timestamp and date
        feature_columns = [col for col in self.df.columns if col not in ['timestamp', 'date']]
        #print(self.df.shape)
        #print(self.current_step)
        obs = self.df.iloc[self.current_step][feature_columns].values
        #print(obs)
        return obs.astype(np.float32)

    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Cash: {self.cash}, Position: {self.position}")
