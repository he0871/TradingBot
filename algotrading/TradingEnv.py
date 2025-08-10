import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
import random
import glob
import os

class TradingEnv(gym.Env):
    def __init__(self, processed_data_path="data/processed"):
        super(TradingEnv, self).__init__()

        # Load all processed parquet files
        parquet_files = glob.glob(os.path.join(processed_data_path, "SPY-*-processed.parquet"))
        
        if not parquet_files:
            raise ValueError(f"No processed parquet files found in {processed_data_path}")
        
        # Create list of dataframes from all processed parquet files
        self.envs = []
        for file in parquet_files:
            df = pd.read_parquet(file)
            # Ensure timestamp is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                # Extract date from timestamp
                df['date'] = df['timestamp'].dt.date
            self.envs.append(df)
        
        print(f"Loaded {len(self.envs)} days of trading data")
        
        # Use the first dataframe for initialization
        self.df = self.envs[0] if self.envs else pd.DataFrame()
        self.current_env_idx = 0
        
        # Define action space: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)

        # Define observation space using normalized features and indicators
        # We'll use normalized OHLC, volume, RSI, MACD, and CCI
        feature_columns = ['norm_open', 'norm_high', 'norm_low', 'norm_close', 'norm_volume', 
                          'rsi', 'macd', 'macd_signal', 'macd_histogram', 'cci']
        
        # Verify all feature columns exist in the dataframe
        missing_columns = [col for col in feature_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in data: {missing_columns}")
        
        # Add 2 for position and entry_price_ratio features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(feature_columns) + 2,), dtype=np.float32
        )
        
        # Trading state
        self.cash = 10000
        self.position = 0  # 0 = no position, 1 = long position
        self.entry_price = 0
        self.current_step = 0
        self.feature_columns = feature_columns

    def reset(self, seed=None, options=None):
        # Set seed if provided
        if seed is not None:
            random.seed(seed)
            
        # Select a random day
        self.current_env_idx = random.randint(0, len(self.envs) - 1)
        self.df = self.envs[self.current_env_idx]
        
        # Reset trading state
        self.cash = 10000
        self.position = 0
        self.entry_price = 0
        self.current_step = 0
        
        return self._get_observation(), {}

    def step(self, action):
        # Initialize return values
        done = False
        reward = 0
        truncated = False
        
        # Get current price
        current_price = self.df.iloc[self.current_step]["close"]

        # Process action
        if action == 1:  # Buy
            if self.position == 0:
                self.entry_price = current_price
                self.position = 1

        elif action == 2:  # Sell
            if self.position == 1:
                # Calculate profit/loss
                reward = current_price - self.entry_price
                self.cash += reward
                self.position = 0
                self.entry_price = 0

        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        if self.current_step >= len(self.df) - 1:
            done = True
            
            # If we still have a position at the end, close it
            if self.position == 1:
                final_price = self.df.iloc[-1]["close"]
                reward += final_price - self.entry_price
                self.cash += reward

        # Get new observation
        obs = self._get_observation()
        
        # Additional info dictionary
        info = {
            "cash": self.cash,
            "position": self.position,
            "current_price": current_price
        }
        
        return obs, reward, done, truncated, info

    def _get_observation(self):
        # Get feature values
        if self.current_step < len(self.df):
            obs = self.df.iloc[self.current_step][self.feature_columns].values
            
            # Calculate ratio: entry_price / current_price
            current_price = self.df.iloc[self.current_step]['close']
            if self.entry_price > 0 and current_price > 0:
                ratio = self.entry_price / current_price
            else:
                ratio = 0  # Default ratio when no position
            
            # Append position and ratio to observation
            obs = np.append(obs, [self.position, ratio])
            return obs.astype(np.float32)
        else:
            # Return zeros if current_step is out of bounds
            return np.zeros(len(self.feature_columns) + 2, dtype=np.float32)

    def render(self):
        if self.current_step < len(self.df):
            timestamp = self.df.iloc[self.current_step]['timestamp']
            price = self.df.iloc[self.current_step]['close']
            print(f"Date: {timestamp}, Step: {self.current_step}, Price: {price:.2f}, "
                  f"Cash: {self.cash:.2f}, Position: {self.position}")
        else:
            print("Episode finished")
            
    def close(self):
        pass
