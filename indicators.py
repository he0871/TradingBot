import pandas as pd
from config import WINDOW

def get_indicators(df):
    # Compute indicators: SMA, EMA, RSI (as in ML4T)
    df['SMA'] = df['close'].rolling(window=WINDOW).mean()
    df['EMA'] = df['close'].ewm(span=WINDOW).mean()
    delta = df['close'].diff() 
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    avg_gain = up.rolling(window=14, min_periods=1).mean()
    avg_loss = down.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    df.fillna(method='bfill', inplace=True)
    return df
