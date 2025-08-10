import os
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import timedelta



def create_client():

    API_KEY = os.getenv("ALPACA_API_KEY")
    API_SECRET = os.getenv("ALPACA_API_SECRET")

    API_KEY = "PKP6BC14573XB64GYB78"
    API_SECRET = "hfD0n2ORbvpucdMNZj4tHfRlXwgUvrREMaFlnIMr"

    trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
    return StockHistoricalDataClient(API_KEY, API_SECRET)


def get_minute_history(client, symbol: str, minutes: int = 100, end: pd.Timestamp = None):

    if end is None:
        end = pd.Timestamp.utcnow()
    start = end - pd.Timedelta(minutes=minutes)
    print("start fetch data")
    start = end - timedelta(minutes=minutes)
    #timeframe = TimeFrame(1,'Minute')
    req = StockBarsRequest(symbol_or_symbols=symbol, start=start, end=end, timeframe=TimeFrame.Minute)
    bars = client.get_stock_bars(req).df
    #print(bars)
    bars = bars.xs(symbol, level=0)
    return bars  # includes open, high, low, close, volume

def fetch_bars(symbols, start, end, timeframe=TimeFrame.Minute):
    req = StockBarsRequest(symbol_or_symbols=symbols, start=start, end=end, timeframe=timeframe)
    df = client.get_stock_bars(req).df
    return df.reset_index().pivot(index="timestamp", columns="symbol", values="close").dropna()

def bollinger_signal(bars: pd.DataFrame, window: int = 20, num_std: float = 2.0):
    df = bars.copy()
    df["sma"] = df["close"].rolling(window).mean()
    df["std"] = df["close"].rolling(window).std()
    df["upper"] = df["sma"] + num_std * df["std"]
    df["lower"] = df["sma"] - num_std * df["std"]

    latest = df.iloc[-1]
    previous = df.iloc[-2]

    # Check cross above upper band (Sell signal)
    if previous["close"] <= previous["upper"] and latest["close"] > latest["upper"]:
        return "sell"

    # Check cross below lower band (Buy signal)
    elif previous["close"] >= previous["lower"] and latest["close"] < latest["lower"]:
        return "buy"

    return "hold"

if __name__ ==  "__main__":
    custom_end = pd.Timestamp("2025-06-20 08:30:00", tz="UTC")
    client = create_client()
    bars = get_minute_history(client, "TSLA", 50, custom_end)
    signal = bollinger_signal(bars)
    print(signal)