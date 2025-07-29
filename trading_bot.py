from stable_baselines3 import PPO
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import os
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import time
from assemble_data import process_data_for_date

# Alpaca API credentials
API_KEY = "<API_KEY>"
API_SECRET = "<API_SECRET>"
client = StockHistoricalDataClient(API_KEY, API_SECRET)

# Alpaca Trading API credentials
TRADING_API_KEY = "<TRADING_API_KEY>"
TRADING_API_SECRET = "<TRADING_API_SECRET>"
trading_client = TradingClient(TRADING_API_KEY, TRADING_API_SECRET, paper=True)

# Symbols to trade
symbols = ["SPY", "GLD", "TLT"]

# Paths
MODEL_PATH = "model/RL_trading_model.zip"
RAW_FOLDER = "data/current_raw/"
PROCESSED_FOLDER = "data/current_processed/"

current_date = datetime.now().date()
TARGET_DATE = current_date.strftime("%Y-%m-%d")

def download_data():
    """
    Download the last 10 minutes of 1-minute timeframe data for the specified symbols.
    Save the data into the RAW_FOLDER.
    """
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=11)

    # Create raw data folder if it doesn't exist
    Path(RAW_FOLDER).mkdir(parents=True, exist_ok=True)

    # Fetch data from Alpaca
    request_params = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Minute,
        start=start_time,
        end=end_time
    )
    bars = client.get_stock_bars(request_params).df.reset_index()
    
    print(bars)
    # Group data by symbol and save as parquet files
    for symbol in symbols:
        symbol_data = bars[bars['symbol'] == symbol].copy()
        if not symbol_data.empty:
            file_path = os.path.join(RAW_FOLDER, f"{symbol}-{end_time.date()}.parquet")
            symbol_data.to_parquet(file_path, index=False)
            print(f"Saved {symbol} data to {file_path}")
        else:
            print(f"No data found for {symbol} in the last 10 minutes.")

def assemble_data():
    """
    Assemble raw data into a processed dataset with lag features.
    Save the processed data into the PROCESSED_FOLDER.
    """
    
    process_data_for_date(TARGET_DATE, shift=10, raw_folder=RAW_FOLDER, output_folder=PROCESSED_FOLDER)

def execute_trade(action):
    """
    Execute a paper trade based on the predicted action.
    :param action: The predicted action (e.g., 0 for hold, 1 for buy SPY, 2 for sell SPY, etc.)
    """
    # Define action mapping (example: 0 = hold, 1 = buy SPY, 2 = sell SPY)
    action_mapping = {
        0: "hold",
        1: {"symbol": "SPY", "side": OrderSide.BUY},
        2: {"symbol": "SPY", "side": OrderSide.SELL},
    }

    if action == 0:
        print("Action: Hold. No trade executed.")
        return

    trade_details = action_mapping.get(action)
    if not trade_details:
        print(f"Invalid action: {action}")
        return

    symbol = trade_details["symbol"]
    side = trade_details["side"]

    # Create a market order
    order = MarketOrderRequest(
        symbol=symbol,
        qty=1,  # Example: trade 1 share
        side=side,
        time_in_force=TimeInForce.GTC
    )

    try:
        # Submit the order
        trade_response = trading_client.submit_order(order)
        print(f"Executed {side} order for {symbol}: {trade_response}")
    except Exception as e:
        print(f"Error executing trade: {e}")

def predict_action():
    """
    Load the RL model and predict the action based on the last 10 minutes of processed data.
    Execute the predicted action as a paper trade.
    """
    # Load the model
    model = PPO.load(MODEL_PATH)
    print("Model loaded successfully.")

    # Load processed data
    processed_file = os.path.join(PROCESSED_FOLDER, f"dataset_{TARGET_DATE}.parquet")
    if not os.path.exists(processed_file):
        print("Processed data file not found.")
        return

    df = pd.read_parquet(processed_file)
    print(df)
    # Check if the DataFrame is empty
    if df.empty:
        print("Processed data is empty. Skipping prediction.")
        return

    # Extract the last row as the observation
    obs = df.iloc[-1].values.reshape(1, -1)

    # Predict the action
    action, _ = model.predict(obs)
    print(f"Predicted action: {action}")

    # Execute the trade based on the predicted action
    execute_trade(action[0])

if __name__ == "__main__":
    while True:
        # Step 1: Download data
        download_data()

        # Step 2: Assemble data
        assemble_data()

        # Step 3: Predict action and execute trade
        predict_action()

        # Wait for a minute before the next iteration
        time.sleep(60)


