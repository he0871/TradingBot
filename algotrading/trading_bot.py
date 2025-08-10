from stable_baselines3 import PPO
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import os
import glob

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import time

# Alpaca API credentials
API_KEY = "PKNWZK2CTTZO5T8ZGEZV"
API_SECRET = "<secret>"
client = StockHistoricalDataClient(API_KEY, API_SECRET)

# Alpaca Trading API credentials
TRADING_API_KEY = "PKNWZK2CTTZO5T8ZGEZV"
TRADING_API_SECRET = "<secret>"
trading_client = TradingClient(TRADING_API_KEY, TRADING_API_SECRET, paper=True)

# Symbols to trade
symbols = ["SPY", "GLD", "TLT"]

# Paths
MODEL_PATH = "model/ppo_trading_model.zip"
RAW_FOLDER = "data/current_raw/"
PROCESSED_FOLDER = "data/current_processed/"

current_date = datetime.now().date()
TARGET_DATE = current_date.strftime("%Y-%m-%d")

current_position = 0
entry_price = 0
pending_sell_order_id = None

def download_data():
    """
    Download data from the beginning of today for the specified symbols.
    Save the data into the RAW_FOLDER.
    """
    end_time = datetime.now()
    # Set start time to beginning of today (midnight)
    start_time = datetime.combine(end_time.date(), datetime.min.time())

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
            print(f"No data found for {symbol} from beginning of today.")

def assemble_data():
    """
    Process raw data to calculate normalized values and technical indicators.
    Save the processed data into the PROCESSED_FOLDER.
    """
    # Create processed folder if it doesn't exist
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)
    
    # Get all SPY parquet files from the current raw folder
    spy_files = glob.glob(os.path.join(RAW_FOLDER, f"SPY-{TARGET_DATE}.parquet"))
    
    if not spy_files:
        print(f"No SPY data found for {TARGET_DATE} in {RAW_FOLDER}")
        return
    
    # Import the indicator processing functions
    from indicator import normalize_data, calculate_all_indicators
    
    # Process each file
    for file_path in spy_files:
        print(f"Processing {file_path}...")
        
        # Read parquet file
        df = pd.read_parquet(file_path)
        
        # Normalize data
        normalized_df = normalize_data(df)
        
        # Calculate indicators
        df_with_indicators = calculate_all_indicators(normalized_df)
        
        # Drop rows with NaN values in the indicators
        df_clean = df_with_indicators.dropna(subset=['rsi', 'macd', 'macd_signal', 'macd_histogram', 'cci'])
        
        # Extract filename
        filename = os.path.basename(file_path)
        
        # Save to processed folder
        output_path = os.path.join(PROCESSED_FOLDER, filename.replace('.parquet', '-processed.parquet'))
        df_clean.to_parquet(output_path)
        
        print(f"Saved processed data to {output_path}")
    
    print(f"Processed SPY data for {TARGET_DATE}")


def execute_trade(action, current_price):
    """
    Execute a paper trade based on the predicted action.
    :param action: The predicted action (0 for check status, 1 for buy)
    :param current_price: Current stock price
    """
    global current_position, entry_price, pending_sell_order_id
    
    if action == 0:
        # Check if sell order is filled
        if pending_sell_order_id:
            try:
                order_status = trading_client.get_order_by_id(pending_sell_order_id)
                if order_status.status == 'filled':
                    print(f"Sell order {pending_sell_order_id} filled!")
                    current_position = 0
                    entry_price = 0
                    pending_sell_order_id = None
                    print(f"Position updated: Now holding {current_position} shares")
                else:
                    print(f"Sell order {pending_sell_order_id} status: {order_status.status}")
            except Exception as e:
                print(f"Error checking order status: {e}")
        else:
            print("Action: Hold. No pending orders to check.")
        return

    elif action == 1:
        # Buy stock and immediately place sell order at buying price + 1
        if current_position > 0:
            print(f"Action: Buy blocked. Already holding {current_position} shares.")
            return
        
        symbol = "SPY"
        
        # Create a market buy order
        buy_order = MarketOrderRequest(
            symbol=symbol,
            qty=1,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC
        )

        try:
            # Submit the buy order
            buy_response = trading_client.submit_order(buy_order)
            print(f"Executed BUY order for {symbol}: {buy_response}")
            
            # Update position and entry price
            current_position = 1
            entry_price = current_price
            print(f"Position updated: Now holding {current_position} shares at entry price {entry_price}")
            
            # Immediately place a limit sell order at entry_price + 1
            sell_price = entry_price + 1
            sell_order = LimitOrderRequest(
                symbol=symbol,
                qty=1,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
                limit_price=sell_price
            )
            
            sell_response = trading_client.submit_order(sell_order)
            pending_sell_order_id = sell_response.id
            print(f"Placed SELL limit order at ${sell_price}: {sell_response}")
            
        except Exception as e:
            print(f"Error executing trade: {e}")
    
    else:
        print(f"Invalid action: {action}")

def predict_action():
    """
    Load the RL model and predict the action based on the processed data.
    Execute the predicted action as a paper trade.
    """
    global entry_price, current_position
    
    # Load the model
    model = PPO.load(MODEL_PATH)
    print("Model loaded successfully.")

    # Load processed data
    processed_file = os.path.join(PROCESSED_FOLDER, f"SPY-{TARGET_DATE}-processed.parquet")
    if not os.path.exists(processed_file):
        print(f"Processed data file not found: {processed_file}")
        return

    df = pd.read_parquet(processed_file)
    print(f"Loaded data with {len(df)} rows")
    
    # Check if the DataFrame is empty
    if df.empty:
        print("Processed data is empty. Skipping prediction.")
        return

    # Get the feature columns we need for the observation
    feature_columns = ['norm_open', 'norm_high', 'norm_low', 'norm_close', 'norm_volume', 
                      'rsi', 'macd', 'macd_signal', 'macd_histogram', 'cci']
    
    # Verify all feature columns exist in the dataframe
    missing_columns = [col for col in feature_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing columns in data: {missing_columns}")
        return
    
    # Extract the last row as the observation
    last_row = df.iloc[-1]
    features = last_row[feature_columns].values
    
    # Calculate entry price ratio
    current_price = last_row['close']
    if entry_price > 0 and current_price > 0:
        ratio = entry_price / current_price
    else:
        ratio = -1  # Default ratio when no position or invalid prices

    # Create the observation vector (features + ratio + position)
    # Convert features to float32 to ensure compatibility with the model
    features = features.astype(np.float32)
    obs = np.append(features, [ratio, current_position]).astype(np.float32)
    
    # Predict the action
    action, _ = model.predict(obs, deterministic=True)
    
    # Convert numpy array to scalar integer
    action = int(action) if isinstance(action, np.ndarray) else int(action)
    
    print(f"Predicted action: {action}, Current position: {current_position}, Current price: {current_price}")

    # Execute the trade based on the predicted action
    execute_trade(action, current_price)

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


