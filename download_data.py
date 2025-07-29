from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
import pandas as pd
import os

# Setup Alpaca credentials
API_KEY = "<API_KEY>"
API_SECRET = "<API_SECRET>"
client = StockHistoricalDataClient(API_KEY, API_SECRET)

# Symbols: SPY (S&P 500 ETF), GLD (Gold ETF), TLT (Bond ETF)
symbols = ["SPY", "GLD", "TLT"]

# Get date range: last 10 trading days
end_date = datetime.now() - timedelta(days=60)
start_date = end_date - timedelta(days=67)  # extra buffer for weekends

# Create data directory if it doesn't exist
data_dir = "/Users/jingyuanhe/code/algotrading/data/raw/"
os.makedirs(data_dir, exist_ok=True)

# Download 1-minute data for all symbols
def fetch_all_symbols_data():
    request_params = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Minute,
        start=start_date,
        end=end_date
    )
    bars = client.get_stock_bars(request_params).df
    return bars

# Group data by date and symbol, save one parquet per symbol per day
if __name__ == '__main__':
    # Fetch all data at once
    all_data = fetch_all_symbols_data()
    
    # Reset index to work with timestamp
    all_data = all_data.reset_index()
    
    # Convert timestamp to date for grouping
    all_data['date'] = all_data['timestamp'].dt.date
    
    # Group by date and symbol, save each symbol's data separately
    for date, day_data in all_data.groupby('date'):
        for symbol in symbols:
            # Filter data for this symbol
            symbol_data = day_data[day_data['symbol'] == symbol].copy()
            
            if not symbol_data.empty:
                # Remove the date column before saving
                symbol_data_clean = symbol_data.drop(['date', 'symbol'], axis=1)
                
                # Save to parquet file named by symbol and date
                filename = f"{data_dir}/{symbol}-{date}.parquet"
                symbol_data_clean.to_parquet(filename, index=False)
                
                print(f"Saved {symbol} for {date}: {len(symbol_data_clean)} records")
            else:
                print(f"No data for {symbol} on {date}")
    
    print(f"Data saved from {start_date.date()} to {end_date.date()}")