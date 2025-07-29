import pandas as pd
import os
import argparse
from datetime import datetime
from pathlib import Path

def process_data_for_date(target_date, shift=10, raw_folder="data/raw", output_folder="data/processed"):
    """
    Process all raw data files for a specific date:
    1. Read all parquet files containing the target date
    2. Keep only close prices
    3. Create lag features
    4. Store processed data
    
    Args:
        target_date (str): Date in format YYYY-MM-DD
        shift (int): Number of lag periods to create
        raw_folder (str): Path to raw data folder
        output_folder (str): Path to output folder
    
    Example:
        python assemble_data.py 2025-06-09 --shift 10 --raw-folder data/raw --output-folder data/processed
    """
    
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Get all parquet files containing the target date
    raw_path = Path(raw_folder)
    parquet_files = list(raw_path.glob(f"*{target_date}*.parquet"))
    
    if not parquet_files:
        print(f"No parquet files found for date {target_date} in {raw_folder}")
        return
    
    print(f"Found {len(parquet_files)} files for date {target_date}")
    
    # Read and process each symbol
    processed_dfs = {}
    
    for file_path in parquet_files:
        # Extract symbol from filename (assuming format: SYMBOL-date.parquet or similar)
        filename = file_path.stem
        symbol = filename.split('-')[0]  # Get first part before dash
        
        print(f"Processing {symbol}...")
        
        # Read data
        df = pd.read_parquet(file_path)
        
        # Convert timestamp to datetime and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Keep only close price
        if 'close' not in df.columns:
            print(f"Warning: 'close' column not found in {symbol}, skipping...")
            continue
            
        df_close = df[['close']].copy()
        
        # Normalize symbol name
        symbol_name = symbol.lower()
        if symbol == 'SPY':
            symbol_name = 'spx'
        elif symbol == 'GLD':
            symbol_name = 'gold'
        elif symbol == 'TLT':
            symbol_name = 'bond'
        
        # Rename close column to symbol name
        df_close = df_close.rename(columns={'close': symbol_name})
        
        processed_dfs[symbol_name] = df_close
    
    print(f"Processed {len(processed_dfs)} symbols")
    
    # Combine all symbols on timestamp
    if processed_dfs:
        df_all = None
        for symbol_name, df in processed_dfs.items():
            if df_all is None:
                df_all = df.copy()
            else:
                df_all = df_all.join(df, how='outer')
        
        # Handle missing data
        #print(f"Combined shape before cleaning: {df_all.shape}")
        #print(f"Missing values per column:\n{df_all.isnull().sum()}")
        
        # Forward fill small gaps, then drop remaining NaNs
        df_all = df_all.fillna(method='ffill', limit=2)
        df_all = df_all.dropna()
        
        #print(f"Shape after cleaning: {df_all.shape}")
        
        # Create lag features for all columns
        for col in df_all.columns:
            for i in range(1, shift + 1):
                df_all[f"{col}_t-{i}"] = df_all[col].shift(i)
        
        # Drop rows with NaN values (caused by lagging)
        df_all = df_all.dropna()
        
        # Normalize data based on first row (set first row values to 1.0)
        if not df_all.empty:
            first_row = df_all.iloc[0]
            df_all = df_all / first_row
            print("Data normalized based on first row values")
        
        # Filter to trading hours (1:30 PM to 8:00 PM) if data has time component
        try:
            df_all = df_all.between_time('13:30', '20:00')
        except:
            print("Warning: Could not filter by trading hours, keeping all data")
        
        # Save processed data
        if not df_all.empty:
            print(f"Final processed shape: {df_all.shape}")
            
            # Create output filename
            output_file = Path(output_folder) / f"dataset_{target_date}.parquet"
            df_all.to_parquet(output_file)
            print(f"Saved processed data to: {output_file}")
        else:
            print("No data remaining after processing")
    else:
        print("No valid data to process")


def main():
    parser = argparse.ArgumentParser(description='Process raw trading data for a specific date')
    parser.add_argument('date', type=str, help='Target date in YYYY-MM-DD format')
    parser.add_argument('--shift', type=int, default=10, help='Number of lag periods to create (default: 10)')
    parser.add_argument('--raw-folder', type=str, default='data/raw', help='Path to raw data folder (default: data/raw)')
    parser.add_argument('--output-folder', type=str, default='data/processed', help='Path to output folder (default: data/processed)')
    
    args = parser.parse_args()
    
    # Validate date format
    try:
        datetime.strptime(args.date, '%Y-%m-%d')
    except ValueError:
        print("Error: Date must be in YYYY-MM-DD format")
        return
    
    print(f"Processing data for date: {args.date}")
    print(f"Creating {args.shift} lag features")
    print(f"Raw data folder: {args.raw_folder}")
    print(f"Output folder: {args.output_folder}")
    print("-" * 50)
    
    process_data_for_date(
        target_date=args.date,
        shift=args.shift,
        raw_folder=args.raw_folder,
        output_folder=args.output_folder
    )


if __name__ == "__main__":
    main()

