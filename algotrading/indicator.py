import pandas as pd
import numpy as np
import os
import glob
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.trend import CCIIndicator
from sklearn.preprocessing import MinMaxScaler

def load_spy_data(raw_folder="data/raw"):
    """
    Load all SPY data from parquet files in the raw folder.
    
    Args:
        raw_folder (str): Path to raw data folder
        
    Returns:
        dict: Dictionary with date as key and dataframe as value
    """
    # Get all SPY parquet files
    spy_files = glob.glob(os.path.join(raw_folder, "SPY-*.parquet"))
    
    if not spy_files:
        raise ValueError(f"No SPY parquet files found in {raw_folder}")
    
    # Dictionary to store dataframes by date
    data_by_date = {}
    
    # Read each file and store in dictionary
    for file_path in spy_files:
        # Extract date from filename
        filename = os.path.basename(file_path)
        date_str = filename.replace('SPY-', '').replace('.parquet', '')
        
        # Read parquet file
        df = pd.read_parquet(file_path)
        
        # Store in dictionary
        data_by_date[date_str] = df
    
    return data_by_date

def normalize_data(df):
    """
    Normalize the OHLC and volume data using Min-Max scaling.
    
    Args:
        df (pd.DataFrame): Dataframe with price data
        
    Returns:
        pd.DataFrame: Dataframe with normalized data and original data
    """
    # Create a copy of the dataframe
    result_df = df.copy()
    
    # Columns to normalize
    cols_to_normalize = ['open', 'high', 'low', 'close', 'volume']
    
    # Initialize scaler
    scaler = MinMaxScaler()
    
    # Normalize each column and add as new column
    for col in cols_to_normalize:
        if col in result_df.columns:
            # Create new column name
            norm_col = f'norm_{col}'
            
            # Reshape data for scaler
            values = result_df[col].values.reshape(-1, 1)
            
            # Fit and transform
            normalized_values = scaler.fit_transform(values)
            
            # Add to dataframe
            result_df[norm_col] = normalized_values
    
    return result_df

def calculate_rsi(df, window=14, column='close'):
    """
    Calculate the Relative Strength Index (RSI) for the given dataframe.
    
    Args:
        df (pd.DataFrame): Dataframe with price data
        window (int): The window period for RSI calculation
        column (str): The column name to use for calculation
        
    Returns:
        pd.DataFrame: Original dataframe with RSI column added
    """
    # Create a copy of the dataframe to avoid modifying the original
    result_df = df.copy()
    
    # Calculate RSI
    rsi_indicator = RSIIndicator(close=result_df[column], window=window)
    result_df['rsi'] = rsi_indicator.rsi()
    
    return result_df

def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9, column='close'):
    """
    Calculate the Moving Average Convergence Divergence (MACD) for the given dataframe.
    
    Args:
        df (pd.DataFrame): Dataframe with price data
        fast_period (int): The fast period for MACD calculation
        slow_period (int): The slow period for MACD calculation
        signal_period (int): The signal period for MACD calculation
        column (str): The column name to use for calculation
        
    Returns:
        pd.DataFrame: Original dataframe with MACD columns added
    """
    # Create a copy of the dataframe to avoid modifying the original
    result_df = df.copy()
    
    # Calculate MACD
    macd_indicator = MACD(
        close=result_df[column],
        window_fast=fast_period,
        window_slow=slow_period,
        window_sign=signal_period
    )
    
    result_df['macd'] = macd_indicator.macd()
    result_df['macd_signal'] = macd_indicator.macd_signal()
    result_df['macd_histogram'] = macd_indicator.macd_diff()
    
    return result_df

def calculate_cci(df, window=20, constant=0.015, high_col='high', low_col='low', close_col='close'):
    """
    Calculate the Commodity Channel Index (CCI) for the given dataframe.
    
    Args:
        df (pd.DataFrame): Dataframe with price data
        window (int): The window period for CCI calculation
        constant (float): The constant value for CCI calculation
        high_col (str): The column name for high prices
        low_col (str): The column name for low prices
        close_col (str): The column name for close prices
        
    Returns:
        pd.DataFrame: Original dataframe with CCI column added
    """
    # Create a copy of the dataframe to avoid modifying the original
    result_df = df.copy()
    
    # Calculate CCI
    cci_indicator = CCIIndicator(
        high=result_df[high_col],
        low=result_df[low_col],
        close=result_df[close_col],
        window=window,
        constant=constant
    )
    
    result_df['cci'] = cci_indicator.cci()
    
    return result_df

def calculate_all_indicators(df):
    """
    Calculate all indicators (RSI, MACD, CCI) for the given dataframe.
    
    Args:
        df (pd.DataFrame): Dataframe with price data
        
    Returns:
        pd.DataFrame: Dataframe with all indicators added
    """
    # Calculate RSI
    df = calculate_rsi(df)
    
    # Calculate MACD
    df = calculate_macd(df)
    
    # Calculate CCI
    df = calculate_cci(df)
    
    return df

def process_and_save_data(raw_folder="data/raw", processed_folder="data/processed"):
    """
    Process SPY data to normalize, calculate indicators, and save results by day.
    
    Args:
        raw_folder (str): Path to raw data folder
        processed_folder (str): Path to output folder for processed data
    """
    # Create processed folder if it doesn't exist
    os.makedirs(processed_folder, exist_ok=True)
    
    # Load SPY data by date
    spy_data_by_date = load_spy_data(raw_folder)
    
    # Process each day's data
    for date_str, df in spy_data_by_date.items():
        print(f"Processing data for {date_str}...")
        
        # Normalize data
        normalized_df = normalize_data(df)
        
        # Calculate indicators
        df_with_indicators = calculate_all_indicators(normalized_df)
        
        # Drop rows with NaN values in the indicators
        df_clean = df_with_indicators.dropna(subset=['rsi', 'macd', 'macd_signal', 'macd_histogram', 'cci'])
        
        # Save to processed folder
        output_path = os.path.join(processed_folder, f"SPY-{date_str}-processed.parquet")
        df_clean.to_parquet(output_path)
        
        print(f"Saved processed data to {output_path}")
    
    print(f"\nProcessed {len(spy_data_by_date)} days of SPY data")

def main():
    """
    Main function with command-line interface.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate trading indicators for SPY data')
    parser.add_argument('--raw-folder', type=str, default='data/raw', 
                        help='Path to raw data folder (default: data/raw)')
    parser.add_argument('--processed-folder', type=str, default='data/processed', 
                        help='Path to output folder for processed data (default: data/processed)')
    
    args = parser.parse_args()
    
    # Process and save data with command-line arguments
    process_and_save_data(
        raw_folder=args.raw_folder,
        processed_folder=args.processed_folder
    )

if __name__ == "__main__":
    main()