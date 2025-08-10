# Data Documentation for Trading Environment

This directory contains the trading data used in the `TradingEnv` class. The data is expected to be in Parquet format and should include the following columns:

- `date`: The date of the trading data point.
- `close`: The closing price of the asset.
- Additional features such as technical indicators (e.g., RSI, MACD) may also be included.

## Data Format

The data should be structured as follows:

| date       | close | feature_1 | feature_2 | ... |
|------------|-------|-----------|-----------|-----|
| YYYY-MM-DD | float | float     | float     | ... |
| ...        | ...   | ...       | ...       | ... |

## Obtaining the Data

To obtain the trading data, you can either:

1. Download it from a financial data provider that offers historical trading data in Parquet format.
2. Convert your existing trading data from CSV or another format to Parquet using tools like Pandas.

Ensure that the data is preprocessed and cleaned before using it with the `TradingEnv` class.