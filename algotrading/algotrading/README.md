# Trading Environment for Algorithmic Trading

This project implements a custom trading environment using OpenAI Gym, designed for algorithmic trading strategies. The environment simulates trading actions and calculates rewards based on the performance of those actions.

## Project Structure

- **TradingEnv.py**: Contains the `TradingEnv` class, which defines the trading environment, including methods for resetting the environment, taking actions (buy, sell, hold), and calculating rewards.

- **data/**: This directory contains the trading data used by the environment. Refer to `data/README.md` for details on the data format and how to obtain it.

- **tests/**: This directory includes unit tests for the `TradingEnv` class. The tests are located in `tests/test_trading_env.py` and verify the functionality of the environment.

- **requirements.txt**: Lists the required Python packages for the project. Use this file to install the necessary dependencies with pip.

## Installation

To set up the project, clone the repository and install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

To use the `TradingEnv`, create an instance of the environment by providing the path to your trading data in Parquet format:

```python
from TradingEnv import TradingEnv

env = TradingEnv('path/to/your/data.parquet')
```

You can then reset the environment, take actions, and render the results as needed.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.