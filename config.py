import os

# --- CONFIGURATION --- #
ALPACA_API_KEY = os.environ.get("APCA_API_KEY_ID", "YOUR_KEY")
ALPACA_SECRET_KEY = os.environ.get("APCA_API_SECRET_KEY", "YOUR_SECRET")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
SYMBOL = "AAPL"
TIMEFRAME = "5Min"
WINDOW = 50  # lookback window for indicators

# Q-Learning Hyperparameters
ACTIONS = ['buy', 'sell', 'hold']
ALPHA = 0.1  # learning rate
GAMMA = 0.95  # discount factor
EPSILON = 0.1  # exploration rate

# Hallucination parameters
HALLUCINATION_STEPS = 10  # Number of simulated steps per real tick
HALLUCINATION_NOISE = 0.005  # Max percent change for simulated prices
