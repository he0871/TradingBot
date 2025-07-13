import time
import logging
import pandas as pd
import alpaca_trade_api as tradeapi
from collections import deque
from agent import QLearningAgent
from indicators import get_indicators
from hallucination import hallucinate_step
from config import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, SYMBOL, TIMEFRAME, WINDOW,
    ACTIONS, HALLUCINATION_STEPS
)

LOG_FILE = "trading_bot.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def reward_function(position, price, prev_price):
    if prev_price is None:
        return 0
    if position == 1:  # long
        return price - prev_price
    elif position == -1:
        return prev_price - price
    else:
        return 0

class Trader:
    def __init__(self):
        self.api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, 'v2')
        self.agent = QLearningAgent()
        self.position = 0  # 1: long, 0: flat
        self.prev_price = None
        self.price_history = deque(maxlen=WINDOW+20)
        logger.info("Trading bot started.")

    def run(self):
        while True:
            try:
                barset = self.api.get_bars(SYMBOL, TIMEFRAME, limit=WINDOW+20).df
                if barset.empty:
                    logger.warning("No data returned. Skipping iteration.")
                    time.sleep(60)
                    continue

                prices = barset[barset['symbol'] == SYMBOL]
                prices = prices.reset_index()
                df = prices[['close']]
                self.price_history.extend(df['close'].tolist())
                df = pd.DataFrame({'close': list(self.price_history)})
                df = get_indicators(df)
                indicators = df.iloc[-1][['SMA', 'EMA', 'RSI']].values

                # Q-Learning step (real tick)
                state = self.agent.get_state(indicators)
                action_idx = self.agent.choose_action(state)
                action = ACTIONS[action_idx]
                current_price = df.iloc[-1]['close']

                # Execute trade based on action
                if action == 'buy' and self.position == 0:
                    self.api.submit_order(symbol=SYMBOL, qty=1, side='buy', type='market', time_in_force='gtc')
                    self.position = 1
                    logger.info(f"BUY executed at {current_price:.2f}")
                elif action == 'sell' and self.position == 1:
                    self.api.submit_order(symbol=SYMBOL, qty=1, side='sell', type='market', time_in_force='gtc')
                    self.position = 0
                    logger.info(f"SELL executed at {current_price:.2f}")

                reward = reward_function(self.position, current_price, self.prev_price)
                next_state = self.agent.get_state(indicators)
                self.agent.update(state, action_idx, reward, next_state)
                logger.info(
                    f"Step info | Position: {self.position} | Action: {action} | Price: {current_price:.2f} | Reward: {reward:.4f} | State: {state}"
                )
                self.prev_price = current_price

                # --- HALLUCINATION: Simulate extra Q-learning updates off-market ---
                sim_df = df.copy()
                sim_position = self.position
                sim_prev_price = self.prev_price
                for _ in range(HALLUCINATION_STEPS):
                    sim_df, sim_price = hallucinate_step(sim_df, sim_position)
                    sim_indicators = sim_df.iloc[-1][['SMA', 'EMA', 'RSI']].values
                    sim_state = self.agent.get_state(sim_indicators)
                    sim_action_idx = self.agent.choose_action(sim_state)
                    sim_action = ACTIONS[sim_action_idx]
                    sim_reward = reward_function(sim_position, sim_price, sim_prev_price)
                    # Simulate position changes in hallucination
                    if sim_action == 'buy' and sim_position == 0:
                        sim_position = 1
                    elif sim_action == 'sell' and sim_position == 1:
                        sim_position = 0
                    sim_next_state = self.agent.get_state(sim_indicators)
                    self.agent.update(sim_state, sim_action_idx, sim_reward, sim_next_state)
                    logger.debug(
                        f"Hallucination | Position: {sim_position} | Action: {sim_action} | Price: {sim_price:.2f} | Reward: {sim_reward:.4f} | State: {sim_state}"
                    )
                    sim_prev_price = sim_price

                time.sleep(60)  # Wait for 1 minute before next real tick

            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                time.sleep(60)

if __name__ == "__main__":
    trader = Trader()
    trader.run()
