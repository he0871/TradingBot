import unittest
import pandas as pd
from algotrading.TradingEnv import TradingEnv

class TestTradingEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = TradingEnv(parquet_path='path/to/your/data.parquet')

    def test_reset(self):
        observation = self.env.reset()
        self.assertEqual(self.env.current_step, 0)
        self.assertEqual(self.env.cash, 10000)
        self.assertEqual(self.env.position, 0)
        self.assertEqual(self.env.entry_price, 0)
        self.assertEqual(observation.shape[0], self.env.observation_space.shape[0])

    def test_step_buy(self):
        self.env.reset()
        action = 1  # Buy
        observation, reward, done, _ = self.env.step(action)
        self.assertEqual(self.env.position, 1)
        self.assertGreater(self.env.entry_price, 0)

    def test_step_sell(self):
        self.env.reset()
        self.env.step(1)  # Buy
        action = 2  # Sell
        observation, reward, done, _ = self.env.step(action)
        self.assertEqual(self.env.position, 0)
        self.assertGreaterEqual(self.env.cash, 10000)

    def test_step_hold(self):
        self.env.reset()
        action = 0  # Hold
        observation, reward, done, _ = self.env.step(action)
        self.assertEqual(self.env.position, 0)
        self.assertEqual(self.env.cash, 10000)

    def test_done_condition(self):
        self.env.reset()
        for _ in range(self.env.n_steps):
            self.env.step(0)  # Hold
        self.assertTrue(self.env.current_step >= self.env.n_steps - 1)

if __name__ == '__main__':
    unittest.main()