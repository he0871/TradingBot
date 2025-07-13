import numpy as np
from collections import defaultdict
from config import ACTIONS, ALPHA, GAMMA, EPSILON

class QLearningAgent:
    def __init__(self):
        self.q_table = defaultdict(lambda: np.zeros(len(ACTIONS)))

    def get_state(self, indicators):
        # Discretize indicators for state representation (simple version)
        return tuple(np.round(indicators, 2))

    def choose_action(self, state):
        if np.random.rand() < EPSILON:
            return np.random.choice(len(ACTIONS))
        else:
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        best_next = np.max(self.q_table[next_state])
        self.q_table[state][action] += ALPHA * (reward + GAMMA * best_next - self.q_table[state][action])
