import numpy as np
import pandas as pd
from indicators import get_indicators
from config import HALLUCINATION_NOISE

def hallucinate_step(df, position):
    # Simulate a next close price with some noise
    last_price = df.iloc[-1]['close']
    pct_change = np.random.uniform(-HALLUCINATION_NOISE, HALLUCINATION_NOISE)
    new_price = last_price * (1 + pct_change)
    new_row = {'close': new_price}
    new_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    new_df = get_indicators(new_df)
    return new_df, new_price
