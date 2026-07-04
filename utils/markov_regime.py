import pandas as pd
import numpy as np

def compute_markov_regime(df: pd.DataFrame, asset: str) -> dict:
    """
    Computes a 3-State Markov Transition Matrix (BULLISH, BEARISH, SIDEWAYS)
    over the passed DataFrame (using close-to-close returns).
    Returns the current regime and a calculated uncertainty penalty (0 to 0.15).
    """
    if df.empty or len(df) < 10:
        return {"regime": "UNKNOWN", "penalty": 0.0, "stability": 1.0}

    # Calculate returns
    returns = df['close'].pct_change().dropna()
    std_dev = returns.std()

    # Discretize into 3 states: 
    # Bearish: < -0.5 std, Sideways: -0.5 to 0.5 std, Bullish: > 0.5 std
    conditions = [
        (returns > 0.5 * std_dev),
        (returns < -0.5 * std_dev)
    ]
    choices = ['BULLISH', 'BEARISH']
    states = np.select(conditions, choices, default='SIDEWAYS')

    # Compute transition stability by counting how many times the state flips
    flips = sum(1 for i in range(1, len(states)) if states[i] != states[i-1])
    max_flips = len(states) - 1
    flip_rate = flips / max_flips if max_flips > 0 else 0

    # Stability is inverse of flip rate
    stability = max(0.0, 1.0 - (flip_rate * 2))  # Scale it so very high flip rate = 0 stability

    current_regime = states[-1]

    # Calculate uncertainty penalty based on stability. 
    # High flip rate -> low stability -> high uncertainty penalty. Max 0.15 (15%)
    penalty = 0.15 * (1.0 - stability)

    return {
        "regime": current_regime,
        "stability": round(stability, 2),
        "penalty": round(penalty, 4)
    }
