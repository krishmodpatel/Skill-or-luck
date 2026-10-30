"""How often does the random-entry test raise a false alarm?

Each rule is run on an independent random walk, so its positions carry no
information about EUR/USD, and then tested against real EUR/USD returns. A well
calibrated test should give p <= 0.05 in about 5% of runs.

Run from the repository folder: python scripts/calibrate_permutation_test.py
"""

import numpy as np
import pandas as pd

from skill_or_luck import stats
from skill_or_luck.backtest import run_backtest
from skill_or_luck.data import load_prices
from skill_or_luck.rules import RULES, signal

RUNS, PERMUTATIONS = 300, 199

prices = load_prices("eurusd").loc["2009-01-01":"2026-04-28"]
volatility = np.log(prices).diff().std()
rng = np.random.default_rng(123)

for rule, spec in RULES.items():
    p_values = []
    for _ in range(RUNS):
        unrelated = pd.Series(np.exp(np.cumsum(rng.normal(0, volatility, len(prices)))), index=prices.index)
        bt = run_backtest(prices, signal(rule, unrelated, spec["default"]), cost_bps=0)
        p_values.append(stats.random_entry_test(bt.position, bt.asset_returns, PERMUTATIONS, rng)["p_value"])
    p_values = np.array(p_values)
    print(f"{rule:15s} p <= 0.05 in {np.mean(p_values <= 0.05):.1%} of {RUNS} runs "
          f"(p <= 0.10: {np.mean(p_values <= 0.10):.1%})")
