"""Trading rules.

Every rule maps a close-price series to a target position for each day: +1 long,
-1 short, 0 flat. The position for day t uses closes up to and including day t only;
the backtest applies it to the return of day t+1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ma_crossover(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """Long when the fast moving average is above the slow one, short when below."""
    if not 0 < fast < slow:
        raise ValueError("need 0 < fast < slow")
    spread = close.rolling(fast).mean() - close.rolling(slow).mean()
    return np.sign(spread).fillna(0.0)


def hold_until_exit(long_entry, long_exit, short_entry, short_exit) -> np.ndarray:
    """Turn entry/exit conditions into positions that persist until an exit fires.

    The state has to be tracked bar by bar because an exit only applies to the
    position it belongs to. An exit and a new entry can happen on the same bar,
    which turns a long into a short (or the reverse).
    """
    long_entry, long_exit, short_entry, short_exit = (
        np.asarray(x, dtype=bool) for x in (long_entry, long_exit, short_entry, short_exit)
    )
    positions = np.zeros(len(long_entry))
    state = 0
    for i in range(len(positions)):
        if (state == 1 and long_exit[i]) or (state == -1 and short_exit[i]):
            state = 0
        if state == 0:
            if long_entry[i]:
                state = 1
            elif short_entry[i]:
                state = -1
        positions[i] = state
    return positions


def breakout(close: pd.Series, entry: int, exit: int) -> pd.Series:
    """Price-action breakout (Donchian channel, as in the Turtle rules).

    Go long when the close breaks above the highest close of the previous `entry`
    days; exit when it falls below the lowest close of the previous `exit` days.
    Shorts mirror this. Channels use closes because the FX series has no highs/lows.
    """
    if entry < 1 or exit < 1:
        raise ValueError("lookbacks must be positive")
    previous = close.shift(1)
    return pd.Series(
        hold_until_exit(
            long_entry=close > previous.rolling(entry).max(),
            long_exit=close < previous.rolling(exit).min(),
            short_entry=close < previous.rolling(entry).min(),
            short_exit=close > previous.rolling(exit).max(),
        ),
        index=close.index,
    )


def mean_reversion(close: pd.Series, lookback: int, threshold: float) -> pd.Series:
    """Fade large deviations from the moving average.

    z = (close - mean) / std over `lookback` days. Go long when z < -threshold and
    hold until z is back to 0; shorts mirror this.
    """
    if lookback < 2 or threshold <= 0:
        raise ValueError("need lookback >= 2 and threshold > 0")
    z = (close - close.rolling(lookback).mean()) / close.rolling(lookback).std()
    return pd.Series(
        hold_until_exit(
            long_entry=z < -threshold,
            long_exit=z >= 0,
            short_entry=z > threshold,
            short_exit=z <= 0,
        ),
        index=close.index,
    )


# Defaults are standard textbook values, fixed before any backtest was run:
# the 50/200-day "golden cross", Turtle System 2 (55-day entry, 20-day exit),
# and a 20-day, 2-standard-deviation band (as in Bollinger bands).
RULES = {
    "ma_crossover": {
        "function": ma_crossover,
        "default": {"fast": 50, "slow": 200},
        "grid": [
            {"fast": f, "slow": s}
            for f in range(5, 101, 5)
            for s in range(20, 261, 20)
            if f < s
        ],
    },
    "breakout": {
        "function": breakout,
        "default": {"entry": 55, "exit": 20},
        "grid": [
            {"entry": n, "exit": m}
            for n in range(10, 251, 10)
            for m in (5, 10, 20, 30, 40, 50, 60, 80, 100, 120)
            if m < n
        ],
    },
    "mean_reversion": {
        "function": mean_reversion,
        "default": {"lookback": 20, "threshold": 2.0},
        # A z-score over n points can never exceed (n - 1) / sqrt(n) (Samuelson's
        # inequality), so settings above that bound could never trade and are skipped.
        "grid": [
            {"lookback": n, "threshold": k}
            for n in range(5, 101, 5)
            for k in np.arange(0.5, 3.01, 0.25).round(2).tolist()
            if k < (n - 1) / np.sqrt(n)
        ],
    },
}


def signal(rule: str, close: pd.Series, params: dict) -> pd.Series:
    return RULES[rule]["function"](close, **params)
