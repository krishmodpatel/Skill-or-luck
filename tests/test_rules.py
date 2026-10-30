import numpy as np
import pandas as pd
import pytest

from skill_or_luck.rules import RULES, breakout, hold_until_exit, ma_crossover, mean_reversion, signal


@pytest.mark.parametrize("rule", list(RULES))
def test_no_look_ahead(rule, random_walk):
    """Changing future prices must not change any earlier position."""
    params = RULES[rule]["default"]
    full = signal(rule, random_walk, params)
    cut = 1000
    altered = random_walk.copy()
    altered.iloc[cut:] *= np.linspace(0.5, 2.0, len(altered) - cut)
    pd.testing.assert_series_equal(full.iloc[:cut], signal(rule, altered, params).iloc[:cut])


@pytest.mark.parametrize("rule", list(RULES))
def test_positions_are_valid(rule, random_walk):
    positions = signal(rule, random_walk, RULES[rule]["default"])
    assert set(np.unique(positions)) <= {-1.0, 0.0, 1.0}
    assert positions.index.equals(random_walk.index)


def test_grid_sizes_are_about_200():
    for spec in RULES.values():
        assert 190 <= len(spec["grid"]) <= 230


def test_z_score_bound_used_to_prune_the_grid(random_walk):
    """Samuelson's inequality: a z-score over n points is at most (n - 1) / sqrt(n)."""
    for n in (5, 10, 20):
        z = (random_walk - random_walk.rolling(n).mean()) / random_walk.rolling(n).std()
        assert z.abs().max() <= (n - 1) / np.sqrt(n) + 1e-9
    for params in RULES["mean_reversion"]["grid"]:
        n = params["lookback"]
        assert params["threshold"] < (n - 1) / np.sqrt(n)


def test_ma_crossover_follows_trend():
    up = pd.Series(np.arange(1.0, 101.0), index=pd.bdate_range("2020-01-01", periods=100))
    positions = ma_crossover(up, 5, 20)
    assert (positions.iloc[:19] == 0).all()  # slow average not yet defined
    assert (positions.iloc[19:] == 1).all()
    assert (ma_crossover(up[::-1].reset_index(drop=True), 5, 20).iloc[19:] == -1).all()


def test_breakout_enters_on_new_high_and_exits_on_new_low():
    prices = pd.Series([10, 10, 10, 11, 12, 12, 11.5, 9, 9, 9], dtype=float)
    positions = breakout(prices, entry=3, exit=2)
    # 11 > max(10,10,10) -> long; stays long until a close below the prior 2-day low
    assert positions.tolist() == [0, 0, 0, 1, 1, 1, 0, -1, -1, -1]


def test_hold_until_exit_flips_on_same_bar():
    entry_long = [True, False, False]
    exit_long = [False, False, True]
    entry_short = [False, False, True]
    exit_short = [False, False, False]
    assert hold_until_exit(entry_long, exit_long, entry_short, exit_short).tolist() == [1, 1, -1]


def test_mean_reversion_waits_for_mean():
    prices = pd.Series([10.0] * 5 + [7.0, 8.0, 9.0, 10.5, 10.5])
    positions = mean_reversion(prices, lookback=5, threshold=1.5)
    assert positions.iloc[5] == 1          # far below the mean: buy
    assert positions.iloc[6] == 1          # still below the mean: hold
    assert positions.iloc[7] == 0          # back above the mean: exit


def test_invalid_parameters_raise(random_walk):
    with pytest.raises(ValueError):
        ma_crossover(random_walk, 50, 20)
    with pytest.raises(ValueError):
        mean_reversion(random_walk, 1, 2.0)
