import numpy as np
import pandas as pd
import pytest

from skill_or_luck.backtest import extract_trades, max_drawdown, run_backtest, summarise


def prices(values):
    return pd.Series(values, index=pd.bdate_range("2021-01-04", periods=len(values)), dtype=float)


def test_position_applies_to_next_day():
    close = prices([100, 110, 121, 121])
    target = pd.Series([1.0, 0.0, 0.0, 0.0], index=close.index)
    bt = run_backtest(close, target, cost_bps=0)
    # decided at day 0's close, so it earns day 1's +10% and nothing else
    assert bt.gross.tolist() == pytest.approx([0, 0.10, 0, 0])


def test_round_trip_pays_cost_twice():
    close = prices([100.0] * 5)
    target = pd.Series([1.0, 1.0, 0.0, 0.0, 0.0], index=close.index)
    bt = run_backtest(close, target, cost_bps=10)
    assert bt.net.sum() == pytest.approx(-0.002)
    assert bt.trades["net_return"].iloc[0] == pytest.approx((1 - 0.001) ** 2 - 1)


def test_flip_pays_cost_twice_and_makes_two_trades():
    close = prices([100.0, 101.0, 100.0, 99.0, 98.0])
    target = pd.Series([1.0, -1.0, -1.0, -1.0, -1.0], index=close.index)
    bt = run_backtest(close, target, cost_bps=5)
    assert len(bt.trades) == 2
    assert bt.trades["direction"].tolist() == [1, -1]
    flip_day_cost = bt.gross.iloc[2] - bt.net.iloc[2]
    assert flip_day_cost == pytest.approx(2 * 0.0005)


def test_open_position_is_closed_at_the_end():
    close = prices([100.0, 100.0, 100.0])
    bt = run_backtest(close, pd.Series(1.0, index=close.index), cost_bps=10)
    assert bt.net.sum() == pytest.approx(-0.002)  # entry and forced exit


def test_always_long_without_costs_equals_buy_and_hold(random_walk):
    bt = run_backtest(random_walk, pd.Series(1.0, index=random_walk.index), cost_bps=0)
    growth = (1 + bt.net).prod()
    # decided at the first close, so the strategy holds every return after it
    assert growth == pytest.approx(random_walk.iloc[-1] / random_walk.iloc[0])


def test_trade_returns_match_daily_returns(random_walk):
    target = pd.Series(np.sign(np.sin(np.arange(len(random_walk)) / 15)), index=random_walk.index)
    bt = run_backtest(random_walk, target, cost_bps=3)
    from_trades = np.log1p(bt.trades["net_return"]).sum()
    from_days = np.log1p(bt.net).sum()
    # small differences come from costs and returns landing on the same day
    assert from_trades == pytest.approx(from_days, abs=5e-4)


def test_missing_targets_are_rejected():
    close = prices([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        run_backtest(close, pd.Series([1.0, np.nan, 1.0], index=close.index), 0)


def test_no_trades():
    close = prices([1.0, 2.0, 3.0, 2.0])
    bt = run_backtest(close, pd.Series(0.0, index=close.index), cost_bps=5)
    assert bt.trades.empty
    metrics = summarise(bt.net, bt.position, bt.trades)
    assert metrics["trades"] == 0
    assert metrics["sharpe"] == 0.0 and np.isnan(metrics["win_rate"])


def test_max_drawdown_counts_a_loss_on_the_first_day():
    returns = pd.Series([-0.5, 1.0], index=pd.bdate_range("2021-01-04", periods=2))
    assert max_drawdown(returns) == pytest.approx(-0.5)


def test_extract_trades_directions_and_lengths():
    position = pd.Series([0, 1, 1, 0, -1, -1, -1, 1], index=pd.bdate_range("2021-01-04", periods=8),
                         dtype=float)
    trades = extract_trades(position, position * 0.0, 0.0)
    assert trades["direction"].tolist() == [1, -1, 1]
    assert trades["days"].tolist() == [2, 3, 1]
