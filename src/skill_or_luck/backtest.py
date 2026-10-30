"""Turn target positions into daily returns, trades and summary metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Backtest:
    asset_returns: pd.Series  # simple daily returns of the asset
    position: pd.Series       # position held during each day
    gross: pd.Series          # strategy returns before costs
    net: pd.Series            # strategy returns after costs
    trades: pd.DataFrame


def run_backtest(close: pd.Series, target: pd.Series, cost_bps: float) -> Backtest:
    """Backtest a target-position series over the period covered by `close`.

    The target decided with day t's price is held from that price to day t+1's, so a
    rule never earns a return it has already seen. This assumes the trade fills at the
    same price the decision used, which is optimistic. The strategy starts flat and any
    open position is closed at the last close. `cost_bps` is charged per unit of
    position traded (one side), so a long-to-short flip pays it twice.
    """
    target = target.reindex(close.index)
    if target.isna().any():
        raise ValueError("target positions are missing for some dates")
    asset_returns = close.pct_change().fillna(0.0)
    position = target.shift(1).fillna(0.0)

    turnover = position.diff().abs().fillna(0.0)  # day 0 is always flat
    turnover.iloc[-1] += abs(position.iloc[-1])  # liquidate at the end
    cost_rate = cost_bps / 1e4

    gross = position * asset_returns
    net = gross - turnover * cost_rate
    return Backtest(asset_returns, position, gross, net, extract_trades(position, gross, cost_rate))


def extract_trades(position: pd.Series, gross: pd.Series, cost_rate: float) -> pd.DataFrame:
    """One row per trade: a run of consecutive days with the same non-zero position."""
    held = (position != 0).to_numpy()
    starts = held & (position != position.shift()).to_numpy()
    days = pd.DataFrame({
        "trade": np.cumsum(starts)[held],
        "day": position.index[held],
        "direction": position.to_numpy()[held],
        "growth": 1 + gross.to_numpy()[held],
    })
    by_trade = days.groupby("trade")
    growth = by_trade["growth"].prod()
    return pd.DataFrame({
        "first_day": by_trade["day"].first(),
        "last_day": by_trade["day"].last(),
        "direction": by_trade["direction"].first(),
        "days": by_trade.size(),
        "gross_return": growth - 1,
        # every trade pays the cost once on entry and once on exit
        "net_return": growth * (1 - cost_rate) ** 2 - 1,
    }).reset_index(drop=True)


def periods_per_year(returns: pd.Series) -> float:
    years = (returns.index[-1] - returns.index[0]).days / 365.25
    return len(returns) / years


def sharpe(returns: pd.Series, ppy: float) -> float:
    """Annualised Sharpe ratio with a zero risk-free rate; 0 for a strategy that stayed flat."""
    std = returns.std(ddof=1)
    if std == 0:
        return 0.0
    return float(returns.mean() / std * np.sqrt(ppy))


def max_drawdown(returns: pd.Series) -> float:
    equity = np.concatenate([[1.0], (1 + returns).cumprod().to_numpy()])
    return float((equity / np.maximum.accumulate(equity) - 1).min())


def summarise(returns: pd.Series, position: pd.Series | None = None,
              trades: pd.DataFrame | None = None) -> dict:
    ppy = periods_per_year(returns)
    years = len(returns) / ppy
    result = {
        "cagr": float((1 + returns).prod() ** (1 / years) - 1),
        "volatility": float(returns.std(ddof=1) * np.sqrt(ppy)),
        "sharpe": sharpe(returns, ppy),
        "max_drawdown": max_drawdown(returns),
    }
    if position is not None:
        result["exposure"] = float((position != 0).mean())
    if trades is not None:
        net = trades["net_return"]
        wins, losses = net[net > 0], net[net <= 0]
        result.update({
            "trades": len(trades),
            "win_rate": float((net > 0).mean()) if len(net) else float("nan"),
            "avg_trade": float(net.mean()) if len(net) else float("nan"),
            "payoff_ratio": float(wins.mean() / -losses.mean())
            if len(wins) and len(losses) and losses.mean() < 0 else float("nan"),
        })
    return result
