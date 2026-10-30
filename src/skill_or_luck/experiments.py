"""Run every experiment for one asset and save the tables."""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import periods_per_year, run_backtest, sharpe, summarise
from .data import load_prices
from .rules import RULES, signal
from . import stats

ASSETS = {
    # Costs are one-way, per unit of position, and include slippage. They are
    # assumptions, not measured fills; cost_sensitivity.csv shows how much they matter.
    "eurusd": {"label": "EUR/USD", "cost_bps": 1.0},
    "nifty50": {"label": "Nifty 50", "cost_bps": 5.0},
}
START = pd.Timestamp("2009-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-29")
HOLDOUT_END = pd.Timestamp("2026-10-29")
TRAIN_YEARS = 3
COST_LEVELS_BPS = [0, 1, 2, 5, 10, 20]
SEED = 2026


def rng_for(*keys: str) -> np.random.Generator:
    """A separate, reproducible random stream per experiment, independent of run order."""
    return np.random.default_rng([SEED, zlib.crc32("/".join(keys).encode())])


def column_sharpes(returns: pd.DataFrame, ppy: float) -> pd.Series:
    """Sharpe ratio of each column; 0 for a setting that never traded in the window."""
    std = returns.std(ddof=1)
    return (returns.mean() / std.where(std > 0) * np.sqrt(ppy)).fillna(0.0)


def params_label(params: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in params.items())


def default_rule_tests(asset, prices, main, cost, n_perm, n_boot):
    """Fixed textbook parameters on the main sample: metrics, intervals, random-entry test."""
    rows, curves, nulls = [], {}, {}
    hold = run_backtest(main, pd.Series(1.0, index=main.index), cost)
    ppy = periods_per_year(hold.net)
    row = {"rule": "buy_and_hold", "params": "", **summarise(hold.net, hold.position, hold.trades)}
    row["sharpe_low"], row["sharpe_high"] = stats.block_bootstrap_sharpe(
        hold.net, ppy, n_boot, rng_for(asset, "buy_and_hold", "sharpe"))
    rows.append(row)
    curves["buy_and_hold"] = hold.net

    for rule, spec in RULES.items():
        bt = run_backtest(main, signal(rule, prices, spec["default"]), cost)
        row = {"rule": rule, "params": params_label(spec["default"]),
               **summarise(bt.net, bt.position, bt.trades)}
        row["sharpe_low"], row["sharpe_high"] = stats.block_bootstrap_sharpe(
            bt.net, ppy, n_boot, rng_for(asset, rule, "sharpe"))
        ci = stats.bootstrap_trades(bt.trades["net_return"], n_boot, rng_for(asset, rule, "trades"))
        row.update({"avg_trade_low": ci["mean_low"], "avg_trade_high": ci["mean_high"],
                    "win_rate_low": ci["win_low"], "win_rate_high": ci["win_high"]})
        test = stats.random_entry_test(bt.position, bt.asset_returns, n_perm, rng_for(asset, rule, "perm"))
        row["p_random_entry"] = test["p_value"]
        row["trades_needed"] = stats.trades_needed_for_mean(
            bt.trades["net_return"].mean(), bt.trades["net_return"].std(ddof=1))
        rows.append(row)
        curves[rule] = bt.net
        nulls[rule] = test
    return pd.DataFrame(rows), pd.DataFrame(curves), nulls, ppy


def cost_sensitivity(prices, main, ppy):
    rows = []
    for rule, spec in RULES.items():
        target = signal(rule, prices, spec["default"])
        for bps in COST_LEVELS_BPS:
            rows.append({"rule": rule, "cost_bps": bps,
                         "sharpe": sharpe(run_backtest(main, target, bps).net, ppy)})
    return pd.DataFrame(rows)


def execution_delay(prices, main, cost, ppy):
    """Sharpe ratio if every trade filled one day later than the base case assumes."""
    rows = []
    for rule, spec in RULES.items():
        target = signal(rule, prices, spec["default"])
        for delay in (0, 1):
            delayed = target.shift(delay).fillna(0.0)
            rows.append({"rule": rule, "extra_delay_days": delay,
                         "sharpe": sharpe(run_backtest(main, delayed, cost).net, ppy)})
    return pd.DataFrame(rows)


def grid_returns(prices, main, cost) -> dict[str, pd.DataFrame]:
    """Net daily returns on the main sample for every setting in every rule's grid."""
    return {
        rule: pd.DataFrame({i: run_backtest(main, signal(rule, prices, p), cost).net
                            for i, p in enumerate(spec["grid"])})
        for rule, spec in RULES.items()
    }


def walk_forward(grid: dict[str, pd.DataFrame], ppy: float):
    """Each year, pick the setting with the best Sharpe over the previous TRAIN_YEARS
    years and record how it does in the following year, which it has not seen.
    The stitched series does not charge for switching settings at year ends."""
    rows, stitched = [], {}
    for rule, returns in grid.items():
        pieces = []
        for year in range(START.year + TRAIN_YEARS, HOLDOUT_START.year):
            train = returns.loc[f"{year - TRAIN_YEARS}-01-01":f"{year - 1}-12-31"]
            test = returns.loc[f"{year}-01-01":f"{year}-12-31"]
            in_sample, out_sample = column_sharpes(train, ppy), column_sharpes(test, ppy)
            best = in_sample.idxmax()
            rows.append({
                "rule": rule, "test_year": year,
                "chosen": params_label(RULES[rule]["grid"][best]),
                "in_sample_sharpe": in_sample[best],
                "out_of_sample_sharpe": out_sample[best],
                "median_out_of_sample_sharpe": out_sample.median(),
                # share of settings that did worse than the chosen one out of sample
                "chosen_oos_percentile": (out_sample < out_sample[best]).mean(),
                "rank_correlation": in_sample.corr(out_sample, method="spearman"),
            })
            pieces.append(test[best])
        stitched[rule] = pd.concat(pieces)
    return pd.DataFrame(rows), pd.DataFrame(stitched)


def walk_forward_summary(walk: pd.DataFrame, stitched: pd.DataFrame, ppy: float) -> pd.DataFrame:
    by_rule = walk.groupby("rule", sort=False)
    summary = pd.DataFrame({
        "folds": by_rule.size(),
        "mean_in_sample_sharpe": by_rule["in_sample_sharpe"].mean(),
        "mean_out_of_sample_sharpe": by_rule["out_of_sample_sharpe"].mean(),
        "mean_median_setting_oos_sharpe": by_rule["median_out_of_sample_sharpe"].mean(),
        "share_of_folds_oos_below_is": (walk["out_of_sample_sharpe"] < walk["in_sample_sharpe"])
        .groupby(walk["rule"], sort=False).mean(),
        "mean_chosen_oos_percentile": by_rule["chosen_oos_percentile"].mean(),
        "mean_rank_correlation": by_rule["rank_correlation"].mean(),
    })
    summary["stitched_oos_sharpe"] = column_sharpes(stitched, ppy)
    return summary.reset_index()


def deflated_sharpe_table(grid: dict[str, pd.DataFrame], ppy: float) -> pd.DataFrame:
    """Is the best backtest still convincing once we account for how many we tried?"""
    scopes = {rule: [rule] for rule in grid} | {"all_rules": list(grid)}
    rows = []
    for scope, rules in scopes.items():
        trials = pd.concat({r: grid[r] for r in rules}, axis=1)
        daily_sharpes = column_sharpes(trials, 1.0)
        rule, i = daily_sharpes.idxmax()
        dsr, hurdle = stats.deflated_sharpe(trials[(rule, i)], daily_sharpes)
        rows.append({
            "scope": scope, "trials": trials.shape[1],
            "best_rule": rule, "best_params": params_label(RULES[rule]["grid"][i]),
            "best_sharpe": daily_sharpes.max() * np.sqrt(ppy),
            "psr_vs_zero": stats.probabilistic_sharpe(trials[(rule, i)]),
            "luck_hurdle_sharpe": hurdle * np.sqrt(ppy),
            "deflated_sharpe": dsr,
        })
    return pd.DataFrame(rows)


def holdout_test(asset, prices, cost, grid, ppy, n_boot):
    """The six-month window. It is computed in every run but never used to choose
    anything: the tuned settings are picked from main-sample returns only. Because
    each window starts flat, the return of the holdout's first day is in neither sample."""
    window = prices.loc[HOLDOUT_START:HOLDOUT_END]
    train_start = HOLDOUT_START - pd.DateOffset(years=TRAIN_YEARS)
    rows, curves = [], {}
    hold = run_backtest(window, pd.Series(1.0, index=window.index), cost)
    rows.append({"rule": "buy_and_hold", "params": "", "choice": "-",
                 "total_return": (1 + hold.net).prod() - 1, "sharpe": sharpe(hold.net, ppy),
                 "trades": 1})
    curves["buy_and_hold"] = hold.net
    for rule, spec in RULES.items():
        train = grid[rule].loc[train_start:]
        chosen = spec["grid"][column_sharpes(train, ppy).idxmax()]
        for choice, params in (("default", spec["default"]), ("tuned on previous 3 years", chosen)):
            bt = run_backtest(window, signal(rule, prices, params), cost)
            low, high = stats.block_bootstrap_sharpe(bt.net, ppy, n_boot, rng_for(asset, rule, choice))
            rows.append({"rule": rule, "params": params_label(params), "choice": choice,
                         "total_return": (1 + bt.net).prod() - 1, "sharpe": sharpe(bt.net, ppy),
                         "sharpe_low": low, "sharpe_high": high, "trades": len(bt.trades)})
            if choice == "default":
                curves[rule] = bt.net
    return pd.DataFrame(rows), pd.DataFrame(curves), window


def analyse_asset(asset: str, n_perm: int, n_boot: int, out_dir: Path) -> dict:
    prices = load_prices(asset)
    cost = ASSETS[asset]["cost_bps"]
    main = prices.loc[START:HOLDOUT_START - pd.Timedelta(days=1)]
    if main.index[0] - START > pd.Timedelta(days=10):
        raise ValueError(f"{asset} data starts {main.index[0].date()}, after {START.date()}")

    summary, curves, nulls, ppy = default_rule_tests(asset, prices, main, cost, n_perm, n_boot)
    costs = cost_sensitivity(prices, main, ppy)
    delay = execution_delay(prices, main, cost, ppy)
    grid = grid_returns(prices, main, cost)
    walk, stitched = walk_forward(grid, ppy)
    deflated = deflated_sharpe_table(grid, ppy)
    holdout, holdout_curves, window = holdout_test(asset, prices, cost, grid, ppy, n_boot)

    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {"summary": summary, "cost_sensitivity": costs, "execution_delay": delay,
              "walk_forward": walk,
              "walk_forward_summary": walk_forward_summary(walk, stitched, ppy),
              "deflated_sharpe": deflated, "holdout": holdout}
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False, float_format="%.6g")

    return {"asset": asset, "label": ASSETS[asset]["label"], "cost_bps": cost, "ppy": ppy,
            "main_dates": (main.index[0].date(), main.index[-1].date(), len(main)),
            "holdout_dates": (window.index[0].date(), window.index[-1].date(), len(window)),
            "curves": curves, "nulls": nulls, "stitched": stitched,
            "holdout_curves": holdout_curves, **tables}
