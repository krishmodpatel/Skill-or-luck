"""Write results/REPORT.md from the experiment tables. Numbers only; no interpretation."""

from __future__ import annotations

import json
import math
import platform
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from scipy.stats import binom

from . import stats
from .experiments import COST_LEVELS_BPS, HOLDOUT_START, START, TRAIN_YEARS
from .plots import NAMES


def pct(x, digits=1):
    return "–" if pd.isna(x) else f"{100 * x:.{digits}f}%"


def num(x, digits=2):
    return "–" if pd.isna(x) else f"{x:.{digits}f}"


def interval(low, high, fmt):
    return "–" if pd.isna(low) else f"[{fmt(low)}, {fmt(high)}]"


def md_table(rows: list[dict]) -> str:
    header = list(rows[0])
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(r[h]) for h in header) + " |" for r in rows]
    return "\n".join(lines)


def own_record(wins: int, trades: int, payoff: float) -> str:
    """How much a small personal trading record can say about the true win rate."""
    breakeven = 1 / (1 + payoff)  # p * payoff = (1 - p) * 1
    low, high = stats.wilson_interval(wins, trades)
    cp_low, cp_high = stats.clopper_pearson(wins, trades)
    p_value = binom.sf(wins - 1, trades, breakeven)  # P(X >= wins) at breakeven
    rate = wins / trades
    needed = stats.trades_needed(breakeven, rate) if rate > breakeven else math.inf
    return "\n".join([
        f"- Record: {wins} wins in {trades} trades (win rate {pct(rate)}), average win "
        f"{payoff:.2f} times the average loss.",
        f"- 95% interval for the true win rate: {pct(low, 0)} to {pct(high, 0)} (Wilson); "
        f"{pct(cp_low, 0)} to {pct(cp_high, 0)} (Clopper–Pearson, exact).",
        f"- Break-even win rate at that payoff ratio: {pct(breakeven)}.",
        f"- Chance of {wins} or more wins in {trades} trades if the true win rate were exactly "
        f"break-even: {pct(p_value)} (one-sided p-value).",
        f"- Trades needed to show the win rate is above break-even, if the true rate were "
        f"{pct(rate)} (5% one-sided test, 80% power): about {needed}.",
        "- Caveat: the payoff ratio is itself estimated from the same few trades, so the "
        "real uncertainty is larger than shown.",
    ])


def power_section() -> str:
    win_rows = [{"Edge to detect": f"{pct(p1, 0)} win rate vs 50%", "Test": test,
                 "Trades needed": f"{stats.trades_needed(0.5, p1, one_sided=one):,}"}
                for p1 in (0.52, 0.55) for test, one in (("one-sided", True), ("two-sided", False))]
    sharpe_rows = [{"True annual Sharpe": s, "Years of daily data needed": num(stats.years_needed(s), 1)}
                   for s in (0.25, 0.5, 1.0, 2.0)]
    return "\n\n".join([
        "## Power analysis (no data needed)",
        "Significance level 5%, power 80%, normal approximation; equal-sized wins and losses "
        "for the win-rate rows.",
        md_table(win_rows), md_table(sharpe_rows),
    ])


def asset_section(r: dict, figure_dir: str) -> str:
    s = r["summary"]
    main_first, main_last, main_n = r["main_dates"]
    hold_first, hold_last, hold_n = r["holdout_dates"]

    performance_rows, trade_rows = [], []
    for _, row in s.iterrows():
        performance_rows.append({
            "Rule": NAMES[row["rule"]], "Parameters": row["params"] or "–",
            "CAGR": pct(row["cagr"]), "Sharpe": num(row["sharpe"]),
            "Sharpe 95% CI": interval(row.get("sharpe_low"), row.get("sharpe_high"), num),
            "Max drawdown": pct(row["max_drawdown"]), "Time in market": pct(row["exposure"], 0),
        })
        if row["rule"] == "buy_and_hold":
            continue
        needed = row["trades_needed"]
        trade_rows.append({
            "Rule": NAMES[row["rule"]], "Trades": int(row["trades"]),
            "Win rate": pct(row["win_rate"]),
            "Win rate 95% CI": interval(row["win_rate_low"], row["win_rate_high"], pct),
            "Avg trade": pct(row["avg_trade"], 2),
            "Avg trade 95% CI": interval(row["avg_trade_low"], row["avg_trade_high"], lambda x: pct(x, 2)),
            "Payoff ratio": num(row["payoff_ratio"]),
            "Random-entry p": num(row["p_random_entry"], 3),
            "Trades needed": f"{needed:,.0f}" if math.isfinite(needed) else "– (avg trade ≤ 0)",
        })

    wf = r["walk_forward_summary"]
    wf_rows = [{
        "Rule": NAMES[w["rule"]], "Folds": int(w["folds"]),
        "Mean Sharpe, training years": num(w["mean_in_sample_sharpe"]),
        "Mean Sharpe, next year": num(w["mean_out_of_sample_sharpe"]),
        "Median setting, next year": num(w["mean_median_setting_oos_sharpe"]),
        "Folds where next year was worse": pct(w["share_of_folds_oos_below_is"], 0),
        "Chosen setting's percentile next year": pct(w["mean_chosen_oos_percentile"], 0),
        "Rank correlation (train vs next year)": num(w["mean_rank_correlation"]),
        "Stitched out-of-sample Sharpe": num(w["stitched_oos_sharpe"]),
    } for _, w in wf.iterrows()]

    dsr_rows = [{
        "Search": "all three rules" if d["scope"] == "all_rules" else NAMES[d["scope"]],
        "Settings tried": int(d["trials"]),
        "Best setting": f"{NAMES[d['best_rule']]} ({d['best_params']})",
        "Best Sharpe": num(d["best_sharpe"]),
        "Prob. Sharpe > 0 (PSR)": pct(d["psr_vs_zero"]),
        "Sharpe expected from luck alone": num(d["luck_hurdle_sharpe"]),
        "Deflated Sharpe (prob. above luck)": pct(d["deflated_sharpe"]),
    } for _, d in r["deflated_sharpe"].iterrows()]

    hold_rows = [{
        "Rule": NAMES[h["rule"]], "Parameters": h["params"] or "–", "Chosen by": h["choice"],
        "Return": pct(h["total_return"], 2), "Sharpe": num(h["sharpe"]),
        "Sharpe 95% CI": interval(h.get("sharpe_low"), h.get("sharpe_high"), num),
        "Trades": int(h["trades"]),
    } for _, h in r["holdout"].iterrows()]

    costs = r["cost_sensitivity"].pivot(index="rule", columns="cost_bps", values="sharpe")
    cost_rows = [{"Rule": NAMES[rule], **{f"{bps} bp": num(costs.loc[rule, bps]) for bps in COST_LEVELS_BPS}}
                 for rule in costs.index]
    delay = r["execution_delay"].pivot(index="rule", columns="extra_delay_days", values="sharpe")
    delay_rows = [{"Rule": NAMES[rule], "Fill at the decision price": num(delay.loc[rule, 0]),
                   "Fill one day later": num(delay.loc[rule, 1])} for rule in delay.index]

    f = figure_dir
    return "\n\n".join([
        f"## {r['label']}",
        f"Main sample: {main_first} to {main_last} ({main_n:,} daily prices). "
        f"Holdout: {hold_first} to {hold_last} ({hold_n} prices). "
        f"Assumed cost: {r['cost_bps']:g} bp per trade side, including slippage. "
        f"Sharpe ratios are annualised with a zero risk-free rate.",
        "### Default rules on the main sample",
        md_table(performance_rows),
        md_table(trade_rows),
        "Random-entry p: share of random-timing versions (same trades, same holding periods, "
        "same time in market) that did at least as well, gross of costs. Trades needed: trades "
        "required for the observed average trade to be significant (5% one-sided, 80% power).",
        f"![Equity curves]({f}/equity.png)",
        f"![Random-entry test]({f}/random_entry.png)",
        "### Cost sensitivity (Sharpe after costs, default rules)",
        md_table(cost_rows),
        f"![Cost sensitivity]({f}/cost_sensitivity.png)",
        "### Execution delay (Sharpe after costs, default rules)",
        md_table(delay_rows),
        f"### Overfitting: walk-forward test ({TRAIN_YEARS}-year training window)",
        md_table(wf_rows),
        f"![Walk-forward]({f}/walk_forward.png)",
        "### Deflated Sharpe ratio (all settings, main sample)",
        md_table(dsr_rows),
        "### Six-month holdout",
        md_table(hold_rows),
        f"![Holdout]({f}/holdout.png)",
    ])


def write_report(results: list[dict], path: Path) -> None:
    manifest_path = path.parent / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    packages = ", ".join(f"{p} {version(p)}" for p in ("numpy", "pandas", "scipy", "matplotlib"))
    data_lines = [f"- **{r['label']}**: {manifest.get(r['asset'], {}).get('source', 'unknown')}, "
                  f"retrieved {manifest.get(r['asset'], {}).get('retrieved_utc', '?')} UTC"
                  for r in results]
    text = "\n\n".join([
        "# Results",
        "Generated by `python -m skill_or_luck run`. All figures are backtests on historical "
        "data, not live trading results.",
        f"Design: default parameters were fixed before any backtest. Main sample starts "
        f"{START.date()}; data from {HOLDOUT_START.date()} is a holdout that is computed in "
        "every run but never used to choose anything.",
        "\n".join(data_lines),
        f"Python {platform.python_version()}; {packages}.",
        *[asset_section(r, r["asset"]) for r in results],
        power_section(),
    ])
    path.write_text(text + "\n", encoding="utf-8")