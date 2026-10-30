"""Figures for the report. One function per figure; all take the dict from analyse_asset."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

NAMES = {"buy_and_hold": "Buy and hold", "ma_crossover": "MA crossover",
         "breakout": "Breakout", "mean_reversion": "Mean reversion"}
COLORS = {"buy_and_hold": "#8a8984", "ma_crossover": "#2a78d6",
          "breakout": "#eb6834", "mean_reversion": "#1baf7a"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
})


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _growth_chart(curves, title, path, log=True):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for name in curves:
        growth = (1 + curves[name]).cumprod()
        ax.plot(growth.index, growth, color=COLORS[name], label=NAMES[name],
                linewidth=1.5 if name == "buy_and_hold" else 2)
        ax.annotate(f"{growth.iloc[-1]:.2f}", (growth.index[-1], growth.iloc[-1]),
                    xytext=(4, 0), textcoords="offset points", va="center", color=MUTED, fontsize=9)
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.2g}"))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda y, _: f"{y:.2g}"))
    ax.axhline(1, color=MUTED, linewidth=0.8)
    ax.set_ylabel("Growth of 1 (after costs)" + (", log scale" if log else ""))
    ax.set_title(title, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=len(curves))
    _save(fig, path)


def equity(result, out_dir):
    first, last, _ = result["main_dates"]
    _growth_chart(result["curves"],
                  f"{result['label']}: default rules vs buy and hold, {first} to {last}",
                  out_dir / "equity.png")


def holdout(result, out_dir):
    first, last, n = result["holdout_dates"]
    _growth_chart(result["holdout_curves"],
                  f"{result['label']}: six-month holdout, {first} to {last} ({n} days)",
                  out_dir / "holdout.png", log=False)


def random_entry(result, out_dir):
    ppy = result["ppy"]
    n_perm = len(next(iter(result["nulls"].values()))["null"])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (rule, test) in zip(axes, result["nulls"].items()):
        null = test["null"] * ppy * 100
        ax.hist(null, bins=50, color=GRID, edgecolor=MUTED, linewidth=0.3)
        ax.axvline(test["observed"] * ppy * 100, color=COLORS[rule], linewidth=2, label="Rule")
        ax.set_title(f"{NAMES[rule]}  (p = {test['p_value']:.3f})", loc="left")
        ax.set_xlabel("Gross return, % per year")
    axes[0].set_ylabel("Random-entry strategies")
    axes[0].legend(loc="upper left")
    fig.suptitle(f"{result['label']}: each rule against {n_perm:,} random-entry versions of itself",
                 x=0.01, ha="left", fontsize=11)
    _save(fig, out_dir / "random_entry.png")


def walk_forward(result, out_dir):
    walk = result["walk_forward"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, (rule, folds) in zip(axes, walk.groupby("rule", sort=False)):
        ax.plot(folds["test_year"], folds["in_sample_sharpe"], color="#2a78d6", marker="o",
                markersize=4, label="Chosen setting, training years")
        ax.plot(folds["test_year"], folds["out_of_sample_sharpe"], color="#eb6834", marker="o",
                markersize=4, label="Same setting, next year")
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_title(NAMES[rule], loc="left")
        ax.set_xlabel("Test year")
    axes[0].set_ylabel("Sharpe ratio")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"{result['label']}: best of ~200 settings on 3 years, then tested on the next year",
                 x=0.01, ha="left", fontsize=11)
    _save(fig, out_dir / "walk_forward.png")


def cost_sensitivity(result, out_dir):
    costs = result["cost_sensitivity"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for rule, rows in costs.groupby("rule", sort=False):
        ax.plot(rows["cost_bps"], rows["sharpe"], color=COLORS[rule], marker="o",
                markersize=4, label=NAMES[rule])
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.axvline(result["cost_bps"], color=MUTED, linewidth=0.8, linestyle=":")
    ax.annotate("assumed cost", (result["cost_bps"], ax.get_ylim()[1]), xytext=(4, -12),
                textcoords="offset points", color=MUTED, fontsize=9)
    ax.set_xlabel("Cost per trade side, basis points")
    ax.set_ylabel("Sharpe ratio after costs")
    ax.set_title(f"{result['label']}: default rules under different cost assumptions", loc="left")
    ax.legend()
    _save(fig, out_dir / "cost_sensitivity.png")


def all_figures(result, out_dir: Path) -> None:
    for draw in (equity, holdout, random_entry, walk_forward, cost_sensitivity):
        draw(result, out_dir)
