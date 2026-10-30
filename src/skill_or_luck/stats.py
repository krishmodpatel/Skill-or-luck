"""Statistical tools for separating skill from luck."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

EULER_MASCHERONI = 0.5772156649015329


# --- Bootstrap -------------------------------------------------------------------

def bootstrap_trades(net_returns, n_boot: int, rng: np.random.Generator, alpha: float = 0.05) -> dict:
    """Resample trades with replacement; percentile intervals for mean return and win rate.

    Treats trades as independent draws, which is an approximation: consecutive trades
    can share a market regime.
    """
    x = np.asarray(net_returns, dtype=float)
    if len(x) < 2:
        return {"mean_low": np.nan, "mean_high": np.nan, "win_low": np.nan, "win_high": np.nan}
    samples = x[rng.integers(0, len(x), size=(n_boot, len(x)))]
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    mean_low, mean_high = np.percentile(samples.mean(axis=1), q)
    win_low, win_high = np.percentile((samples > 0).mean(axis=1), q)
    return {"mean_low": mean_low, "mean_high": mean_high, "win_low": win_low, "win_high": win_high}


def block_bootstrap_sharpe(returns, ppy: float, n_boot: int, rng: np.random.Generator,
                           block: int = 20, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile interval for the annualised Sharpe ratio from a circular block bootstrap.

    Resampling blocks of `block` consecutive days instead of single days keeps
    short-range dependence such as volatility clustering. 20 days (about a trading
    month) is close to the common n^(1/3) rule of thumb (16 for 17 years of data).
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    n_blocks = math.ceil(n / block)
    sharpes = np.full(n_boot, np.nan)
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        sample = r[(starts[:, None] + np.arange(block)).ravel()[:n] % n]
        sd = sample.std(ddof=1)
        if sd > 0:
            sharpes[i] = sample.mean() / sd * math.sqrt(ppy)
    if np.isnan(sharpes).all():  # the strategy never traded
        return float("nan"), float("nan")
    low, high = np.nanpercentile(sharpes, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


# --- Random-entry permutation test -------------------------------------------------

def runs(position: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Split a position series into trades: their directions, lengths and flat days."""
    position = np.asarray(position, dtype=float)
    change = np.flatnonzero(np.diff(position)) + 1
    bounds = np.concatenate([[0], change, [len(position)]])
    values = position[bounds[:-1]]
    lengths = np.diff(bounds)
    held = values != 0
    return values[held], lengths[held], int(lengths[~held].sum())


def random_entry_positions(directions, lengths, flat_days: int, rng: np.random.Generator) -> np.ndarray:
    """Rearrange a rule's trades in time without changing the trades themselves.

    The sequence of long and short trades is kept, holding periods are shuffled among
    trades of the same direction, and the flat days are split into random gaps.
    Neighbouring trades in the same direction keep at least one flat day between them
    so they cannot merge; the number of trades, days long, days short and trading
    costs therefore stay exactly the same.
    """
    directions = np.asarray(directions, dtype=float)
    lengths = np.asarray(lengths, dtype=int).copy()
    for side in (-1.0, 1.0):
        idx = np.flatnonzero(directions == side)
        lengths[idx] = lengths[rng.permutation(idx)]

    k = len(lengths)
    gaps = np.zeros(k + 1, dtype=int)
    gaps[1:-1] = directions[1:] == directions[:-1]  # one flat day keeps them separate
    free = flat_days - gaps.sum()  # never negative: the rule's own trades had these gaps
    # stars and bars: a uniformly random split of the free flat days into k + 1 gaps
    bars = np.sort(rng.choice(free + k, size=k, replace=False))
    gaps += np.diff(np.concatenate([[-1], bars, [free + k]])) - 1

    values = np.zeros(2 * k + 1)
    sizes = np.zeros(2 * k + 1, dtype=int)
    values[1::2] = directions
    sizes[1::2] = lengths
    sizes[0::2] = gaps
    return np.repeat(values, sizes)


def random_entry_test(position, asset_returns, n_perm: int, rng: np.random.Generator) -> dict:
    """Is the rule's timing better than random timing?

    The null strategies are built from the rule's own trades by
    `random_entry_positions`, so they hold the same trades for the same number of
    days and pay the same costs; only the timing differs. Costs therefore cancel and
    the comparison uses gross returns. p = (1 + #null >= observed) / (1 + n_perm),
    one-sided.
    """
    position = np.asarray(position, dtype=float)
    r = np.asarray(asset_returns, dtype=float)
    directions, lengths, flat_days = runs(position)
    observed = float(np.mean(position * r))
    if len(lengths) == 0:
        return {"observed": observed, "null": np.array([]), "p_value": np.nan}
    null = np.array([
        np.mean(random_entry_positions(directions, lengths, flat_days, rng) * r)
        for _ in range(n_perm)
    ])
    p_value = (1 + np.sum(null >= observed)) / (1 + n_perm)
    return {"observed": observed, "null": null, "p_value": float(p_value)}


# --- Proportions and power ---------------------------------------------------------

def wilson_interval(wins: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    z = stats.norm.ppf(0.5 + confidence / 2)
    p = wins / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def clopper_pearson(wins: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    alpha = 1 - confidence
    low = stats.beta.ppf(alpha / 2, wins, n - wins + 1) if wins > 0 else 0.0
    high = stats.beta.ppf(1 - alpha / 2, wins + 1, n - wins) if wins < n else 1.0
    return float(low), float(high)


def trades_needed(p0: float, p1: float, alpha: float = 0.05, power: float = 0.8,
                  one_sided: bool = True) -> int:
    """Trades needed to tell a true win rate p1 from p0 (normal approximation)."""
    z_alpha = stats.norm.ppf(1 - alpha if one_sided else 1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n = ((z_alpha * math.sqrt(p0 * (1 - p0)) + z_beta * math.sqrt(p1 * (1 - p1))) / (p1 - p0)) ** 2
    return math.ceil(n)


def years_needed(annual_sharpe: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Years of daily data needed to show a Sharpe ratio is above zero (one-sided).

    The estimated annualised Sharpe ratio has a standard error of about 1/sqrt(years)
    for roughly normal, independent returns (Lo, 2002).
    """
    z = stats.norm.ppf(1 - alpha) + stats.norm.ppf(power)
    return (z / annual_sharpe) ** 2


def trades_needed_for_mean(mean: float, sd: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Trades needed for a one-sided t-style test of mean trade return > 0."""
    if mean <= 0 or sd <= 0:
        return float("inf")
    z = stats.norm.ppf(1 - alpha) + stats.norm.ppf(power)
    return (z * sd / mean) ** 2


# --- Multiple testing ----------------------------------------------------------------

def probabilistic_sharpe(returns, benchmark_sr: float = 0.0) -> float:
    """Probability that the true per-period Sharpe ratio exceeds `benchmark_sr`.

    Bailey & Lopez de Prado (2012); adjusts the standard error for skewness and
    fat tails. Both Sharpe ratios are per period (daily), not annualised.
    """
    r = np.asarray(returns, dtype=float)
    sr = r.mean() / r.std(ddof=1)
    skew = stats.skew(r)
    kurt = stats.kurtosis(r, fisher=False)  # equals 3 for a normal distribution
    denom = math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    return float(stats.norm.cdf((sr - benchmark_sr) * math.sqrt(len(r) - 1) / denom))


def expected_max_sharpe(trial_sharpes) -> float:
    """Expected best per-period Sharpe among N trials if every true Sharpe were zero.

    Bailey & Lopez de Prado (2014), used as the hurdle in the deflated Sharpe ratio.
    """
    trial_sharpes = np.asarray(trial_sharpes, dtype=float)
    n = len(trial_sharpes)
    if n < 2:
        return 0.0
    sd = trial_sharpes.std(ddof=1)
    return float(sd * ((1 - EULER_MASCHERONI) * stats.norm.ppf(1 - 1 / n)
                       + EULER_MASCHERONI * stats.norm.ppf(1 - 1 / (n * math.e))))


def deflated_sharpe(returns, trial_sharpes) -> tuple[float, float]:
    """Deflated Sharpe ratio: the PSR measured against the best-of-N-by-luck hurdle."""
    hurdle = expected_max_sharpe(trial_sharpes)
    return probabilistic_sharpe(returns, hurdle), hurdle
