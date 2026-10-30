# Skill or luck?

Can you tell a profitable trading backtest from luck? This project runs three simple
trading rules over daily prices, charges trading costs, and then applies five
statistical checks that try to answer that question honestly.

> **Status (29 October 2026).** The full analysis has been run on **EUR/USD**
> (results below and in [`results/REPORT.md`](results/REPORT.md)). The code supports
> the **Nifty 50** as well, but its data comes from Yahoo Finance, which could not be
> reached from the environment the project was built in, so there are no Nifty 50
> results yet. See [Reproducing](#reproducing) to produce them.

## The question

A backtest replays a trading rule over past prices. With enough rules and
parameters to choose from, some backtest will always look good, even if no rule has
any real predictive power. This project asks, for each rule:

1. Is its timing better than random timing with the same trades?
2. How uncertain are its results, given how few trades it made?
3. How many trades would be needed to prove an edge of the size it shows?
4. If we pick the best of ~200 parameter settings, does it stay good on later data?
5. Once we account for how many settings were tried, is the best one still convincing?

## Method

### Data

| Asset | Source | Notes |
|---|---|---|
| EUR/USD | Federal Reserve H.10 noon buying rates in New York ([FRED series DEXUSEU](https://fred.stlouisfed.org/series/DEXUSEU)) | One rate per US business day, taken at noon New York time (not a closing price). By default the code uses the [datahub.io copy](https://github.com/datasets/exchange-rates) of this series (PDDL licence), pinned to a fixed commit so results reproduce exactly. That copy stores EUR per USD to 4 decimals, so inverting it changes values by less than 0.01%, against a typical daily move of about 0.3%. `fetch eurusd --latest` downloads FRED directly instead; it rounds USD per EUR rather than EUR per USD, which can change results slightly. |
| Nifty 50 | Yahoo Finance `^NSEI`, via `yfinance` | Daily closes of the price index (dividends not included). Yahoo data is convenient but unofficial and sometimes revised; the official source is NSE Indices. Any `Date`/`Close` CSV can be used instead with `python -m skill_or_luck import-csv nifty50 file.csv`. |

Every download is sorted, de-duplicated and checked (positive prices, at least 500
rows). Gaps longer than 7 days, moves larger than 15%, and moves more than 8 times
the typical daily move that immediately reverse (the usual sign of a bad price) are
reported in `results/data_manifest.json`, together with the source, retrieval time
and a SHA-256 hash of the file. The EUR/USD file is committed in `data/raw/`; the
Nifty 50 file is not, because Yahoo's terms do not allow redistribution.

### Periods

- **Main sample:** 1 January 2009 to 28 April 2026 (17.3 years). Earlier data is used
  only to warm up the indicators.
- **Holdout:** 29 April 2026 to 29 October 2026, or the latest price available. It is
  computed in every run but never used to choose anything: defaults were fixed in
  advance and tuned settings are picked from main-sample data only. (Each window
  starts flat, so the holdout's first daily return belongs to neither sample.)

Six months on its own cannot answer the question: it contains about 125 daily
returns and only a handful of trades per rule (see the power analysis below), so it
serves as a final out-of-sample check rather than the main test.

### Rules

All three rules can be long (+1), short (−1) or flat (0).

| Rule | Logic | Default (fixed before any backtest) |
|---|---|---|
| Moving-average crossover | Long when the fast simple moving average is above the slow one, short when below | 50 / 200 days |
| Breakout (price action) | Long when the price exceeds the highest price of the previous *N* days; exit when it falls below the lowest price of the previous *M* days; shorts mirror this (Donchian channel, as in the Turtle rules) | N = 55, M = 20 |
| Mean reversion | z = (price − mean) / std over *L* days. Long when z < −k, held until z ≥ 0; shorts mirror this | L = 20, k = 2 |

The default parameters are common textbook values, chosen before any backtest was run
and not changed afterwards. They are the main test. The parameter grids (~200
settings per rule) are used only for the overfitting experiments. Mean-reversion
settings that can never trade are left out of the grid: a z-score over *n* points
can never exceed (n − 1)/√n (Samuelson's inequality), so a 5-day z-score never
reaches 2.

### Backtest

- **No look-ahead.** A position decided with day *t*'s price earns the return from day
  *t* to day *t + 1*: `strategy_return[t+1] = position[t] × asset_return[t+1]`. A unit
  test checks that changing future prices never changes past positions.
- **Execution.** The base case assumes each trade fills at the same price the
  decision used. That is optimistic, so the report also shows results with every
  trade filled one day later.
- **Costs.** Each unit of position traded pays a fixed cost covering commission,
  spread and slippage: 1 basis point per side for EUR/USD and 5 basis points for the
  Nifty 50. Going from long to short trades two units and pays twice. These are
  assumptions, not measured fills, so results are also shown for 0 to 20 basis points.
- **Metrics.** CAGR, annualised volatility, Sharpe ratio (mean / standard deviation
  × √(trading days per year), risk-free rate taken as zero), maximum drawdown, time
  in the market, and per trade: win rate, average net return and payoff ratio
  (average win / average loss).

### Statistical tests

**Bootstrap intervals.** Trades are resampled with replacement 5,000 times to get
95% intervals for the average trade and the win rate. For the Sharpe ratio, daily
returns are resampled in blocks of 20 days (a circular block bootstrap), so that
calm and volatile periods stay together. Twenty days is about one trading month and
close to the common n^(1/3) rule of thumb (16 here).

**Random-entry permutation test.** For each rule, 5,000 random strategies are built
from the rule's own trades. They keep the same sequence of long and short trades;
holding periods are shuffled among trades in the same direction, and the flat days
are split into random gaps, with at least one flat day between two trades in the
same direction so that they cannot merge. Every random strategy therefore has the
same number of trades, the same days long and short, and the same costs; only the
timing differs. The p-value is the share that did at least as well as the rule:
`p = (1 + #{random ≥ rule}) / (1 + 5,000)`.

To check that the test does not raise false alarms, `scripts/calibrate_permutation_test.py`
applies each rule to an unrelated random walk (so its positions contain no
information about EUR/USD) and tests the result against real EUR/USD returns, 300
times per rule. p ≤ 0.05 occurred in 6.7%, 6.0% and 5.0% of runs for the three
rules, close to the 5% expected of a well-calibrated test (the standard error of
each figure is about 1.3 percentage points).

**Power analysis.** For a win rate *p₁* against *p₀*, the number of trades needed
for a one-sided test at 5% with 80% power is

    n = [ (z₀.₉₅ √(p₀(1−p₀)) + z₀.₈ √(p₁(1−p₁))) / (p₁ − p₀) ]²

which gives 3,863 trades to tell 52% from 50% (4,904 for a two-sided test). For a
Sharpe ratio *S*, the standard error of the estimate after *T* years is about
1/√T (Lo, 2002), so proving *S* > 0 takes about (z₀.₉₅ + z₀.₈)² / S² years:
24.7 years of data for a true Sharpe ratio of 0.5.

**Walk-forward test.** Each year from 2012 to 2025, every setting in a rule's grid is
scored on the previous three years, the best one is chosen, and it is then scored on
the year it has not seen. If past backtests carried information, the chosen setting
should rank above the middle of the grid in its next year, and settings that ranked
high in training should tend to rank high again.

**Deflated Sharpe ratio** (Bailey & López de Prado, 2014). If *N* strategies with
no real edge are tested, the best one will still show a positive Sharpe ratio. The
expected size of that best-by-luck Sharpe ratio is

    SR₀ = σ_SR × [ (1 − γ) Φ⁻¹(1 − 1/N) + γ Φ⁻¹(1 − 1/(N e)) ]

where σ_SR is the spread of Sharpe ratios across the *N* trials and γ ≈ 0.5772 is
the Euler–Mascheroni constant. The deflated Sharpe ratio is the probability that the
best strategy's true Sharpe ratio exceeds SR₀, allowing for skewness and fat tails:

    DSR = Φ( (SR − SR₀) √(T − 1) / √(1 − skew × SR + (kurtosis − 1)/4 × SR²) )

with Sharpe ratios measured per day, *T* the number of days, and kurtosis in its
ordinary form (3 for a normal distribution, not "excess" kurtosis).

## Results: EUR/USD

All numbers are backtests on historical data, not live trading. Full tables:
[`results/REPORT.md`](results/REPORT.md).

**No default rule is distinguishable from luck.** Over 4,334 trading days:

| Rule | Sharpe (95% CI) | Trades | Random-entry p | Trades needed for significance |
|---|---|---|---|---|
| Buy and hold | −0.07 (−0.53 to 0.38) | 1 | – | – |
| MA crossover 50/200 | −0.03 (−0.49 to 0.46) | 27 | 0.585 | – (average trade negative) |
| Breakout 55/20 | 0.02 (−0.47 to 0.50) | 79 | 0.424 | 64,308 |
| Mean reversion 20/2.0 | 0.24 (−0.21 to 0.68) | 163 | 0.087 | 791 |

Mean reversion comes closest, but its Sharpe interval includes zero, 9% of
random-timing versions did at least as well (and with three rules tested, a
Bonferroni-corrected threshold would be 0.05 / 3 ≈ 0.017), its Sharpe ratio falls to
0.13 if trades fill one day later, and it falls to about zero at 10 basis points of
cost per side.

![EUR/USD equity curves](results/eurusd/equity.png)

![Random-entry test](results/eurusd/random_entry.png)

**Choosing parameters by past performance did not help.** In the walk-forward test,
the setting chosen as best over the previous three years ranked, on average, at the
42nd (MA crossover), 38th (breakout) and 45th (mean reversion) percentile of its grid
in the following year, below the middle. The rank correlation between training and
next-year Sharpe ratios across settings averaged −0.13, −0.13 and −0.04: past
rankings carried no useful information. The chosen settings' Sharpe ratios fell
from 0.98, 0.58 and 1.02 in training to 0.00, −0.31 and 0.09 the next year. Some
fall is expected from selection alone even when skill exists; the percentiles and
rank correlations are the more telling evidence. The 14 folds share overlapping
training years, so they are not independent.

![Walk-forward test](results/eurusd/walk_forward.png)

**The best backtest does not survive deflation.** The best of all 628 settings was
mean reversion with L = 20 and k = 3.0: Sharpe ratio 0.53. On its own, the
probability that its true Sharpe ratio is above zero is 99.0%. But the best of 628
zero-edge strategies would be expected to reach a Sharpe ratio of about 0.40, and
the probability of beating that is 72%, well short of the 95% usually required.

**The six-month holdout is too short to judge.** From 29 April to 23 October 2026
(123 prices; the source data ended on 23 October when it was downloaded), buy and
hold lost 4.3%. The default MA crossover gained 4.3% from a single short trade, and
the Sharpe-ratio intervals for the rules are 3 to 6 units wide. One interval
(breakout 20/5, chosen on the previous three years) excludes zero, but with five
trades, six rule versions tested and a bootstrap built from about six 20-day blocks,
that is weak evidence.

## Reproducing

Requires Python 3.10 or later. Run all commands from the repository folder.

```bash
git clone https://github.com/<your-github-username>/skill-or-luck.git
cd skill-or-luck
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -e ".[dev,nifty]"          # "nifty" adds yfinance for the Nifty 50 download

python -m pytest                        # 47 unit tests
python -m skill_or_luck run eurusd      # uses the committed EUR/USD file -> results/
python -m skill_or_luck fetch nifty50   # download the Nifty 50 from Yahoo Finance
python -m skill_or_luck run             # all assets with data (about 10 s each)
python -m skill_or_luck own-record --wins 5 --trades 12 --payoff 2.28
python scripts/calibrate_permutation_test.py
```

`python -m skill_or_luck fetch eurusd` re-downloads the pinned EUR/USD copy and
should reproduce the same file hash; `fetch eurusd --latest` takes newer data from
FRED instead (see the data notes above). Random numbers come from fixed seeds, so
the same data gives the same results.

## What a small personal record can show

`own-record` applies the same reasoning to a trader's own results. For a
self-reported record of 5 wins in 12 trades, with the average win 2.28 times the
average loss:

- the 95% interval for the true win rate is 19% to 68%;
- the break-even win rate at that payoff ratio is 1 / (1 + 2.28) = 30.5%;
- if the true win rate were exactly break-even, 5 or more wins out of 12 would still
  happen 29% of the time;
- about 110 trades would be needed to show the win rate is above break-even, if it
  really were 41.7%.

So the record is consistent with an edge, no edge, or a negative edge.

## Project layout

```
src/skill_or_luck/
  data.py         download, validate and cache prices
  rules.py        the three trading rules and their parameter grids
  backtest.py     positions -> daily returns, trades and metrics
  stats.py        bootstrap, permutation test, power, deflated Sharpe ratio
  experiments.py  runs every experiment for one asset
  plots.py        figures
  report.py       writes results/REPORT.md
  __main__.py     command-line interface
scripts/          calibration check for the permutation test
tests/            unit tests
data/raw/         EUR/USD prices (Nifty 50 is downloaded, not committed)
results/          generated tables, figures and REPORT.md
```

The analysis itself (rules, backtest and statistics) is about 270 lines of code; the
rest handles data, the experiment loop, figures and the report.

## Limitations

- **Costs and carry are simplified.** Costs are a flat assumption per trade. EUR/USD
  positions also earn or pay the interest-rate difference between the dollar and the
  euro (carry), which is ignored. That difference reached roughly 3 percentage points
  a year in 2018–2019, so for rules that hold positions for months it could be as
  large as the effects measured here.
- **Execution.** The base case fills at the decision price; the one-day-delay table
  shows how much that matters.
- **Nifty 50 is a price index.** Dividends (roughly 1–1.5% a year) are excluded from
  buy and hold, and short positions are assumed to be possible (for example through
  index futures) at the same cost.
- **Independence assumptions.** The trade bootstrap treats trades as independent;
  the power formulas assume roughly normal, independent returns.
- **Correlated trials.** The ~200 settings of a rule are strongly correlated, so the
  effective number of independent trials is smaller than the count used in the
  deflated Sharpe ratio. Using the full count makes that test somewhat stricter than
  necessary.
- **Walk-forward costs.** The stitched out-of-sample series does not charge for
  switching settings at the start of each year.
- **Few rules, one market so far.** Three simple rules on one currency pair cannot
  say anything about trading in general.

## Possible improvements

- Add carry (interest-rate differentials) to the EUR/USD returns.
- Replace the fixed cost with spread estimates from intraday or bid/ask data.
- Estimate the effective number of independent trials (for example by clustering
  correlated settings) before deflating.
- Add White's Reality Check or Hansen's SPA test as a second multiple-testing method.
- Test the rules on more markets.

## References

- Bailey, D. H. and López de Prado, M. (2012). The Sharpe ratio efficient frontier. *Journal of Risk* 15(2).
- Bailey, D. H. and López de Prado, M. (2014). The deflated Sharpe ratio: correcting for selection bias, backtest overfitting and non-normality. *Journal of Portfolio Management* 40(5).
- Lo, A. W. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal* 58(4).
- White, H. (2000). A reality check for data snooping. *Econometrica* 68(5).
- Efron, B. and Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.

## How this was built

The code, tests and documentation were written on 29 October 2026 with the help of
an AI assistant (Claude), then checked by running the test suite and the full
analysis, and reviewed by a second, independent AI review pass whose findings were
fixed the same day. The analysis of 2009–2026 prices is retrospective: it was run on
that date, on data downloaded that day. See [DEVLOG.md](DEVLOG.md) for what was done
and changed, and when.

## Licence

MIT. See [LICENSE](LICENSE).
