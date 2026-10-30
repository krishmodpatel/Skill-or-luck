# Development log

This log separates three kinds of dates: the period the **data** covers, the day the
**analysis was run**, and the day the **code was written and published**. The
analysis is retrospective: it looks back at historical prices; it is not a record
of trades made at the time.

## 2026-10-29 — first version

Written with the help of an AI assistant (Claude) in one session; tests and the
full analysis were run in a Linux environment with Python 3.13.

**Data**
- EUR/USD: Federal Reserve H.10 series DEXUSEU (New York noon buying rates),
  1999-01-04 to 2026-10-23, the latest available at download. FRED itself could not
  be reached from the build environment, so the datahub.io copy of the same series
  was used (commit `c28e171`, inverted from EUR per USD). First retrieved
  2026-10-28 20:39 UTC (2026-10-29 02:09 IST) and downloaded again at 21:26 UTC after
  the review below, with an identical file hash. 281 missing values (all on weekdays)
  were dropped; no gaps over 7 days and no suspicious moves were found.
- Nifty 50: Yahoo Finance could not be reached from the build environment (its
  network policy blocks the site), so no Nifty 50 results exist yet. The download
  code (`fetch_nifty50`) has not been run against the live service; its parsing step
  is covered by a unit test using a hand-made table.

**Decisions made before running any backtest**
- Default parameters: 50/200 moving averages, 55/20 breakout, 20-day/2.0 z-score.
- Main sample from 2009-01-01; holdout from 2026-04-29 (the six months requested
  for the project), not used for any choice.
- Costs: 1 bp per side for EUR/USD, 5 bp per side for the Nifty 50.
- 5,000 permutations and 5,000 bootstrap resamples; seed 2026.

**Changes after the first run (found by looking at the output)**
- Some mean-reversion settings could never trade: a z-score over *n* points cannot
  exceed (n − 1)/√n, so for example a 5-day window can never reach 2.0. Their Sharpe
  ratio was undefined (0/0), which broke the deflated-Sharpe calculation. These 6
  settings were removed from the grid (220 → 214), with a test of the bound.
- In the walk-forward test, a chosen setting that did not trade at all in the next
  year produced an undefined Sharpe ratio and was silently skipped, which made the
  results look better than they were. A strategy that stays flat now scores 0.
- The default-parameter results did not change with these fixes.

**Changes after an independent review (same day)**

A separate AI review pass, which had not seen the code being written, checked the
formulas, recomputed several results independently (they matched), and reported
problems. Fixed:
- *Random-entry test.* The random strategies could merge two neighbouring trades in
  the same direction, so they did not always keep the rule's number of trades and
  costs, as the documentation claimed (for the MA crossover, 15 trades on average
  instead of 27). The null now keeps the sequence of long and short trades, shuffles
  holding periods within each direction, and leaves at least one flat day between
  same-direction neighbours, so trades, days long and short, and costs are
  identical. This changed the p-values of the default rules from 0.573, 0.456 and
  0.123 to 0.585, 0.424 and 0.087. A calibration script was added: on rules with no
  information it gives p ≤ 0.05 in 5–7% of 300 runs per rule.
- *Command line.* `fetch` and `run` without asset names failed because of how
  `argparse` checks defaults. Fixed, and command-line tests added.
- *Reproducibility.* EUR/USD now comes from the pinned mirror by default and the file
  is committed, so a fresh clone reproduces the results exactly. `--latest` uses FRED.
- *Wording.* The EUR/USD series is a noon rate, not a close or a fixing; the base
  case fills at the decision price, which is optimistic (a one-day-delay table was
  added); the holdout is computed in every run but never used for any choice.
- *Data checks.* Added a check for moves over 8 times the typical move that reverse
  the next day; the 15% check alone was too loose for currencies.
- *Smaller items.* pandas 2.1 compatibility, `yfinance` made optional, a Sharpe
  interval for buy and hold, a dead line removed, an all-NaN warning handled.

**Verified**
- 47 unit tests pass (`python -m pytest`), also with warnings treated as errors.
- `python -m skill_or_luck run` completes in about 10 seconds for EUR/USD.
- A clean install in a new virtual environment, a fresh download and a re-run gave
  identical tables and the same data hash.

**Not verified**
- Nifty 50 download and results (see above).
- Behaviour on Windows and macOS (the code uses only portable libraries, but it was
  only run on Linux).
