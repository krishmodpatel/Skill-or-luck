import numpy as np
import pytest

from skill_or_luck import stats


def test_wilson_interval_for_5_of_12():
    low, high = stats.wilson_interval(5, 12)
    assert low == pytest.approx(0.193, abs=0.001)
    assert high == pytest.approx(0.680, abs=0.001)


def test_clopper_pearson_edges_and_known_value():
    assert stats.clopper_pearson(0, 10)[0] == 0.0
    assert stats.clopper_pearson(10, 10)[1] == 1.0
    low, high = stats.clopper_pearson(5, 12)
    assert low == pytest.approx(0.1517, abs=0.001)
    assert high == pytest.approx(0.7233, abs=0.001)


def test_trades_needed_for_52_percent():
    assert stats.trades_needed(0.5, 0.52) == pytest.approx(3863, abs=5)
    assert stats.trades_needed(0.5, 0.52, one_sided=False) == pytest.approx(4904, abs=5)


def test_years_needed_for_sharpe():
    assert stats.years_needed(0.5) == pytest.approx(24.7, abs=0.1)
    assert stats.years_needed(1.0) == pytest.approx(stats.years_needed(0.5) / 4)


def test_runs_and_random_positions_preserve_trades():
    position = np.array([0, 1, 1, 0, 1, 0, 0, -1, 0, 1, 1, 1, -1, -1], dtype=float)
    directions, lengths, flat = stats.runs(position)
    assert directions.tolist() == [1, 1, -1, 1, -1] and lengths.tolist() == [2, 1, 1, 3, 2]
    assert flat == 5
    rng = np.random.default_rng(1)
    for _ in range(200):
        shuffled = stats.random_entry_positions(directions, lengths, flat, rng)
        d, n, f = stats.runs(shuffled)
        assert len(shuffled) == len(position) and f == flat
        assert d.tolist() == directions.tolist()  # same trades, none merged
        assert sorted(n[d == 1]) == sorted(lengths[directions == 1])
        assert sorted(n[d == -1]) == sorted(lengths[directions == -1])
        # same turnover, so the same trading costs
        assert np.abs(np.diff(np.r_[0, shuffled, 0])).sum() == np.abs(np.diff(np.r_[0, position, 0])).sum()


def test_random_positions_for_an_always_invested_rule():
    position = np.array([0] + [1] * 5 + [-1] * 3 + [1] * 2 + [-1] * 4, dtype=float)
    directions, lengths, flat = stats.runs(position)
    shuffled = stats.random_entry_positions(directions, lengths, flat, np.random.default_rng(2))
    assert len(stats.runs(shuffled)[1]) == 4


def test_random_entry_test_detects_foresight_and_not_noise():
    rng = np.random.default_rng(2)
    returns = rng.normal(0, 0.01, 2000)
    noise = np.repeat(rng.choice([-1.0, 1.0], 100), 20)
    assert stats.random_entry_test(noise, returns, 999, rng)["p_value"] > 0.01
    # knowing tomorrow's sign is the strongest possible edge
    foresight = np.sign(returns)
    assert stats.random_entry_test(foresight, returns, 999, rng)["p_value"] == pytest.approx(0.001)


def test_random_entry_test_with_no_trades_is_nan():
    result = stats.random_entry_test(np.zeros(100), np.ones(100), 10, np.random.default_rng(0))
    assert np.isnan(result["p_value"])


def test_bootstrap_intervals_contain_the_estimate():
    rng = np.random.default_rng(3)
    trades = rng.normal(0.002, 0.02, 300)
    ci = stats.bootstrap_trades(trades, 2000, rng)
    assert ci["mean_low"] < trades.mean() < ci["mean_high"]
    assert ci["win_low"] < (trades > 0).mean() < ci["win_high"]
    assert np.isnan(stats.bootstrap_trades([0.01], 100, rng)["mean_low"])


def test_block_bootstrap_sharpe_interval():
    rng = np.random.default_rng(4)
    daily = rng.normal(0.0004, 0.01, 2520)  # about 10 years; true Sharpe about 0.63
    sample_sharpe = daily.mean() / daily.std(ddof=1) * np.sqrt(252)
    low, high = stats.block_bootstrap_sharpe(daily, 252, 1000, rng)
    assert low < sample_sharpe < high
    # standard error is about 1/sqrt(10 years), so the 95% interval is about 1.24 wide
    assert 0.9 < high - low < 1.6


def test_probabilistic_sharpe_basics():
    rng = np.random.default_rng(5)
    strong = rng.normal(0.002, 0.01, 2000)
    assert stats.probabilistic_sharpe(strong) > 0.99
    assert stats.probabilistic_sharpe(-strong) < 0.01


def test_deflation_raises_the_bar_with_more_trials():
    rng = np.random.default_rng(6)
    returns = rng.normal(0.0005, 0.01, 2500)
    few = rng.normal(0, 0.02, 5)
    many = rng.normal(0, 0.02, 500)
    dsr_few, hurdle_few = stats.deflated_sharpe(returns, few)
    dsr_many, hurdle_many = stats.deflated_sharpe(returns, many)
    assert hurdle_many > hurdle_few > 0
    assert dsr_many < dsr_few <= stats.probabilistic_sharpe(returns)
    assert stats.expected_max_sharpe([0.1]) == 0.0


def test_expected_max_sharpe_matches_simulation():
    """For N independent zero-edge strategies the hurdle approximates their expected best."""
    rng = np.random.default_rng(7)
    n_trials, sd = 100, 0.02
    trials = rng.standard_normal(n_trials)
    trials = (trials - trials.mean()) / trials.std(ddof=1) * sd  # sample sd exactly `sd`
    simulated_best = rng.normal(0, sd, (20000, n_trials)).max(axis=1).mean()
    assert stats.expected_max_sharpe(trials) == pytest.approx(simulated_best, rel=0.03)
