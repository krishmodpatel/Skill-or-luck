import numpy as np
import pandas as pd
import pytest

from skill_or_luck.__main__ import main


def test_own_record_runs(capsys):
    main(["own-record", "--wins", "5", "--trades", "12", "--payoff", "2.28"])
    assert "30.5%" in capsys.readouterr().out


def test_bad_arguments_give_a_clean_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for argv in (["fetch", "bitcoin"], ["run"], ["run", "nifty50"],
                 ["own-record", "--wins", "13", "--trades", "12", "--payoff", "2"]):
        with pytest.raises(SystemExit) as exit_info:
            main(argv)
        assert exit_info.value.code == 2


def test_run_end_to_end_on_synthetic_data(tmp_path, monkeypatch):
    """Synthetic prices (not market data) covering the whole design, 2007 to 2026."""
    monkeypatch.chdir(tmp_path)
    dates = pd.bdate_range("2007-01-01", "2026-10-29")
    prices = 1.2 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.006, len(dates))))
    pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"), "Close": prices}).to_csv("p.csv", index=False)
    main(["import-csv", "eurusd", "p.csv"])
    main(["run", "eurusd", "--n-perm", "20", "--n-boot", "20"])
    report = (tmp_path / "results" / "REPORT.md").read_text()
    assert "## EUR/USD" in report and "Six-month holdout" in report
    assert (tmp_path / "results" / "eurusd" / "walk_forward.png").exists()
