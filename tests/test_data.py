import numpy as np
import pandas as pd
import pytest

from skill_or_luck import data


def test_clean_prices_sorts_deduplicates_and_drops_missing():
    dates = list(pd.bdate_range("2020-01-01", periods=600))
    values = np.linspace(1.0, 2.0, 600)
    raw = pd.Series(values, index=dates)
    raw.iloc[10] = np.nan
    messy = pd.concat([raw.iloc[300:], raw.iloc[:300], raw.iloc[[5]] * 1.001])  # unsorted, duplicate
    clean, notes = data.clean_prices(messy)
    assert clean.index.is_monotonic_increasing and clean.index.is_unique
    assert len(clean) == 599
    assert clean.iloc[5] == pytest.approx(values[5] * 1.001)  # the later duplicate wins
    assert any("missing" in n for n in notes)


def test_clean_prices_rejects_bad_data():
    dates = pd.bdate_range("2020-01-01", periods=600)
    with pytest.raises(ValueError):
        data.clean_prices(pd.Series(np.r_[np.ones(599), -1.0], index=dates))
    with pytest.raises(ValueError):
        data.clean_prices(pd.Series(np.ones(100), index=dates[:100]))


def test_clean_prices_reports_gaps_and_jumps():
    dates = list(pd.bdate_range("2020-01-01", periods=600))
    dates[300:] = [d + pd.Timedelta(days=30) for d in dates[300:]]
    values = np.ones(600)
    values[450:] = 1.5
    _, notes = data.clean_prices(pd.Series(values, index=dates))
    assert any("gap" in n for n in notes)
    assert any("suspicious daily move" in n for n in notes)


def test_clean_prices_flags_a_bad_print_that_reverses(random_walk):
    prices = random_walk.copy()
    prices.iloc[800] *= 1.10  # one wrong price: up 10%, then straight back down
    notes = [n for n in data.clean_prices(prices)[1] if "suspicious" in n]
    assert len(notes) == 1 and str(prices.index[800].date()) in notes[0]


def test_parse_fred_csv_handles_missing_marker():
    text = "observation_date,DEXUSEU\n2026-01-01,.\n2026-01-02,1.1000\n"
    series = data.parse_fred_csv(text)
    assert np.isnan(series.iloc[0]) and series.iloc[1] == pytest.approx(1.1)


def test_parse_mirror_inverts_eur_per_usd():
    text = "Date,Country,Exchange rate\n2026-01-02,Euro,0.8000\n2026-01-02,Japan,150.0\n"
    series = data.parse_mirror_csv(text)
    assert len(series) == 1 and series.iloc[0] == pytest.approx(1.25)


def test_normalise_yahoo_history_drops_timezone():
    index = pd.DatetimeIndex(["2026-04-29 00:00", "2026-04-30 00:00"]).tz_localize("Asia/Kolkata")
    frame = pd.DataFrame({"Close": [24000.0, 24100.0]}, index=index)
    series = data.normalise_yahoo_history(frame)
    assert series.index.tz is None
    assert series.index[0] == pd.Timestamp("2026-04-29")
    with pytest.raises(ValueError):
        data.normalise_yahoo_history(frame.iloc[0:0])


def test_import_csv_round_trip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dates = pd.bdate_range("2020-01-01", periods=600)
    pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"), "Close": np.linspace(100, 120, 600)}).to_csv(
        "nifty.csv", index=False)
    data.import_csv("nifty50", "nifty.csv")
    loaded = data.load_prices("nifty50")
    assert len(loaded) == 600 and loaded.iloc[-1] == pytest.approx(120)
    assert "nifty50" in (tmp_path / "results" / "data_manifest.json").read_text(encoding="utf-8")
