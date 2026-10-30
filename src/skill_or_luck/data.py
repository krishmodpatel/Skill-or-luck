"""Download, validate and cache daily prices."""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
MANIFEST = Path("results/data_manifest.json")

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXUSEU"
# The same Fed H.10 series, republished under the PDDL licence by datahub.io. It is
# stored as EUR per USD rounded to 4 decimals, so it is inverted back to USD per EUR.
# Pinned to a commit so the file cannot change underneath a reproduction.
MIRROR_URL = (
    "https://raw.githubusercontent.com/datasets/exchange-rates/"
    "c28e171e36a46bd2941cf66156acd8efaf9906f5/data/daily.csv"
)
NIFTY_TICKER = "^NSEI"

MAX_GAP_DAYS = 7         # longer gaps than this are reported (holidays are shorter)
MAX_ABS_LOG_MOVE = 0.15  # moves this large are reported even if they do not reverse
SPIKE_MULTIPLE = 8       # a move this many times the typical one that then reverses
                         # is the usual sign of a bad print


def _download_text(url: str, timeout: int = 30) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def parse_fred_csv(text: str) -> pd.Series:
    """FRED CSVs have a date column (named DATE or observation_date) and the series."""
    df = pd.read_csv(io.StringIO(text), na_values=["."])
    dates = pd.to_datetime(df.iloc[:, 0])
    return pd.Series(df.iloc[:, 1].to_numpy(dtype=float), index=dates, name="close")


def parse_mirror_csv(text: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    euro = df[df["Country"] == "Euro"]
    eur_per_usd = euro["Exchange rate"].to_numpy(dtype=float)
    return pd.Series(1.0 / eur_per_usd, index=pd.to_datetime(euro["Date"]), name="close")


def fetch_eurusd(latest: bool = False) -> tuple[pd.Series, str]:
    """The pinned mirror by default, so results reproduce exactly; FRED for newer data.

    The two differ slightly: FRED publishes USD per EUR to 4 decimals, the mirror
    EUR per USD to 4 decimals, so inverted values differ in the 5th significant digit.
    """
    if latest:
        return parse_fred_csv(_download_text(FRED_URL)), f"FRED DEXUSEU ({FRED_URL})"
    return (
        parse_mirror_csv(_download_text(MIRROR_URL)),
        f"FRED DEXUSEU via datahub.io mirror, inverted from EUR per USD ({MIRROR_URL})",
    )


def normalise_yahoo_history(df: pd.DataFrame) -> pd.Series:
    """Yahoo returns timezone-aware timestamps at local midnight; keep the calendar date."""
    if df.empty:
        raise ValueError("Yahoo Finance returned no rows")
    index = pd.DatetimeIndex(df.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    return pd.Series(df["Close"].to_numpy(dtype=float), index=index.normalize(), name="close")


def fetch_nifty50() -> tuple[pd.Series, str]:
    try:
        import yfinance as yf  # only needed for this asset
    except ImportError as exc:
        raise ImportError('install the Nifty 50 extra first: pip install -e ".[nifty]"') from exc

    history = yf.Ticker(NIFTY_TICKER).history(
        start="2007-01-01", interval="1d", auto_adjust=False, actions=False, raise_errors=True
    )
    return normalise_yahoo_history(history), f"Yahoo Finance {NIFTY_TICKER} via yfinance"



def clean_prices(prices: pd.Series) -> tuple[pd.Series, list[str]]:
    """Sort, de-duplicate and check a price series. Returns the series and notes."""
    prices = prices.astype(float).copy()
    prices.index = pd.DatetimeIndex(prices.index).normalize()
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()

    notes = []
    missing = int(prices.isna().sum())
    if missing:
        notes.append(f"dropped {missing} missing values")
    prices = prices.dropna()

    if len(prices) < 500:
        raise ValueError(f"only {len(prices)} prices; at least 500 are needed")
    if (prices <= 0).any():
        raise ValueError("prices must be positive")

    gaps = prices.index.to_series().diff().dt.days
    for date, days in gaps[gaps > MAX_GAP_DAYS].items():
        notes.append(f"{days}-day gap ending {date.date()}")
    moves = np.log(prices).diff()
    typical = moves.abs().rolling(250, min_periods=50).median().shift(1)
    reverses = (moves * moves.shift(-1) < 0) & (moves.shift(-1).abs() > 0.5 * moves.abs())
    suspicious = (moves.abs() > MAX_ABS_LOG_MOVE) | ((moves.abs() > SPIKE_MULTIPLE * typical) & reverses)
    for date, move in moves[suspicious].items():
        notes.append(f"suspicious daily move {move:+.2%} (log) on {date.date()}; check the source")
    return prices, notes


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_prices(asset: str, prices: pd.Series, source: str) -> list[str]:
    prices, notes = clean_prices(prices)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{asset}.csv"
    prices.rename("close").rename_axis("date").to_csv(path, float_format="%.6f")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    manifest[asset] = {
        "source": source,
        "retrieved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "first_date": str(prices.index[0].date()),
        "last_date": str(prices.index[-1].date()),
        "rows": len(prices),
        "sha256": _sha256(path),
        "validation_notes": notes,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return notes


def fetch(asset: str, latest: bool = False) -> list[str]:
    prices, source = fetch_eurusd(latest) if asset == "eurusd" else fetch_nifty50()
    return save_prices(asset, prices, source)


def import_csv(asset: str, path: str) -> list[str]:
    """Import a manually downloaded file with 'Date' and 'Close' columns."""
    df = pd.read_csv(path)
    columns = {c.strip().lower(): c for c in df.columns}
    if "date" not in columns or "close" not in columns:
        raise ValueError("the CSV needs 'Date' and 'Close' columns")
    prices = pd.Series(
        pd.to_numeric(df[columns["close"]], errors="coerce").to_numpy(),
        index=pd.to_datetime(df[columns["date"]]),
    )
    return save_prices(asset, prices, f"manual CSV import ({Path(path).name})")


def load_prices(asset: str) -> pd.Series:
    path = RAW_DIR / f"{asset}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m skill_or_luck fetch {asset}`")
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    return df["close"]
