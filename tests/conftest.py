import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def random_walk():
    """Synthetic prices for tests only (not market data): a seeded random walk."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2015-01-01", periods=1500)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates)))), index=dates)
