import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def prices():
    """Synthetic hourly price series oscillating around a slow trend."""
    n = 200
    t = np.arange(n)
    price = 100 + 0.05 * t + 5 * np.sin(t / 6)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
            "high": price * 1.01,
            "low": price * 0.99,
            "price": price,
            "volume": np.full(n, 10.0),
        }
    )
