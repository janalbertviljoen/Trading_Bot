"""Reproducible backtest harness behind the trading-bot MCP server.

Caches price datasets as CSV under ``data/``, splits them chronologically into
in-sample / out-of-sample segments, runs the repo strategy
(``trade_indicators.detect_all`` -> ``trade_wallet.make_trades``) and summarises
performance against buy-and-hold.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
BASELINE_PATH = REPO_ROOT / "benchmarks" / "baseline.json"

sys.path.insert(0, str(REPO_ROOT))

import read_data  # noqa: E402
import trade_indicators  # noqa: E402
import trade_wallet  # noqa: E402

DETECT_KEYS = ["threshold", "rolling_window"]
TRADE_KEYS = [
    "initial_investment",
    "initial_trade_fraction",
    "trade_fraction_cash",
    "trade_fraction_volume",
]
DEFAULT_CONFIG: dict[str, float] = {
    "threshold": 0.03,
    "rolling_window": 24,
    "initial_investment": 100,
    "initial_trade_fraction": 0.5,
    "trade_fraction_cash": 0.1,
    "trade_fraction_volume": 0.1,
}
MAX_COMBINATIONS = 200


def dataset_path(symbol: str, interval: str, start_date: str, end_date: str) -> Path:
    """Build the cache path of a dataset.

    Args:
        symbol (str): Trading pair, e.g. "BTCUSDT".
        interval (str): Bar size, e.g. "1h".
        start_date (str): First date, YYYY-MM-DD.
        end_date (str): Exclusive end date, YYYY-MM-DD.

    Returns:
        Path: ``data/<symbol>_<interval>_<start_date>_<end_date>.csv``.
    """
    return DATA_DIR / f"{symbol}_{interval}_{start_date}_{end_date}.csv"


def fetch_dataset(
    symbol: str, interval: str, start_date: str, end_date: str | None
) -> dict[str, Any]:
    """Fetch price history from Binance and cache it as a CSV for reproducible backtests.

    Args:
        symbol (str): Trading pair, e.g. "BTCUSDT".
        interval (str): Bar size, e.g. "1h".
        start_date (str): First date to fetch, YYYY-MM-DD (UTC).
        end_date (str | None): Exclusive end date, YYYY-MM-DD. If None, all data up
            to now is kept and the last timestamp's date is used in the file name.

    Returns:
        dict: ``describe_dataset`` output for the written file.

    Edge cases:
        Overwrites an existing file with the same name.
        Always downloads up to now, then truncates to ``end_date``.

    Raises:
        requests.RequestException: Network failure.
    """
    df = read_data.fetch_btc_history(
        interval=interval, start_date=start_date, symbol=symbol
    )
    if end_date:
        df = df[df["timestamp"] < pd.Timestamp(end_date)].reset_index(drop=True)
        last_date = end_date
    else:
        last_date = str(df["timestamp"].max().strftime("%Y-%m-%d"))
    path = dataset_path(symbol, interval, start_date, last_date)
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(path, index=False)
    return describe_dataset(path)


def describe_dataset(path: Path) -> dict[str, Any]:
    """Summarise a cached dataset.

    Args:
        path (Path): CSV file with a ``timestamp`` column.

    Returns:
        dict: ``dataset`` (str, file stem), ``rows`` (int), ``start`` and ``end``
        (str, first and last ``timestamp`` as written in the CSV).

    Raises:
        FileNotFoundError: ``path`` does not exist.
        IndexError: The file has no rows.
    """
    df = pd.read_csv(path, usecols=["timestamp"])
    return {
        "dataset": path.stem,
        "rows": len(df),
        "start": df["timestamp"].iloc[0],
        "end": df["timestamp"].iloc[-1],
    }


def list_datasets() -> list[dict[str, Any]]:
    """List all cached datasets.

    Returns:
        list[dict]: ``describe_dataset`` output per ``data/*.csv``, sorted by file
        name; empty if ``data/`` does not exist.
    """
    if not DATA_DIR.exists():
        return []
    return [describe_dataset(p) for p in sorted(DATA_DIR.glob("*.csv"))]


def load_dataset(dataset: str) -> pd.DataFrame:
    """Load a cached dataset by name.

    Args:
        dataset (str): Dataset name (file stem) as returned by ``list_datasets``.

    Returns:
        pandas.DataFrame: CSV contents with ``timestamp`` parsed to datetime64[ns]
        and a ``RangeIndex`` (columns timestamp, high, low, price, volume).

    Raises:
        ValueError: No such dataset; the message lists the available names.
    """
    path = DATA_DIR / f"{dataset}.csv"
    if not path.exists():
        available = [d["dataset"] for d in list_datasets()]
        raise ValueError(f"Unknown dataset '{dataset}'. Available: {available}")
    return pd.read_csv(path, parse_dates=["timestamp"])


def split_dataset(
    df: pd.DataFrame, holdout_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time series chronologically into in-sample and out-of-sample parts.

    Formula:
        split = int(len(df) * (1 - holdout_fraction))
        train = rows [0, split), test = rows [split, n)

    Args:
        df (pandas.DataFrame): Time-ordered data.
        holdout_fraction (float): Fraction of rows held out at the end, in [0, 1).

    Returns:
        tuple[pandas.DataFrame, pandas.DataFrame]: ``(train, test)``, each with the
        index reset to 0..len-1 (required by ``trade_wallet.update_wallet``).

    Edge cases:
        holdout_fraction == 0 -> ``test`` is empty.

    Raises:
        ValueError: ``holdout_fraction`` outside [0, 1).
    """
    if not 0 <= holdout_fraction < 1:
        raise ValueError("holdout_fraction must be in [0, 1)")
    split = int(len(df) * (1 - holdout_fraction))
    train = df.iloc[:split].reset_index(drop=True)
    test = df.iloc[split:].reset_index(drop=True)
    return train, test


def simulate(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Run the repo's signal detection and wallet logic on a copy of the price data.

    Args:
        df (pandas.DataFrame): Price data with a ``RangeIndex`` from 0 and columns
            ``price`` and ``volume`` (float64).
        config (dict): Must contain every key in ``DETECT_KEYS`` and ``TRADE_KEYS``.
            ``rolling_window`` is cast to int.

    Returns:
        pandas.DataFrame: New frame with indicator, ``signal``, ``direction`` and
        wallet columns (see ``trade_wallet.update_wallet``). ``df`` is unchanged.

    Raises:
        KeyError: A required config key is missing.
    """
    detect = {k: config[k] for k in DETECT_KEYS}
    trade = {k: config[k] for k in TRADE_KEYS}
    detect["rolling_window"] = int(detect["rolling_window"])
    df = trade_indicators.detect_all(df.copy(), **detect)
    return trade_wallet.make_trades(df, **trade)


def summarise(df: pd.DataFrame) -> dict[str, Any]:
    """Compute performance metrics of a simulated run against buy-and-hold.

    Formula:
        drawdown_t = wallet_t / max(wallet_0..t) - 1
        returns_t  = wallet_t / wallet_{t-1} - 1
        sharpe     = mean(returns) / std(returns) * sqrt(len(returns))
        excess     = wallet_final - price_final   (both normalised to 1 at row 0)

    Args:
        df (pandas.DataFrame): Output of ``simulate`` with ``timestamp``,
            ``normalised_wallet_value``, ``normalised_price`` and ``trade_volume``.

    Returns:
        dict: ``start``, ``end`` (str); ``rows`` (int);
        ``final_normalised_wallet``, ``buy_and_hold``, ``excess_vs_buy_and_hold``,
        ``mean_normalised_wallet``, ``max_drawdown`` (float, 6 dp; drawdown <= 0);
        ``sharpe_unannualised`` (float, 4 dp); ``n_buys``, ``n_sells`` (int).

    Assumptions:
        Sharpe uses a zero risk-free rate and is scaled by the number of steps in
        the run, not annualised. No fees are modelled.

    Edge cases:
        Zero return variance -> sharpe 0.0.

    Raises:
        IndexError: ``df`` is empty.
    """
    wallet = df["normalised_wallet_value"]
    price = df["normalised_price"]
    drawdown = wallet / wallet.cummax() - 1
    returns = wallet.pct_change().dropna()
    sharpe = (
        float(returns.mean() / returns.std() * np.sqrt(len(returns)))
        if returns.std() > 0
        else 0.0
    )
    final_wallet = float(wallet.iloc[-1])
    final_price = float(price.iloc[-1])
    return {
        "start": str(df["timestamp"].iloc[0]),
        "end": str(df["timestamp"].iloc[-1]),
        "rows": len(df),
        "final_normalised_wallet": round(final_wallet, 6),
        "buy_and_hold": round(final_price, 6),
        "excess_vs_buy_and_hold": round(final_wallet - final_price, 6),
        "mean_normalised_wallet": round(float(wallet.mean()), 6),
        "max_drawdown": round(float(drawdown.min()), 6),
        "sharpe_unannualised": round(sharpe, 4),
        "n_buys": int((df["trade_volume"] > 0).sum()),
        "n_sells": int((df["trade_volume"] < 0).sum()),
    }


def run_backtest(
    dataset: str, config: dict[str, Any], holdout_fraction: float = 0.0
) -> dict[str, Any]:
    """Backtest one configuration on a cached dataset.

    Args:
        dataset (str): Dataset name from ``list_datasets``.
        config (dict): Overrides merged onto ``DEFAULT_CONFIG``.
        holdout_fraction (float): If > 0, score in-sample and out-of-sample
            segments separately; if 0, score the full series. Default 0.0.

    Returns:
        dict: ``dataset``, ``config`` (merged) and either ``in_sample`` and
        ``out_of_sample`` or ``full`` (each a ``summarise`` dict).

    Edge cases:
        Each segment starts a fresh wallet; out-of-sample is not a continuation.

    Raises:
        ValueError: Unknown dataset or invalid ``holdout_fraction``.
    """
    config = DEFAULT_CONFIG | config
    df = load_dataset(dataset)
    result: dict[str, Any] = {"dataset": dataset, "config": config}
    if holdout_fraction:
        train, test = split_dataset(df, holdout_fraction)
        result["in_sample"] = summarise(simulate(train, config))
        result["out_of_sample"] = summarise(simulate(test, config))
    else:
        result["full"] = summarise(simulate(df, config))
    return result


def optimise(
    dataset: str,
    param_grid: dict[str, list[Any]],
    base_config: dict[str, Any],
    holdout_fraction: float,
    rank_by: str,
    top_n: int,
) -> dict[str, Any]:
    """Grid-search parameters in-sample and re-score the best candidates out-of-sample.

    Args:
        dataset (str): Dataset name from ``list_datasets``.
        param_grid (dict[str, list]): Candidate values per ``DEFAULT_CONFIG`` key.
        base_config (dict): Fixed overrides merged onto ``DEFAULT_CONFIG``.
        holdout_fraction (float): Fraction held out for out-of-sample, in [0, 1).
        rank_by (str): ``summarise`` metric to rank by; descending, except
            ``max_drawdown`` which is sorted ascending.
        top_n (int): Number of candidates to return.

    Returns:
        dict: ``dataset``, ``combinations_tested`` (int), ``rank_by``,
        ``base_config`` and ``top``: a list of ``{"params", "in_sample",
        "out_of_sample"}`` (``out_of_sample`` only if the holdout is non-empty).

    Edge cases:
        Ranking by ``max_drawdown`` ascending puts the *deepest* drawdown first.

    Raises:
        ValueError: Unknown parameter names, more than ``MAX_COMBINATIONS``
            combinations, unknown dataset or invalid ``holdout_fraction``.
        KeyError: ``rank_by`` is not a ``summarise`` metric.
    """
    unknown = set(param_grid) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown parameters {unknown}. Valid: {list(DEFAULT_CONFIG)}")
    combinations = [
        dict(zip(param_grid, combo, strict=True))
        for combo in itertools.product(*param_grid.values())
    ]
    if len(combinations) > MAX_COMBINATIONS:
        raise ValueError(
            f"{len(combinations)} combinations exceeds limit of {MAX_COMBINATIONS}; narrow the grid"
        )

    config = DEFAULT_CONFIG | base_config
    train, test = split_dataset(load_dataset(dataset), holdout_fraction)
    scored: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for combo in combinations:
        candidate = config | combo
        scored.append((candidate, summarise(simulate(train, candidate))))

    scored.sort(key=lambda item: item[1][rank_by], reverse=rank_by != "max_drawdown")
    top: list[dict[str, Any]] = []
    for candidate, in_sample in scored[:top_n]:
        entry: dict[str, Any] = {
            "params": {k: candidate[k] for k in param_grid},
            "in_sample": in_sample,
        }
        if len(test):
            entry["out_of_sample"] = summarise(simulate(test, candidate))
        top.append(entry)
    return {
        "dataset": dataset,
        "combinations_tested": len(combinations),
        "rank_by": rank_by,
        "base_config": config,
        "top": top,
    }


def regression_check(dataset: str, update: bool) -> dict[str, Any]:
    """Compare the default-config backtest with the stored baseline.

    Args:
        dataset (str): Dataset name from ``list_datasets``.
        update (bool): If True, overwrite the stored baseline with the current
            metrics.

    Returns:
        dict: ``status`` ("baseline_written", "changed" or "unchanged"),
        ``dataset``, and ``metrics`` (when written) or ``differences``
        (``{metric: {"baseline", "current"}}``).

    Edge cases:
        The first run for a dataset writes its baseline. Comparison is exact
        equality of the rounded ``summarise`` values.
    """
    current = run_backtest(dataset, {})["full"]
    baselines: dict[str, dict[str, Any]] = (
        json.loads(BASELINE_PATH.read_text()) if BASELINE_PATH.exists() else {}
    )

    if update or dataset not in baselines:
        baselines[dataset] = current
        BASELINE_PATH.parent.mkdir(exist_ok=True)
        BASELINE_PATH.write_text(json.dumps(baselines, indent=2))
        return {"status": "baseline_written", "dataset": dataset, "metrics": current}

    baseline = baselines[dataset]
    diffs: dict[str, dict[str, Any]] = {
        k: {"baseline": baseline[k], "current": current[k]}
        for k in current
        if baseline.get(k) != current[k]
    }
    return {
        "status": "changed" if diffs else "unchanged",
        "dataset": dataset,
        "differences": diffs,
    }
