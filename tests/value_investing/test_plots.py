"""Grid shapes of the price figures (value_investing/plots.py)."""

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from Toolbox.value_investing.plots import next_prices, pick_prices

DAYS = pd.date_range("2024-01-01", periods=5, freq="D")


def _prices(symbols: list[str]) -> pd.DataFrame:
    rows = [
        (s, d, 10.0 + i + 0.1 * k)
        for i, s in enumerate(symbols)
        for k, d in enumerate(DAYS)
    ]
    return pd.DataFrame(rows, columns=["symbol", "date", "close"])


def _trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decision_date": [DAYS[1]],
            "symbol": ["S00"],
            "action": ["buy"],
            "usd": [100.0],
        }
    )


def _shape(figure: object) -> tuple[int, int, int]:
    axes = figure.axes  # type: ignore[attr-defined]
    grid = [a for a in axes if a.get_subplotspec() is not None]
    rows = {a.get_subplotspec().rowspan.start for a in grid}
    cols = {a.get_subplotspec().colspan.start for a in grid}
    shown = sum(1 for a in grid if a.get_visible())
    return len(rows), len(cols), shown


def test_next_prices_ten_picks_make_two_rows_of_five() -> None:
    symbols = [f"S{i:02d}" for i in range(10)]
    figure = next_prices(
        _prices(symbols), _trades(), pd.DataFrame({"symbol": symbols}), years=5
    )
    assert _shape(figure) == (2, 5, 10)


def test_next_prices_three_picks_make_one_row() -> None:
    symbols = ["S00", "S01", "S02"]
    figure = next_prices(
        _prices(symbols), _trades(), pd.DataFrame({"symbol": symbols}), years=5
    )
    rows, cols, shown = _shape(figure)
    assert (rows, cols, shown) == (1, 3, 3)


def test_pick_prices_within_the_page_uses_four_columns() -> None:
    symbols = [f"S{i:02d}" for i in range(8)]
    figure = pick_prices(_prices(symbols), _trades(), 2023, 5, page=20, ends=3)
    assert _shape(figure) == (2, 4, 8)


def test_pick_prices_beyond_the_page_shows_the_ends_in_two_rows() -> None:
    symbols = [f"S{i:02d}" for i in range(12)]
    figure = pick_prices(_prices(symbols), _trades(), 2023, 5, page=8, ends=4)
    # best 4 on the first row, worst 4 on the second
    assert _shape(figure) == (2, 4, 8)
    titles = [a.get_title() for a in figure.axes if a.get_visible()]
    assert titles[0].startswith(
        "S00"
    )  # the same rise on the lowest base is the best change
    assert any(t.startswith("S11") for t in titles[4:])
