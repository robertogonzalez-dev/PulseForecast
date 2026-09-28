"""Rolling-origin backtesting: train on everything up to a cutoff, score the next HORIZON days."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pandas as pd

from pulse_forecast import HORIZON, metrics
from pulse_forecast.models import Forecaster


def cutoffs(dates: pd.Series, n_folds: int, step: int = HORIZON) -> list[pd.Timestamp]:
    last = pd.Timestamp(dates.max())
    return [last - pd.Timedelta(days=HORIZON + step * k) for k in reversed(range(n_folds))]


def splits(
    frame: pd.DataFrame, n_folds: int
) -> Iterator[tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]]:
    for cutoff in cutoffs(frame["date"], n_folds):
        train = frame[frame["date"] <= cutoff]
        test = frame[
            (frame["date"] > cutoff) & (frame["date"] <= cutoff + pd.Timedelta(days=HORIZON))
        ]
        yield cutoff, train, test


def backtest(
    frame: pd.DataFrame, model_factories: list[Callable[[], Forecaster]], n_folds: int = 4
) -> pd.DataFrame:
    """Return one row per (model, fold) with every metric, computed on a featurized frame."""
    rows = []
    for cutoff, train, test in splits(frame, n_folds):
        y = test["units"].to_numpy(dtype=float)
        for factory in model_factories:
            model = factory().fit(train)
            yhat = model.predict(test)
            rows.append(
                {"model": model.name, "cutoff": cutoff.date()}
                | {name: fn(y, yhat) for name, fn in metrics.ALL.items()}
            )
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    return results.groupby("model")[list(metrics.ALL)].mean().sort_values("wape")
