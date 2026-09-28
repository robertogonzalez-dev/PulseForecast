from functools import partial

import numpy as np
import pandas as pd
import pytest

from pulse_forecast import HORIZON, metrics
from pulse_forecast.backtest import backtest, splits, summarize
from pulse_forecast.models import LightGBMForecaster, MovingAverage, SeasonalNaive


def test_metrics() -> None:
    y, yhat = np.array([10.0, 0.0, 10.0]), np.array([8.0, 1.0, 12.0])
    assert metrics.mae(y, yhat) == pytest.approx(5 / 3)
    assert metrics.wape(y, yhat) == pytest.approx(5 / 20)
    assert metrics.bias(y, yhat) == pytest.approx(1 / 20)
    assert np.isnan(metrics.wape(np.zeros(2), np.ones(2)))


def test_splits_do_not_overlap(frame: pd.DataFrame) -> None:
    folds = list(splits(frame, n_folds=3))
    assert len(folds) == 3
    for cutoff, train, test in folds:
        assert train["date"].max() == cutoff
        assert test["date"].min() > cutoff
        assert test["date"].nunique() == HORIZON


def test_lightgbm_beats_naive(frame: pd.DataFrame) -> None:
    fast_lgbm = partial(LightGBMForecaster, n_estimators=100)
    results = backtest(frame, [SeasonalNaive, MovingAverage, fast_lgbm], n_folds=2)
    summary = summarize(results)
    assert set(summary.index) == {"seasonal_naive", "moving_average_28", "lightgbm"}
    assert summary.loc["lightgbm", "wape"] < summary.loc["seasonal_naive", "wape"]


def test_predictions_non_negative(frame: pd.DataFrame) -> None:
    model = LightGBMForecaster(n_estimators=20).fit(frame)
    assert (model.predict(frame) >= 0).all()
    assert model.feature_importance().index[0]
