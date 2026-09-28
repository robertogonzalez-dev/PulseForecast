"""Forecasters sharing a tiny fit/predict interface over feature frames."""

from __future__ import annotations

from typing import Protocol

import lightgbm as lgb
import numpy as np
import pandas as pd

from pulse_forecast import HORIZON
from pulse_forecast.features import FEATURES


class Forecaster(Protocol):
    name: str

    def fit(self, train: pd.DataFrame) -> Forecaster: ...
    def predict(self, frame: pd.DataFrame) -> np.ndarray: ...


class SeasonalNaive:
    """Predict the value from the same weekday HORIZON days earlier (HORIZON is a multiple of 7)."""

    name = "seasonal_naive"

    def fit(self, train: pd.DataFrame) -> SeasonalNaive:
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame[f"lag_{HORIZON}"].fillna(0).to_numpy(dtype=float)


class MovingAverage:
    """Predict the trailing 28-day mean, as of HORIZON days ago."""

    name = "moving_average_28"

    def fit(self, train: pd.DataFrame) -> MovingAverage:
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["roll_mean_28"].fillna(0).to_numpy(dtype=float)


class LightGBMForecaster:
    """One global gradient-boosted model across all SKUs, with a Tweedie loss for count demand."""

    name = "lightgbm"

    def __init__(self, **params: object) -> None:
        defaults: dict[str, object] = {
            "objective": "tweedie",
            "n_estimators": 400,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "verbose": -1,
        }
        self.model = lgb.LGBMRegressor(**{**defaults, **params})

    def fit(self, train: pd.DataFrame) -> LightGBMForecaster:
        train = train.dropna(subset=["units"])
        self.model.fit(train[FEATURES], train["units"])
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model.predict(frame[FEATURES]), 0, None)

    def feature_importance(self) -> pd.Series:
        return pd.Series(self.model.feature_importances_, index=FEATURES).sort_values(
            ascending=False
        )
