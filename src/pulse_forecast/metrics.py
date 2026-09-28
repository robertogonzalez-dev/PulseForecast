"""Forecast accuracy metrics."""

from __future__ import annotations

import numpy as np


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def wape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Weighted absolute percentage error: sum |error| / sum actuals. Robust to zero-demand days."""
    denom = np.sum(np.abs(y))
    return float(np.sum(np.abs(y - yhat)) / denom) if denom else float("nan")


def bias(y: np.ndarray, yhat: np.ndarray) -> float:
    """Positive means over-forecasting, as a share of total actual demand."""
    denom = np.sum(y)
    return float(np.sum(yhat - y) / denom) if denom else float("nan")


ALL = {"mae": mae, "rmse": rmse, "wape": wape, "bias": bias}
