"""Leak-free feature engineering.

Every demand-derived feature for date t only uses units from dates <= t - HORIZON, so one trained
model can forecast any day 1..HORIZON past the last observed date. Price and promo are treated as
known in advance (planned by merchandising).
"""

from __future__ import annotations

import pandas as pd

from pulse_forecast import HORIZON

LAGS = [HORIZON, HORIZON + 7, HORIZON + 14, HORIZON + 21]
WINDOWS = [7, 28]
CALENDAR = ["dow", "dom", "month", "weekofyear", "is_weekend"]
FEATURES = (
    [f"lag_{lag}" for lag in LAGS]
    + [f"roll_mean_{w}" for w in WINDOWS]
    + ["roll_std_28"]
    + CALENDAR
    + ["price", "promo", "sku_code", "category_code"]
)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add model features to a daily panel (see data.COLUMNS). Rows keep their original order."""
    out = df.sort_values(["sku", "date"]).copy()
    by_sku = out.groupby("sku", sort=False)["units"]

    for lag in LAGS:
        out[f"lag_{lag}"] = by_sku.shift(lag)

    shifted = by_sku.shift(HORIZON)
    shifted_by_sku = shifted.groupby(out["sku"], sort=False)
    for w in WINDOWS:
        out[f"roll_mean_{w}"] = shifted_by_sku.transform(lambda s, w=w: s.rolling(w).mean())
    out["roll_std_28"] = shifted_by_sku.transform(lambda s: s.rolling(28).std())

    out["dow"] = out["date"].dt.dayofweek
    out["dom"] = out["date"].dt.day
    out["month"] = out["date"].dt.month
    out["weekofyear"] = out["date"].dt.isocalendar().week.astype(int)
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["sku_code"] = out["sku"].astype("category").cat.codes
    out["category_code"] = out["category"].astype("category").cat.codes
    return out.sort_index()
