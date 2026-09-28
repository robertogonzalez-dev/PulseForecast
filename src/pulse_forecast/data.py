"""Data sources: a realistic synthetic generator and a CommercePulse (DuckDB) loader.

Every loader returns a daily panel with columns:
    date (datetime64), sku (str), category (str), price (float), promo (0/1), units (int)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["date", "sku", "category", "price", "promo", "units"]
CATEGORIES = ["electronics", "home", "apparel", "beauty", "sports"]

# Assumes a gold-layer order-items fact; adjust to match your CommercePulse models.
COMMERCEPULSE_QUERY = """
    select
        cast(order_date as date)                        as date,
        product_id                                      as sku,
        coalesce(category, 'unknown')                   as category,
        avg(unit_price)                                 as price,
        max(case when discount > 0 then 1 else 0 end)   as promo,
        sum(quantity)                                   as units
    from gold.fct_order_items
    group by 1, 2, 3
"""


def generate_sales(
    n_skus: int = 20, days: int = 730, start: str = "2024-01-01", seed: int = 42
) -> pd.DataFrame:
    """Synthetic SKU-day demand: trend, weekly + yearly seasonality, promos, price effects."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    t = np.arange(days)
    frames = []
    for i in range(n_skus):
        base = rng.uniform(5, 60)
        trend = 1 + rng.uniform(-0.2, 0.5) * t / days
        weekly = 1 + rng.uniform(0.1, 0.4) * np.isin(dates.dayofweek, [4, 5])  # Fri/Sat bump
        holiday_peak = np.exp(-(((dates.dayofyear - 340) / 20) ** 2))  # early-December peak
        yearly = 1 + rng.uniform(0.1, 0.5) * holiday_peak
        promo = (rng.random(days) < 0.08).astype(int)
        list_price = rng.uniform(10, 200)
        price = np.round(list_price * np.where(promo == 1, 0.8, 1.0), 2)
        lam = base * trend * weekly * yearly * (1 + 0.6 * promo)
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "sku": f"SKU-{i:03d}",
                    "category": CATEGORIES[i % len(CATEGORIES)],
                    "price": price,
                    "promo": promo,
                    "units": rng.poisson(lam),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def load_commercepulse(db_path: Path, query: str | None = None) -> pd.DataFrame:
    """Load daily SKU demand from a CommercePulse DuckDB warehouse.

    Requires the `commercepulse` extra (duckdb).
    """
    import duckdb

    with duckdb.connect(str(db_path), read_only=True) as con:
        df = con.execute(query or COMMERCEPULSE_QUERY).df()
    return complete_panel(df)


def complete_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing SKU-days with zero demand so lags line up on a regular daily grid."""
    df = df.assign(date=pd.to_datetime(df["date"]))
    full_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    idx = pd.MultiIndex.from_product([df["sku"].unique(), full_dates], names=["sku", "date"])
    out = df.set_index(["sku", "date"]).reindex(idx).reset_index()
    out["units"] = out["units"].fillna(0).astype(int)
    out["promo"] = out["promo"].fillna(0).astype(int)
    out["category"] = out.groupby("sku")["category"].transform(lambda s: s.ffill().bfill())
    out["price"] = out.groupby("sku")["price"].transform(lambda s: s.ffill().bfill())
    return out[COLUMNS]
