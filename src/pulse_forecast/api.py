"""FastAPI service: `uvicorn pulse_forecast.api:app` (reads artifacts from $PULSE_ARTIFACTS)."""

from __future__ import annotations

import json
import os
from datetime import date
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pulse_forecast import HORIZON, __version__
from pulse_forecast.features import make_features
from pulse_forecast.models import LightGBMForecaster

app = FastAPI(title="PulseForecast", version=__version__)


class ForecastRequest(BaseModel):
    sku: str
    horizon: int = Field(default=HORIZON, ge=1, le=HORIZON)
    promo_days: list[date] = Field(default_factory=list, description="Planned promo dates")


class ForecastPoint(BaseModel):
    date: date
    units: float


class ForecastResponse(BaseModel):
    sku: str
    model: str
    forecast: list[ForecastPoint]


@lru_cache
def _artifacts() -> tuple[LightGBMForecaster, pd.DataFrame, Path]:
    root = Path(os.environ.get("PULSE_ARTIFACTS", "artifacts"))
    if not (root / "model.joblib").exists():
        raise HTTPException(503, f"No trained model in {root}/. Run `pulse-train` first.")
    history = pd.read_csv(root / "history.csv", parse_dates=["date"])
    return joblib.load(root / "model.joblib"), history, root


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/skus")
def skus() -> list[str]:
    return sorted(_artifacts()[1]["sku"].unique())


@app.get("/backtest")
def backtest_report() -> dict[str, dict[str, float]]:
    return json.loads((_artifacts()[2] / "backtest.json").read_text())


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest) -> ForecastResponse:
    model, history, _ = _artifacts()
    sku_hist = history[history["sku"] == req.sku]
    if sku_hist.empty:
        raise HTTPException(404, f"Unknown SKU '{req.sku}'")

    last = sku_hist.iloc[-1]
    future_dates = pd.date_range(last["date"] + pd.Timedelta(days=1), periods=req.horizon)
    promos = {pd.Timestamp(d) for d in req.promo_days}
    future = pd.DataFrame(
        {
            "date": future_dates,
            "sku": req.sku,
            "category": last["category"],
            "price": last["price"],
            "promo": [int(d in promos) for d in future_dates],
            "units": float("nan"),
        }
    )
    frame = make_features(pd.concat([history, future], ignore_index=True))
    rows = frame[(frame["sku"] == req.sku) & (frame["date"].isin(future_dates))]
    preds = model.predict(rows)
    return ForecastResponse(
        sku=req.sku,
        model=model.name,
        forecast=[
            ForecastPoint(date=d.date(), units=round(float(p), 2))
            for d, p in zip(rows["date"], preds, strict=True)
        ],
    )
