# PulseForecast

[![CI](https://github.com/robertogonzalez-dev/PulseForecast/actions/workflows/ci.yml/badge.svg)](https://github.com/robertogonzalez-dev/PulseForecast/actions/workflows/ci.yml)

**SKU-level demand forecasting for e-commerce.** It's the predictive layer on top of [CommercePulse](https://github.com/robertogonzalez-dev/CommercePulse): 14-day-ahead unit forecasts per product, honestly evaluated against baselines and served through a REST API.

> **Business question:** How many units of each SKU will sell each day over the next two weeks, and how much does a planned promotion change that?

## Highlights

- **Leak-free by construction.** Every demand feature for day *t* uses only data from *t − 14* or earlier, and a test proves it by scrambling future demand and checking that no feature changes.
- **Honest evaluation.** A rolling-origin backtest compares LightGBM against seasonal-naive and moving-average baselines on WAPE, MAE, RMSE and bias.
- **One global model.** A single LightGBM model with a Tweedie loss (suited to count demand) learns across all SKUs.
- **What-if promos.** Send planned promo dates to the API and see the forecast lift.
- **Production shape.** The project includes a CLI trainer, versioned artifacts, optional MLflow tracking, a FastAPI service and a Docker image. [PulseInfra](https://github.com/robertogonzalez-dev/PulseInfra) deploys it to AWS.

## Architecture

```text
CommercePulse DuckDB ─┐                       ┌─> backtest.json (model comparison)
   (or synthetic)     ├─> features ─> models ─┼─> model.joblib + history.csv
                      │   (lags ≥ 14d,        └─> MLflow run (optional)
                      │    rolling, calendar,
                      │    price, promo)
                      └──────────────────────────> FastAPI: /forecast /skus /backtest /health
```

## Quickstart

```bash
python -m venv .venv && .venv\Scripts\activate      # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
pulse-train                                          # synthetic data -> artifacts/
uvicorn pulse_forecast.api:app --reload              # http://localhost:8000/docs
```

```bash
curl -X POST localhost:8000/forecast -H "Content-Type: application/json" \
     -d '{"sku": "SKU-001", "horizon": 7, "promo_days": ["2025-12-31"]}'
```

Use real CommercePulse data: `pip install -e ".[commercepulse]"`, then run `pulse-train --duckdb ../CommercePulse/data/commercepulse.duckdb`. Edit the SQL in `data.load_commercepulse` to match your gold models.

Docker: `docker build -t pulse-forecast . && docker run -p 8000:8000 pulse-forecast`

## Results

The table is filled in by running `pulse-train`. Paste the output here once real CommercePulse data is wired in.

| Model | WAPE | MAE | Bias |
|---|---|---|---|
| lightgbm | – | – | – |
| seasonal_naive | – | – | – |
| moving_average_28 | – | – | – |

## Project layout

```text
src/pulse_forecast/
  data.py       synthetic generator + CommercePulse DuckDB loader
  features.py   leak-free lag / rolling / calendar features
  models.py     SeasonalNaive, MovingAverage, LightGBMForecaster
  backtest.py   rolling-origin evaluation
  metrics.py    WAPE, MAE, RMSE, bias
  train.py      `pulse-train` CLI
  api.py        FastAPI service
```

See [docs/SCOPE.md](docs/SCOPE.md) for milestones and the roadmap.
