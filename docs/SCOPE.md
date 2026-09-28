# PulseForecast: scope and milestones

## Goal
Show end-to-end applied ML on business data: a forecasting problem framed correctly, evaluated honestly and shipped as a service. Scope is roughly 3 to 4 weekends.

## Definition of done (portfolio-ready)
- [x] Leak-free feature pipeline with a test that proves it
- [x] Rolling-origin backtest against two baselines
- [x] LightGBM global model, FastAPI service, Dockerfile, CI
- [ ] Trained on real CommercePulse data, with a results table in the README
- [ ] One notebook: EDA plus error analysis (which SKUs and days are worst, and why)
- [ ] Deployed through PulseInfra, with the live URL in the README

## Milestones
1. **Data (weekend 1).** Point `load_commercepulse` at the gold layer. Do EDA on sparsity, seasonality and promo frequency.
2. **Modeling (weekend 2).** Tune LightGBM with Optuna, scored by backtest WAPE. Try adding holiday features and a price-elasticity feature.
3. **Evaluation (weekend 3).** Break errors down by category and by volume tier. Add prediction intervals with quantile LightGBM (p10/p50/p90), which matter for inventory decisions.
4. **Serving (weekend 4).** Deploy through PulseInfra. Add a Streamlit tab to CommercePulse that calls the API.

## Stretch goals
- Hierarchical reconciliation (SKU to category to total)
- Model registry in MLflow, with the API loading the "Production" alias
- Drift monitoring: compare live forecasts to actuals weekly
