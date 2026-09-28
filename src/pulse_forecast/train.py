"""Train entry point: backtest all models, fit the winner on all data, save serving artifacts.

pulse-train                          # synthetic data
pulse-train --duckdb path/to/commercepulse.duckdb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from pulse_forecast import HORIZON
from pulse_forecast.backtest import backtest, summarize
from pulse_forecast.data import generate_sales, load_commercepulse
from pulse_forecast.features import make_features
from pulse_forecast.models import LightGBMForecaster, MovingAverage, SeasonalNaive

HISTORY_DAYS = 60  # enough history to compute every lag/rolling feature at serving time


def _log_mlflow(summary_json: dict[str, dict[str, float]], params: dict[str, object]) -> None:
    try:
        import mlflow
    except ImportError:
        return
    with mlflow.start_run(run_name="pulse-train"):
        mlflow.log_params(params)
        for model, scores in summary_json.items():
            mlflow.log_metrics({f"{model}.{k}": v for k, v in scores.items()})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duckdb", type=Path, help="CommercePulse DuckDB file (default: synthetic)"
    )
    parser.add_argument("--skus", type=int, default=20, help="Synthetic SKUs to generate")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    args = parser.parse_args(argv)

    panel = load_commercepulse(args.duckdb) if args.duckdb else generate_sales(n_skus=args.skus)
    frame = make_features(panel)

    results = backtest(frame, [SeasonalNaive, MovingAverage, LightGBMForecaster], args.folds)
    summary = summarize(results)
    print(f"Backtest: {args.folds} folds x {HORIZON}-day horizon\n")
    print(summary.round(3).to_string())

    model = LightGBMForecaster().fit(frame)
    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out / "model.joblib")
    cutoff = panel["date"].max() - pd.Timedelta(days=HISTORY_DAYS)
    panel[panel["date"] > cutoff].to_csv(args.out / "history.csv", index=False)
    summary_json = summary.round(4).to_dict(orient="index")
    (args.out / "backtest.json").write_text(json.dumps(summary_json, indent=2))
    _log_mlflow(summary_json, {"horizon": HORIZON, "folds": args.folds, "rows": len(panel)})

    print(f"\nTop features:\n{model.feature_importance().head(8).to_string()}")
    print(f"\nSaved model and history to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
