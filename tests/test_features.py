import numpy as np
import pandas as pd

from pulse_forecast import HORIZON
from pulse_forecast.data import COLUMNS, complete_panel
from pulse_forecast.features import FEATURES, make_features


def test_panel_shape(panel: pd.DataFrame) -> None:
    assert list(panel.columns) == COLUMNS
    assert len(panel) == 4 * 200
    assert (panel["units"] >= 0).all()


def test_all_features_present(frame: pd.DataFrame) -> None:
    assert set(FEATURES) <= set(frame.columns)


def test_no_leakage_within_horizon(panel: pd.DataFrame) -> None:
    """Scrambling demand after a cutoff must not change features for the next HORIZON days."""
    cutoff = panel["date"].min() + pd.Timedelta(days=120)
    tampered = panel.copy()
    after = tampered["date"] > cutoff
    tampered.loc[after, "units"] = np.random.default_rng(0).integers(0, 10_000, after.sum())

    window = (panel["date"] > cutoff) & (panel["date"] <= cutoff + pd.Timedelta(days=HORIZON))
    demand_features = [f for f in FEATURES if f.startswith(("lag_", "roll_"))]
    pd.testing.assert_frame_equal(
        make_features(panel).loc[window, demand_features],
        make_features(tampered).loc[window, demand_features],
    )


def test_lag_alignment(frame: pd.DataFrame) -> None:
    sku = frame[frame["sku"] == "SKU-000"].sort_values("date").reset_index(drop=True)
    assert sku.loc[HORIZON + 5, f"lag_{HORIZON}"] == sku.loc[5, "units"]


def test_complete_panel_fills_gaps() -> None:
    sparse = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-03"],
            "sku": ["A", "A"],
            "category": ["x", "x"],
            "price": [9.99, 9.99],
            "promo": [0, 1],
            "units": [3, 5],
        }
    )
    full = complete_panel(sparse)
    assert full["units"].tolist() == [3, 0, 5]
    assert full["category"].tolist() == ["x", "x", "x"]
