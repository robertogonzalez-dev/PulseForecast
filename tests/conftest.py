import pandas as pd
import pytest

from pulse_forecast.data import generate_sales
from pulse_forecast.features import make_features


@pytest.fixture(scope="session")
def panel() -> pd.DataFrame:
    return generate_sales(n_skus=4, days=200, seed=7)


@pytest.fixture(scope="session")
def frame(panel: pd.DataFrame) -> pd.DataFrame:
    return make_features(panel)
