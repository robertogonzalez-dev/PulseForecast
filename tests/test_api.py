from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pulse_forecast import HORIZON
from pulse_forecast.api import _artifacts, app
from pulse_forecast.train import main as train_main


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    out: Path = tmp_path_factory.mktemp("artifacts")
    assert train_main(["--skus", "3", "--folds", "1", "--out", str(out)]) == 0
    mp = pytest.MonkeyPatch()
    mp.setenv("PULSE_ARTIFACTS", str(out))
    _artifacts.cache_clear()
    yield TestClient(app)
    mp.undo()
    _artifacts.cache_clear()


def test_health(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"


def test_skus_and_backtest(client: TestClient) -> None:
    assert client.get("/skus").json() == ["SKU-000", "SKU-001", "SKU-002"]
    assert "lightgbm" in client.get("/backtest").json()


def test_forecast(client: TestClient) -> None:
    resp = client.post("/forecast", json={"sku": "SKU-001", "horizon": 7})
    body = resp.json()
    assert resp.status_code == 200
    assert len(body["forecast"]) == 7
    assert all(p["units"] >= 0 for p in body["forecast"])


def test_promo_lifts_forecast(client: TestClient) -> None:
    base = client.post("/forecast", json={"sku": "SKU-001"}).json()["forecast"]
    day = base[3]["date"]
    promo = client.post("/forecast", json={"sku": "SKU-001", "promo_days": [day]}).json()
    assert len(base) == HORIZON
    assert promo["forecast"][3]["units"] > base[3]["units"]


def test_unknown_sku_and_bad_horizon(client: TestClient) -> None:
    assert client.post("/forecast", json={"sku": "nope"}).status_code == 404
    assert client.post("/forecast", json={"sku": "SKU-001", "horizon": 99}).status_code == 422
