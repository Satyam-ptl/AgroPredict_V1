"""
Shared fixtures. The key idea: tests never touch real .joblib files or the OpenWeather
API. `registry.load()` is monkeypatched to populate the registry with small, fake,
deterministic stand-ins instead of reading from disk -- so these tests check the API's
own logic (validation, routing, response shape, error handling), not the trained model's
actual accuracy, which is what the notebook's own evaluation cells are for.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.model_registry import registry

FAKE_CROPS = ["rice", "wheat", "maize", "cotton", "sugarcane"]

# rice/wheat share a cluster, maize/cotton share a different one, sugarcane stands alone --
# enough structure to exercise cluster_alternatives() meaningfully.
FAKE_CLUSTER_MAP = {
    "rice": "cluster_0",
    "wheat": "cluster_0",
    "maize": "cluster_1",
    "cotton": "cluster_1",
    "sugarcane": "cluster_2",
}

FAKE_FEATURES = [
    "nitrogen_N_kg_ha", "phosphorus_P_kg_ha", "potassium_K_kg_ha",
    "temperature_C", "humidity_percent", "rainfall_mm",
    "soil_pH", "soil_moisture_percent",
    "npk_sum", "n_p_ratio", "n_k_ratio", "p_k_ratio", "aridity_index",
    "soil_type", "season",
]


class FakeLabelEncoder:
    classes_ = np.array(FAKE_CROPS)


class FakePipeline:
    """predict_proba always returns the same fixed, descending distribution over the
    5 fake crops, in FAKE_CROPS order -- deterministic, so tests can assert on exact
    ranking without depending on real model behavior."""

    def predict_proba(self, X):
        n_rows = len(X)
        # rice=0.40, wheat=0.25, maize=0.15, cotton=0.12, sugarcane=0.08
        row = np.array([0.40, 0.25, 0.15, 0.12, 0.08])
        return np.tile(row, (n_rows, 1))


def _fake_load(self):
    self.pipeline = FakePipeline()
    self.label_encoder = FakeLabelEncoder()
    self.metadata = {
        "features": FAKE_FEATURES,
        "categorical_options": {
            "soil_type": ["loamy", "clayey", "sandy"],
            "season": ["kharif", "rabi"],
        },
        "n_classes": len(FAKE_CROPS),
        "crop_to_cluster": FAKE_CLUSTER_MAP,
        "test_macro_f1": 0.17,
        "test_top3_acc": 0.35,
        "test_top5_acc": 0.45,
        "test_top5_cluster_acc": 0.92,
    }
    self.crop_to_cluster = FAKE_CLUSTER_MAP


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(registry, "load", _fake_load.__get__(registry))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_manual_payload():
    return {
        "nitrogen_N_kg_ha": 90,
        "phosphorus_P_kg_ha": 42,
        "potassium_K_kg_ha": 43,
        "temperature_C": 26.5,
        "humidity_percent": 80,
        "rainfall_mm": 220,
        "soil_pH": 6.5,
        "soil_moisture_percent": 45,
        "soil_type": "loamy",
        "season": "kharif",
    }
