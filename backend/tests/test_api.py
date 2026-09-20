"""
Run with: pytest (from the backend/ directory)

Covers: does the API start and report itself ready, does /predict return a well-formed
shortlist, does validation actually reject bad input (not just accept everything), and
does /predict/auto fail cleanly (not with a raw traceback) when misconfigured or missing
required location info.
"""
from unittest.mock import AsyncMock

import pytest

from app.weather import WeatherReading


# ---------------------------------------------------------------------------
# Health / options / metadata endpoints
# ---------------------------------------------------------------------------

def test_health_returns_ok_and_metrics(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["n_classes"] == 5
    assert body["test_top5_cluster_acc"] == 0.92


def test_options_returns_categorical_choices(client):
    resp = client.get("/options")
    assert resp.status_code == 200
    body = resp.json()
    assert "loamy" in body["soil_type"]
    assert "kharif" in body["season"]


def test_crop_clusters_returns_full_map(client):
    resp = client.get("/crop-clusters")
    assert resp.status_code == 200
    body = resp.json()
    assert body["rice"] == body["wheat"]  # same cluster in the fake data
    assert body["rice"] != body["sugarcane"]  # different cluster


# ---------------------------------------------------------------------------
# POST /predict -- happy path
# ---------------------------------------------------------------------------

def test_predict_returns_five_ranked_suggestions(client, valid_manual_payload):
    resp = client.post("/predict", json=valid_manual_payload)
    assert resp.status_code == 200
    body = resp.json()

    assert len(body["shortlist"]) == 5
    # FakePipeline always ranks rice highest -- confirms sorting logic, not just "some list"
    assert body["shortlist"][0]["crop"] == "rice"
    assert body["shortlist"][0]["probability"] == pytest.approx(0.40, abs=1e-4)

    probs = [item["probability"] for item in body["shortlist"]]
    assert probs == sorted(probs, reverse=True), "shortlist must be sorted by probability, descending"
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_predict_includes_cluster_alternatives(client, valid_manual_payload):
    resp = client.post("/predict", json=valid_manual_payload)
    body = resp.json()

    rice_entry = next(item for item in body["shortlist"] if item["crop"] == "rice")
    # wheat is rice's cluster-mate in the fake map, and wheat is also in the top-5,
    # so it should NOT be duplicated into rice's alternatives list (already in the shortlist).
    assert "wheat" not in rice_entry["cluster_alternatives"]


def test_predict_weather_source_is_absent_for_manual(client, valid_manual_payload):
    resp = client.post("/predict", json=valid_manual_payload)
    assert resp.json()["weather_source"] is None


# ---------------------------------------------------------------------------
# POST /predict -- validation failures
# ---------------------------------------------------------------------------

def test_predict_rejects_unknown_soil_type(client, valid_manual_payload):
    valid_manual_payload["soil_type"] = "moon_dust"
    resp = client.post("/predict", json=valid_manual_payload)
    assert resp.status_code == 422
    assert "soil_type" in resp.text


def test_predict_rejects_unknown_season(client, valid_manual_payload):
    valid_manual_payload["season"] = "monsoon_plus"
    resp = client.post("/predict", json=valid_manual_payload)
    assert resp.status_code == 422


def test_predict_rejects_missing_required_field(client, valid_manual_payload):
    del valid_manual_payload["rainfall_mm"]
    resp = client.post("/predict", json=valid_manual_payload)
    assert resp.status_code == 422


def test_predict_rejects_out_of_range_ph(client, valid_manual_payload):
    valid_manual_payload["soil_pH"] = 25  # ph is capped at 14 in the schema
    resp = client.post("/predict", json=valid_manual_payload)
    assert resp.status_code == 422


def test_predict_rejects_wrong_type(client, valid_manual_payload):
    valid_manual_payload["nitrogen_N_kg_ha"] = "a lot"
    resp = client.post("/predict", json=valid_manual_payload)
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /predict/auto
# ---------------------------------------------------------------------------

def _auto_payload(**overrides):
    payload = {
        "nitrogen_N_kg_ha": 90,
        "phosphorus_P_kg_ha": 42,
        "potassium_K_kg_ha": 43,
        "soil_pH": 6.5,
        "soil_moisture_percent": 45,
        "soil_type": "loamy",
        "season": "kharif",
    }
    payload.update(overrides)
    return payload


def test_predict_auto_without_city_or_coords_returns_422(client, monkeypatch):
    # No mocking of resolve_weather needed here -- the real function itself rejects
    # this input before ever trying to reach OpenWeather.
    resp = client.post("/predict/auto", json=_auto_payload())
    assert resp.status_code == 422
    assert "city" in resp.text or "latitude" in resp.text


def test_predict_auto_without_api_key_returns_clear_500(client, monkeypatch):
    monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)
    resp = client.post("/predict/auto", json=_auto_payload(city="Nagpur"))
    assert resp.status_code == 500
    assert "OPENWEATHER_API_KEY" in resp.json()["detail"]


def test_predict_auto_happy_path_with_mocked_weather(client, monkeypatch):
    fake_weather = WeatherReading(
        temperature_C=27.0, humidity_percent=70.0, rainfall_mm=150.0, source="openweather:Nagpur",
    )
    monkeypatch.setattr("app.main.resolve_weather", AsyncMock(return_value=fake_weather))

    resp = client.post("/predict/auto", json=_auto_payload(city="Nagpur"))
    assert resp.status_code == 200
    body = resp.json()
    assert body["weather_source"] == "openweather:Nagpur"
    assert len(body["shortlist"]) == 5


def test_predict_auto_propagates_city_not_found(client, monkeypatch):
    from fastapi import HTTPException

    async def raise_not_found(*args, **kwargs):
        raise HTTPException(status_code=404, detail="City not found by OpenWeather: 'Atlantis'")

    monkeypatch.setattr("app.main.resolve_weather", raise_not_found)

    resp = client.post("/predict/auto", json=_auto_payload(city="Atlantis"))
    assert resp.status_code == 404
