import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    json_data = response.json()
    assert "status" in json_data
    assert json_data["dataset"] == "NASA C-MAPSS FD001"

def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_api_predict_single():
    sample_payload = {
        "engine_id": 1,
        "cycle": 100,
        "setting_1": 0.0023,
        "setting_2": -0.0004,
        "setting_3": 100.0,
        "sensor_1": 518.67,
        "sensor_2": 642.25,
        "sensor_3": 1589.70,
        "sensor_4": 1400.60,
        "sensor_5": 14.62,
        "sensor_6": 21.61,
        "sensor_7": 554.36,
        "sensor_8": 2388.05,
        "sensor_9": 9046.19,
        "sensor_10": 1.30,
        "sensor_11": 47.47,
        "sensor_12": 521.66,
        "sensor_13": 2388.02,
        "sensor_14": 8138.62,
        "sensor_15": 8.4195,
        "sensor_16": 0.03,
        "sensor_17": 392,
        "sensor_18": 2388,
        "sensor_19": 100.0,
        "sensor_20": 39.06,
        "sensor_21": 23.4190
    }
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_rul_cycles" in data
    assert "health_status" in data
    assert "estimated_failure_cycle" in data

def test_api_predict_sequence():
    cycle_1 = {
        "engine_id": 1,
        "cycle": 50,
        "setting_1": 0.0,
        "setting_2": 0.0,
        "setting_3": 100.0,
        "sensor_2": 642.50,
        "sensor_3": 1589.00,
        "sensor_4": 1406.00,
        "sensor_7": 553.50,
        "sensor_11": 47.45,
        "sensor_12": 521.60,
        "sensor_15": 8.43
    }
    cycle_2 = {**cycle_1, "cycle": 51}
    response = client.post("/predict", json=[cycle_1, cycle_2])
    assert response.status_code == 200
    assert response.json()["current_cycle"] == 51
