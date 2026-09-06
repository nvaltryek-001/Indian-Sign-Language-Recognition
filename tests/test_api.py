import numpy as np
from fastapi.testclient import TestClient

import app.api as api


class FakeRuntime:
    model = object()

    def predict(self, sequence):
        assert np.asarray(sequence).shape == (45, 258)
        return {
            "prediction": "1. loud",
            "confidence": 0.9,
            "top_predictions": [{"index": 1, "label": "1. loud", "confidence": 0.9}],
            "model_version": "test",
        }


def test_health_reports_loaded_runtime(monkeypatch):
    monkeypatch.setattr(api, "get_runtime", lambda: FakeRuntime())
    with TestClient(api.app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_rejects_wrong_row_width():
    with TestClient(api.app) as client:
        response = client.post("/predict", json={"sequence": [[0.0] * 257] * 45})
    assert response.status_code in (422, 500)


def test_predict_returns_product_contract(monkeypatch):
    monkeypatch.setattr(api, "get_runtime", lambda: FakeRuntime())
    sequence = {"sequence": [[0.0] * 258] * 45}
    with TestClient(api.app) as client:
        response = client.post("/predict/sequence", json=sequence)
    assert response.status_code == 200
    assert response.json()["prediction"] == "1. loud"
    assert response.json()["model_version"] == "test"
    assert response.json()["quality"]["validated"] is True


def test_api_allows_configured_cors_origin():
    with TestClient(api.app) as client:
        response = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"


def test_hand_health_reports_model_and_reference_contract(monkeypatch):
    class FakeHandRuntime:
        model = object()
        class_names = [str(i) for i in range(50)]

    class FakeMatcher:
        references = {str(i): [np.zeros((45, 126), dtype=np.float32)] for i in range(50)}

    monkeypatch.setattr(api, "get_hand_runtime", lambda: FakeHandRuntime())
    monkeypatch.setattr(api, "get_dtw_matcher", lambda: FakeMatcher())
    with TestClient(api.app) as client:
        response = client.get("/health/hand")
    assert response.status_code == 200
    payload = response.json()
    assert payload["model_loaded"] is True
    assert payload["class_count"] == 50
    assert payload["reference_database_available"] is True
    assert payload["reference_classes"] == 50