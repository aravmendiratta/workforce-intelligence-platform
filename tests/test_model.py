from fastapi.testclient import TestClient
from api.main import app, model
import os
import pytest

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Workforce Intelligence ML API"}

def test_predict_endpoint_high_risk():
    # Skip if the model failed to load (e.g. scikit-learn version mismatch in CI)
    if model is None:
        pytest.skip("ML model not loaded — skipping prediction test")

    payload = {
        "Age": 55,
        "Tenure": 20,
        "StressLevel": 9,
        "SelfRatedHealth": 1,
        "Sentiment_Score": -0.8
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "At_Risk_Prediction" in data
    assert data["At_Risk_Prediction"] in [0, 1]
