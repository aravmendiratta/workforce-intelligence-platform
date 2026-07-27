from fastapi.testclient import TestClient
from api.main import app
import os
import pytest

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Workforce Intelligence ML API"}

def test_predict_endpoint_high_risk():
    # Only run prediction tests if the model is trained and available
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'rf_model.pkl')
    if os.path.exists(model_path):
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
