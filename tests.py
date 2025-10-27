import pytest
import os
import pickle
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_model_file_exists():
    """Check if best_model.pkl exists"""
    assert os.path.exists("best_model.pkl"), "best_model.pkl not found!"

def test_model_loads():
    """Try loading the pickle model"""
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    assert model is not None, "Model not loaded properly!"

def test_prediction_endpoint():
    response = client.post(
        "/predict_form",
        data={
            "age": 25,
            "sex": "male",
            "height": 175,
            "weight": 70,
            "children": 2,
            "smoker": "no",
            "region": "northwest"
        }
    )
    assert response.status_code == 200

    


if __name__ == "__main__":
    test_model_file_exists()
    test_model_loads()  
    test_prediction_endpoint()