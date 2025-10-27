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
    """Test FastAPI /predict endpoint"""
    # Modify this sample input based on your actual API schema
    sample_input = {
        "age": 30,
        "bmi": 25.5,
        "children": 2,
        "sex_female": 1,
        "sex_male": 0,
        "smoker_no": 1,
        "smoker_yes": 0,
        "region_northeast": 0,
        "region_northwest": 0,
        "region_southeast": 1,
        "region_southwest": 0
    }
    response = client.post("/predict_form", json=sample_input)
    assert response.status_code == 200
    result = response.json()
    assert "predicted_insurance_cost" in result
