from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Workforce Intelligence ML API", version="1.0.0")

# Define the expected input schema
class EmployeeData(BaseModel):
    Age: int
    Tenure: int
    StressLevel: int
    SelfRatedHealth: int
    Sentiment_Score: float

# Load the model on startup
model = None

@app.on_event("startup")
def load_model():
    global model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'rf_model.pkl')
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Warning: Model not found at {model_path}. You may need to train it first.")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Workforce Intelligence ML API"}

@app.post("/predict")
def predict_risk(data: EmployeeData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    
    # Format for Scikit-Learn
    features = pd.DataFrame([{
        "Age": data.Age,
        "Tenure": data.Tenure,
        "StressLevel": data.StressLevel,
        "SelfRatedHealth": data.SelfRatedHealth,
        "Sentiment_Score": data.Sentiment_Score
    }])
    
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    
    return {
        "At_Risk_Prediction": int(prediction),
        "Risk_Probability": float(probability),
        "Status": "High Risk" if prediction == 1 else "Stable"
    }
