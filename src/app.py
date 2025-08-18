from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path

# Load the trained model
print("🚀 Loading model...")
model = joblib.load("models/logistic_regression_model.pkl")
print("✅ Model loaded successfully!")

# Create FastAPI app
app = FastAPI(title="Titanic Survival Prediction API")

# Enable CORS for local testing and frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins for testing; later restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request body
class Passenger(BaseModel):
    Pclass: int
    Sex: int  # 0 = male, 1 = female
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: int  # 0 = C, 1 = Q, 2 = S

@app.get("/")
def serve_frontend():
    index_path = Path(__file__).parent / "static" / "index.html"
    return FileResponse(index_path)

@app.post("/predict")
def predict_survival(passenger: Passenger):
    print(f"Received request: {passenger}")

    # Convert input into numpy array
    data = np.array([[passenger.Pclass, passenger.Sex, passenger.Age, passenger.SibSp,
                      passenger.Parch, passenger.Fare, passenger.Embarked]])
    
    # Predict
    prediction = model.predict(data)[0]
    survival = "Survived" if prediction == 1 else "Did Not Survive"
    
    print(f"Prediction: {prediction} ({survival})")
    return {"prediction": int(prediction), "survival_status": survival}
