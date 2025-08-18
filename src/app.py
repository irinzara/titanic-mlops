from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model
print("🚀 Loading model...")
model = joblib.load("models/logistic_regression_model.pkl")
print("✅ Model loaded successfully!")

# Create FastAPI app
app = FastAPI(title="Titanic Survival Prediction API")

# Define request body format
class Passenger(BaseModel):
    Pclass: int
    Sex: int  # 0 = female, 1 = male
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: int  # 0 = C, 1 = Q, 2 = S

@app.get("/")
def home():
    print("Root endpoint accessed")  # Logs every time someone hits /
    return {"message": "Titanic Survival Prediction API is running!"}

@app.post("/predict")
def predict_survival(passenger: Passenger):
    print(f"Received request: {passenger}")  # Logs incoming data

    # Convert input into numpy array
    data = np.array([[passenger.Pclass, passenger.Sex, passenger.Age, passenger.SibSp,
                      passenger.Parch, passenger.Fare, passenger.Embarked]])
    
    # Predict
    prediction = model.predict(data)[0]
    survival = "Survived" if prediction == 1 else "Did Not Survive"
    
    print(f"Prediction: {prediction} ({survival})")  # Logs prediction
    return {"prediction": int(prediction), "survival_status": survival}
