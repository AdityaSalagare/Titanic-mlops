from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load trained model
model = joblib.load("model/model.pkl")

# Define request body structure
class PassengerInput(BaseModel):
    pclass: int
    sex: int
    age: float
    fare: float

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Titanic Survival Prediction API"}

@app.post("/predict/")
def predict(data: PassengerInput):
    input_data = pd.DataFrame([[data.pclass, data.sex, data.age, data.fare]], columns=["Pclass", "Sex", "Age", "Fare"])
    prediction = model.predict(input_data)[0]
    return {"Survived": int(prediction)}
