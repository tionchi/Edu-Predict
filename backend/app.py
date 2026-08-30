from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from typing import Literal

from fastapi.middleware.cors import CORSMiddleware
#Create API
app = FastAPI(title="EduPredict API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)

#load trained models
models = {
    "linear_regression": joblib.load("backend/models/linear_regression_model.pkl"),
    "random_forest": joblib.load("backend/models/random_forest_model.pkl"),
    "gradient_boosting": joblib.load("backend/models/gradient_boosting_model.pkl"),
}

#Input data structure
class Student(BaseModel):
    model: Literal["linear_regression", "random_forest", "gradient_boosting"] = "gradient_boosting"
    Hours_Studied: int
    Attendance: int
    Parental_Involvement: str
    Access_to_Resources: str
    Extracurricular_Activities: str
    Sleep_Hours: int
    Previous_Scores: int
    Motivation_Level: str
    Internet_Access: str
    Tutoring_Sessions: int
    Family_Income: str
    Teacher_Quality: str
    Physical_Activity: int
    Learning_Disabilities: str
    Parental_Education_Level: str
    Distance_from_Home: str
    School_Type: str
    Peer_Influence: str
    Gender: str

#Testing endpoi8nt
@app.get("/")
def home():
    return {"message": "EduPredict API is running"}

#prediction endpoint
@app.post("/predict")
def predict(student: Student):
    #convert request to DataFrame
    data=pd.DataFrame([student.model_dump(exclude={"model"})])
    selected_model = models[student.model]
    #make prediction
    prediction=selected_model.predict(data)[0]

    return {
        "predicted_score": round(float(prediction), 2),
        "model": student.model,
    }
