import joblib
import pandas as pd
from fastapi import FastAPI
from schemas import EmployeeInput, PredictionOutput

from src.data_processor import clean_features

app = FastAPI()
model = joblib.load("../models/attrition_pipeline.joblib")


@app.post("/predict", response_model=PredictionOutput)
def predict(employee: EmployeeInput):
    df = clean_features(pd.DataFrame([employee.model_dump()]))
    proba = model.predict_proba(df)[:, 1][0]
    return PredictionOutput(prediction=bool(proba >= 0.5), probability=float(proba))
