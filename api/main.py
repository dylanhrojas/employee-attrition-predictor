from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

from schemas.employee import EmployeeInput, PredictionOutput
from src.data_processor import clean_features

app = FastAPI()

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "attrition_pipeline.joblib"
model = joblib.load(MODEL_PATH)


@app.post("/predict", response_model=PredictionOutput)
def predict(employee: EmployeeInput):
    df = clean_features(pd.DataFrame([employee.model_dump()]))
    proba = model.predict_proba(df)[:, 1][0]
    return PredictionOutput(prediction=bool(proba >= 0.5), probability=float(proba))
