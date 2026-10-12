import logging
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from schemas.employee import EmployeeInput, PredictionOutput
from src.config import THRESHOLD
from src.data_processor import clean_features

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

app = FastAPI()

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "attrition_pipeline.joblib"

if not MODEL_PATH.exists():
    msg = f"Model not found at {MODEL_PATH}. Run `python src/model_trainer.py` to train it"
    logger.error(msg)
    raise FileNotFoundError(msg)

model = joblib.load(MODEL_PATH)


@app.post("/predict", response_model=PredictionOutput)
def predict(employee: EmployeeInput):
    try:
        df = clean_features(pd.DataFrame([employee.model_dump()]))
        proba = model.predict_proba(df)[:, 1][0]
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Internal prediction error")

    decision = bool(proba >= THRESHOLD)

    msg = f"Prediction prob={float(proba):.2f} threshold={THRESHOLD} decision={decision}"

    logger.info(msg)
    return PredictionOutput(prediction=bool(proba >= THRESHOLD), probability=float(proba))