from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data_processor import build_preprocessor, load_and_clean_data

RANDOM_STATE = 42
BEST_C = 0.1
BEST_CLASS_WEIGHT = "balanced"
RAW_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "HR-Employee-Attrition.csv"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "attrition_pipeline.joblib"

X, y = load_and_clean_data(
    RAW_DATA_PATH, target_col="Attrition"
)

preprocessor = build_preprocessor()

pipe = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "clf",
            LogisticRegression(
                C=BEST_C,
                class_weight=BEST_CLASS_WEIGHT,
                solver="lbfgs",
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)
pipe.fit(X, y)

joblib.dump(pipe, MODEL_PATH)
