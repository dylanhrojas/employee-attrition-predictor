import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data_processor import build_preprocessor, load_and_clean_data

RANDOM_STATE = 42
BEST_C = 0.1
BEST_CLASS_WEIGHT = "balanced"

X, y = load_and_clean_data(
    "../data/raw/HR-Employee-Attrition.csv", target_col="Attrition"
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

joblib.dump(pipe, "../models/attrition_pipeline.joblib")
