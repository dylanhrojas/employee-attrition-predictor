import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def load_and_split_data(
    filepath,
    target_col="Attrition",
    test_size=0.2,
    random_state=42,
    stratify=True,
):
    """Returns preprocessed data for modeling.

    Args:
        filepath (str): Path to dataset to transform.
        target_col (str, optional): Target to use as vector. Defaults to "Attrition".
        test_size (float, optional): Arg for train_test_split. Defaults to 0.2.
        random_state (int, optional): Arg for train_test_split. Defaults to 42.
        stratify (bool, optional): Arg for train_test_split. Defaults to True.

    Returns:
        dict: Training and testing data, fitted ColumnTransformer and feats from preprocessor.
         Keys: X_train, X_test, y_train, y_test, feats, preprocessor.
    """
    df = pd.read_csv(filepath)

    # Drop EmployeeCount, StandardHours and Over18 since are constant
    # Drop EmployeeNumber since it's an ID
    # Drop target (Attrition) and store it
    X = df.drop(
        columns=[
            "EmployeeCount",
            "StandardHours",
            "Over18",
            "EmployeeNumber",
            "Attrition",
        ]
    )
    y = df[target_col].map({"No": 0, "Yes": 1})

    # Nominal (binary) feats
    X["OverTime"] = X["OverTime"].map({"No": 0, "Yes": 1})
    X["Gender"] = X["Gender"].map({"Female": 0, "Male": 1})

    # Discard these feats for numerical since they don't need scaling
    binary_feats = ["OverTime", "Gender"]

    numerical_feats = [
        f for f in X.select_dtypes(include="number").columns if f not in binary_feats
    ]

    nominal_feats = [
        "Department",
        "EducationField",
        "JobRole",
        "MaritalStatus",
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "ordinal",
                OrdinalEncoder(
                    categories=["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
                ),
                "BusinessTravel",  # Only apply encoder to this feat
            ),
            ("numerical", StandardScaler(), numerical_feats),
            ("nominal", OneHotEncoder(handle_unknown="ignore"), nominal_feats),
        ],
        remainder="passthrough",
    )

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    feats = preprocessor.get_feature_names_out()

    return {
        "X_train": X_train_transformed,
        "X_test": X_test_transformed,
        "y_train": y_train,
        "y_test": y_test,
        "feats": feats,
        "preprocessor": preprocessor,
    }
