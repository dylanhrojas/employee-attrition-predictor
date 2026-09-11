import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def load_and_split_data(
    filepath, target_col="Attrition", test_size=0.2, random_state=42, stratify=True
):
    """Retrieves clean data (convert and omit certain columns) and then uses
    train_test_split with given arguments for modeling.

    Args:
        filepath: Path to dataset.
        target_col (str, optional): Target vector for y. Defaults to "Attrition".
        test_size (float, optional): Parameter for train_test_split. Defaults to 0.2.
        random_state (int, optional): Parameter for train_test_split. Defaults to 42.
        stratify (bool, optional): Parameter for train_test_split. Defaults to True.

    Returns:
        tuple: Four splits for training and testing.
    """
    X, y = _clean_data(filepath, target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None,
    )

    return X_train, X_test, y_train, y_test


def load_and_clean_data(filepath, target_col="Attrition"):
    """Retrieves clean data (X, y) without splitting, for final model training."""
    return _clean_data(filepath, target_col)


def build_preprocessor():
    """Data preprocessing for dataset for modeling pipeline.

    Returns:
        sklearn.compose.ColumnTransformer: Unfitted ColumnTransfomer for pipelines.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[["Non-Travel", "Travel_Rarely", "Travel_Frequently"]]
                ),
                ["BusinessTravel"],  # Only apply encoder to this feat
            ),
            (
                "numerical",
                StandardScaler(),
                make_column_selector(dtype_include="number"),
            ),
            (
                "nominal",
                OneHotEncoder(handle_unknown="ignore"),
                ["Department", "EducationField", "JobRole", "MaritalStatus"],
            ),
        ],
        remainder="passthrough",
    )

    return preprocessor

def clean_features(df: pd.DataFrame):
    """Returns transformed DataFrame but receiving DataFrame instead of csv_path."""
    clean_df = df.drop(
        columns=[
            "EmployeeCount",
            "StandardHours",
            "Over18",
            "EmployeeNumber"
        ]
    )

    # Binary feats
    clean_df["OverTime"] = clean_df["OverTime"].map({"No": False, "Yes": True})
    clean_df["IsMale"] = clean_df["Gender"].map({"Female": False, "Male": True})

    clean_df = clean_df.drop(columns=["Gender"])

    return clean_df


def _clean_data(filepath, target_col="Attrition"):
    """Private function for reading dataset and ommiting and converting specific
    columns from 01_eda.

    Args:
        filepath: Path to dataset.
        target_col (str, optional): Target vector for y. Defaults to "Attrition".

    Returns:
        tuple: Two vectors X, y for train_test_split process.
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
            target_col,
        ]
    )
    y = df[target_col].map({"No": False, "Yes": True})

    # Binary feats
    X["OverTime"] = X["OverTime"].map({"No": False, "Yes": True})
    X["IsMale"] = X["Gender"].map({"Female": False, "Male": True})

    X = X.drop(columns=["Gender"])

    return X, y
