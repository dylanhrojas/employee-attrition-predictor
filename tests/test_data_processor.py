from pandas.api.types import is_bool_dtype

from src.data_processor import clean_features


def test_drop_constant_feats_and_id(raw_employee_df):
    clean_employee_df = clean_features(raw_employee_df)

    drop_feats = {"EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"}
    assert not drop_feats.issubset(set(clean_employee_df.columns))

def test_gender_to_is_male(raw_employee_df):
    clean_employee_df = clean_features(raw_employee_df)

    assert "IsMale" in clean_employee_df.columns.tolist()
    assert "Gender" not in clean_employee_df.columns.tolist()

def test_overtime_boolean_dtype(raw_employee_df):
    clean_employee_df = clean_features(raw_employee_df)

    assert is_bool_dtype(clean_employee_df.OverTime)