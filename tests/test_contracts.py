import pandas as pd
import pytest
from pandera.errors import SchemaErrors
from src.validation.contracts import validate_credit_data


def sample():
    return pd.DataFrame({
        "SK_ID_CURR": [1, 2, 3],
        "AMT_INCOME_TOTAL": [3000.0, 4000.0, 5000.0],
        "AMT_CREDIT": [10000.0, 20000.0, 30000.0],
        "AMT_ANNUITY": [300.0, 500.0, 600.0],
        "DAYS_BIRTH": [-10000, -15000, -20000],
        "DAYS_EMPLOYED": [-1000.0, 365243.0, -2000.0],
        "TARGET": [0, 1, 0],
    })


def test_valid_passes():
    assert len(validate_credit_data(sample())) == 3


@pytest.mark.parametrize("field,value", [("AMT_INCOME_TOTAL", -1), ("AMT_CREDIT", 0), ("DAYS_BIRTH", -100), ("TARGET", 3)])
def test_invalid_blocked(field, value):
    df = sample()
    df.loc[0, field] = value
    with pytest.raises(SchemaErrors):
        validate_credit_data(df)


def test_duplicate_id_blocked():
    df = sample()
    df.loc[0, "SK_ID_CURR"] = df.loc[1, "SK_ID_CURR"]
    with pytest.raises(SchemaErrors):
        validate_credit_data(df)
