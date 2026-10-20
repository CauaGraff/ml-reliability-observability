import pandas as pd
import pytest
from src.data.loader import normalize_columns


def test_aliases():
    df = pd.DataFrame({"ID": [1], "DAYS_AGE": [-10000], "DAYS_EMPLOYMENT": [-500], "AMT_INCOME_TOTAL": [1000], "AMT_CREDIT": [2000], "AMT_ANNUITY": [100], "TARGET": [0]})
    normalized = normalize_columns(df)
    assert {"SK_ID_CURR", "DAYS_BIRTH", "DAYS_EMPLOYED"} <= set(normalized.columns)


def test_missing_required_column():
    with pytest.raises(ValueError, match="Colunas obrigatórias"):
        normalize_columns(pd.DataFrame({"ID": [1]}))
