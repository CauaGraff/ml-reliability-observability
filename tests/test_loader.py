import pandas as pd
import pytest

from src.data.loader import normalize_columns


def sample(with_target=True):
    data = {
        "ID": [1], "DAYS_AGE": [-10000], "DAYS_EMPLOYMENT": [-500],
        "AMT_INCOME_TOTAL": [1000], "AMT_CREDIT": [2000], "AMT_ANNUITY": [100],
    }
    if with_target:
        data["TARGET"] = [0]
    return pd.DataFrame(data)


def test_aliases():
    df = normalize_columns(sample())
    assert {"SK_ID_CURR", "DAYS_BIRTH", "DAYS_EMPLOYED"}.issubset(df.columns)


def test_inferencia_sem_target():
    df = normalize_columns(sample(False), require_target=False)
    assert "TARGET" not in df.columns


def test_treino_exige_target():
    with pytest.raises(ValueError, match="Colunas obrigatorias"):
        normalize_columns(sample(False))


def test_missing_required_column():
    with pytest.raises(ValueError, match="Colunas obrigatorias"):
        normalize_columns(pd.DataFrame({"ID": [1]}))
