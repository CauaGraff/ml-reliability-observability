from scripts.generate_demo_data import generate_synthetic_credit
from src.data.loader import normalize_columns


def test_synthetic_data_matches_expected_aliases():
    df = normalize_columns(generate_synthetic_credit(rows=50000, seed=8))
    assert len(df) == 50000
    assert df["SK_ID_CURR"].is_unique
    assert df["TARGET"].nunique() == 2
    assert (df["DAYS_BIRTH"] < 0).all()
    assert (df["AMT_INCOME_TOTAL"] > 0).all()
