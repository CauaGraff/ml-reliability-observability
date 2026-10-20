"""Features do baseline; imputação ocorre no pipeline, após split."""
import numpy as np
import pandas as pd

FEATURES = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AGE_YEARS", "EMPLOYED_YEARS"]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df[["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"]].copy()
    x["AGE_YEARS"] = -pd.to_numeric(df["DAYS_BIRTH"], errors="coerce") / 365.25
    employed = pd.to_numeric(df["DAYS_EMPLOYED"], errors="coerce").replace(365243, np.nan)
    x["EMPLOYED_YEARS"] = -employed / 365.25
    # O sentinel 365243 significa 'não informado' e não 999 anos de emprego.
    return x[FEATURES]
