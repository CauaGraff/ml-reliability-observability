"""Ingestão e padronização do esquema, antes de qualquer validação."""
from pathlib import Path
import pandas as pd

ALIASES = {"ID": "SK_ID_CURR", "DAYS_AGE": "DAYS_BIRTH", "DAYS_EMPLOYMENT": "DAYS_EMPLOYED", "GENDER": "CODE_GENDER"}
REQUIRED = ["SK_ID_CURR", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_BIRTH", "DAYS_EMPLOYED", "TARGET"]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    for alias, canonical in ALIASES.items():
        if alias in df.columns:
            if canonical in df.columns:
                raise ValueError(f"Esquema ambíguo: {alias} e {canonical} presentes.")
            df = df.rename(columns={alias: canonical})
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing}. Encontradas: {list(df.columns)[:15]}...")
    return df


def load_credit_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV não encontrado: {path}. Coloque o arquivo em data/raw/application_train.csv")
    return normalize_columns(pd.read_csv(path))
