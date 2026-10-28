"""Leitura e normalizacao de dois esquemas do dataset de credito."""
from pathlib import Path

import pandas as pd

ALIASES = {
    "ID": "SK_ID_CURR",
    "DAYS_AGE": "DAYS_BIRTH",
    "DAYS_EMPLOYMENT": "DAYS_EMPLOYED",
    "GENDER": "CODE_GENDER",
}
REQUIRED_INPUT = (
    "SK_ID_CURR", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
    "DAYS_BIRTH", "DAYS_EMPLOYED",
)


def normalize_columns(df: pd.DataFrame, *, require_target: bool = True) -> pd.DataFrame:
    """Compatibiliza o esquema recebido, sem alterar o objeto original."""
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    if not df.columns.is_unique:
        raise ValueError("CSV possui nomes de colunas duplicados")
    for alias, canonical in ALIASES.items():
        if alias in df.columns:
            if canonical in df.columns:
                raise ValueError(f"Esquema ambiguo: {alias} e {canonical} presentes")
            df = df.rename(columns={alias: canonical})
    required = set(REQUIRED_INPUT)
    if require_target:
        required.add("TARGET")
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Colunas obrigatorias ausentes: {missing}")
    return df


def load_credit_data(path: str | Path, *, require_target: bool = True) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"CSV nao encontrado: {path}. Forneca data/raw/application_train.csv"
        )
    return normalize_columns(pd.read_csv(path), require_target=require_target)
