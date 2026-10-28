"""Cria cenario de drift controlado com lote de referencia independente."""
import pandas as pd

DRIFT_COLUMNS = ("AMT_INCOME_TOTAL", "AMT_CREDIT")
DRIFT_FACTORS = {"AMT_INCOME_TOTAL": 1.35, "AMT_CREDIT": 1.50}
# Minimizacao: nao persistimos demais colunas do CSV bruto nem TARGET.
PRODUCTION_COLUMNS = (
    "SK_ID_CURR", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
    "DAYS_BIRTH", "DAYS_EMPLOYED",
)


def split_reference_production(
    df: pd.DataFrame, sample_size: int = 5000, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if sample_size < 30 or len(df) < 2 * sample_size:
        raise ValueError("Sao necessarias 2 amostras de pelo menos 30 linhas sem sobreposicao")
    sampled = df.sample(n=2 * sample_size, random_state=seed)
    reference = sampled.iloc[:sample_size].loc[:, list(PRODUCTION_COLUMNS)].copy()
    production = sampled.iloc[sample_size:].loc[:, list(PRODUCTION_COLUMNS)].copy()
    return reference, production


def inject_drift(production: pd.DataFrame) -> pd.DataFrame:
    result = production.copy()
    for column, factor in DRIFT_FACTORS.items():
        if column not in result.columns:
            raise ValueError(f"Coluna ausente: {column}")
        result[column] = pd.to_numeric(result[column], errors="raise") * factor
    return result
