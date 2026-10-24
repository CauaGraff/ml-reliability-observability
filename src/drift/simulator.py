"""Simulação reprodutível de um lote de produção com drift controlado."""
import numpy as np
import pandas as pd

DRIFT_COLUMNS = ("AMT_INCOME_TOTAL", "AMT_CREDIT")


def split_reference_production(df: pd.DataFrame, sample_size: int = 5000, seed: int = 42):
    """Separa amostras sem sobreposição; não usa TARGET na análise de drift."""
    if sample_size < 30 or len(df) < 2 * sample_size:
        raise ValueError("São necessárias pelo menos duas amostras de 30 linhas, sem sobreposição.")
    sampled = df.sample(n=2 * sample_size, random_state=seed)
    return sampled.iloc[:sample_size].copy(), sampled.iloc[sample_size:].copy()


def inject_drift(production: pd.DataFrame) -> pd.DataFrame:
    """Aumenta renda em 35% e crédito em 50%; preserva valores ausentes."""
    result = production.copy()
    for column, factor in (("AMT_INCOME_TOTAL", 1.35), ("AMT_CREDIT", 1.50)):
        if column not in result.columns:
            raise ValueError(f"Coluna ausente: {column}")
        result[column] = pd.to_numeric(result[column], errors="raise") * factor
    return result
