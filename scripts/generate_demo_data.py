"""Dataset 100% sintetico para testar a infraestrutura sem dados de credito reais."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def generate_synthetic_credit(rows: int = 60000, seed: int = 42) -> pd.DataFrame:
    """Gera esquema alternativo compatível. NÃO representa dados econômicos reais."""
    if rows < 50000:
        raise ValueError("Use pelo menos 50000 linhas para as amostras de 5000 do holdout")
    rng = np.random.default_rng(seed)
    age = rng.integers(19, 85, size=rows)
    employment_years = rng.integers(0, 15, size=rows)
    employment_days = -(employment_years * 365 + rng.integers(0, 365, size=rows))
    no_employment = rng.random(rows) < 0.05
    employment_days = np.where(no_employment, 365243, employment_days)
    income = rng.lognormal(mean=11.8, sigma=0.65, size=rows)
    credit = income * rng.uniform(1.5, 5.0, size=rows)
    annuity = credit * rng.uniform(0.02, 0.07, size=rows)
    log_odds = -2.7 + 0.4 * (credit / income > 3.5) + 0.15 * no_employment - 0.01 * (age - 45)
    probability = 1.0 / (1.0 + np.exp(-log_odds))
    return pd.DataFrame({
        "ID": np.arange(100001, 100001 + rows),
        "AMT_INCOME_TOTAL": np.round(income, 2),
        "AMT_CREDIT": np.round(credit, 2),
        "AMT_ANNUITY": np.round(annuity, 2),
        "DAYS_AGE": -(age * 365 + rng.integers(0, 365, size=rows)),
        "DAYS_EMPLOYMENT": employment_days,
        "TARGET": rng.binomial(1, probability),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Cria CSV sintetico somente para smoke test")
    parser.add_argument("--force", action="store_true", help="Sobrescreve arquivo existente (CUIDADO)")
    args = parser.parse_args()
    path = ROOT / "data/raw/application_train.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not args.force:
        raise SystemExit(f"Arquivo existente preservado: {path}. Use --force apenas se tiver backup.")
    generate_synthetic_credit().to_csv(path, index=False)
    print(f"CSV DEMONSTRATIVO sintetico criado: {path} | 60000 linhas")
    print("Nao interpretar metricas deste dataset como validacao real de credito")


if __name__ == "__main__":
    main()
