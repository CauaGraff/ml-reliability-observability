"""Gera 2 lotes disjuntos a partir do HOLDOUT, fora do treino do baseline."""
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.data.loader import load_credit_data
from src.drift.simulator import inject_drift, split_reference_production
from src.validation.contracts import validate_credit_data

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    df = validate_credit_data(load_credit_data(ROOT / "data/raw/application_train.csv"))
    # Mesmos parametros do train.py: apenas o conjunto separado para teste.
    _, holdout = train_test_split(df, test_size=0.2, stratify=df["TARGET"], random_state=42)
    reference, production = split_reference_production(holdout, sample_size=5000)
    production = inject_drift(production)
    validate_credit_data(reference, require_target=False)
    validate_credit_data(production, require_target=False)
    output = ROOT / "data/production"
    output.mkdir(parents=True, exist_ok=True)
    reference.to_csv(output / "credit_reference.csv", index=False)
    production.to_csv(output / "credit_production.csv", index=False)
    print(f"Referencia: {output / 'credit_reference.csv'} ({len(reference)} linhas)")
    print(f"Producao simulada: {output / 'credit_production.csv'} ({len(production)} linhas)")
    print("Drift artificial: renda +35%; credito +50% | fora do treino do baseline")


if __name__ == "__main__":
    main()
