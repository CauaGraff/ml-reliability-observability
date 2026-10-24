"""Cria amostras de referência e produção simulada sem alterar o CSV bruto."""
from pathlib import Path
from src.data.loader import load_credit_data
from src.drift.simulator import split_reference_production, inject_drift
from src.validation.contracts import validate_credit_data

ROOT = Path(__file__).resolve().parents[1]


def main():
    df = load_credit_data(ROOT / "data/raw/application_train.csv")
    reference, production = split_reference_production(df, sample_size=5000)
    production = inject_drift(production)
    # As alterações preservam renda e crédito positivos; o contrato continua obrigatório.
    validate_credit_data(production)
    output = ROOT / "data/production"
    output.mkdir(parents=True, exist_ok=True)
    reference.to_csv(output / "credit_reference.csv", index=False)
    production.to_csv(output / "credit_production.csv", index=False)
    print("Referência: data/production/credit_reference.csv (5000 linhas)")
    print("Produção simulada: data/production/credit_production.csv (5000 linhas)")
    print("Drift artificial: renda +35%; crédito +50%")


if __name__ == "__main__":
    main()
