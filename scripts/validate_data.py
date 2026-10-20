from pathlib import Path
from pandera.errors import SchemaErrors
from src.data.loader import load_credit_data
from src.validation.contracts import validate_credit_data

ROOT = Path(__file__).resolve().parents[1]


def main():
    df = load_credit_data(ROOT / "data/raw/application_train.csv")
    print(f"Validando {len(df)} registros...")
    try:
        validate_credit_data(df)
    except SchemaErrors as exc:
        print("CONTRATO REPROVADO. Exemplos de violações:")
        print(exc.failure_cases.head(20).to_string(index=False))
        raise SystemExit(1) from exc
    print("CONTRATO APROVADO: lote liberado para o treinamento.")


if __name__ == "__main__":
    main()
