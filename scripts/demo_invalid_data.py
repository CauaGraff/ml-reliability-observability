"""Demonstração obrigatória: lote propositalmente inválido é bloqueado."""
from pathlib import Path
from pandera.errors import SchemaErrors
from src.data.loader import load_credit_data
from src.validation.contracts import validate_credit_data

ROOT = Path(__file__).resolve().parents[1]


def main():
    df = load_credit_data(ROOT / "data/raw/application_train.csv").head(8).copy()
    # Introduzir anomalias em três regras independentes.
    df.loc[df.index[0], "AMT_INCOME_TOTAL"] = -100
    df.loc[df.index[1], "AMT_CREDIT"] = 0
    df.loc[df.index[2], "SK_ID_CURR"] = df.loc[df.index[3], "SK_ID_CURR"]
    print("Tentando ingerir lote propositalmente inválido...")
    try:
        validate_credit_data(df)
    except SchemaErrors as exc:
        print("SUCESSO: contrato bloqueou o lote inválido.")
        print(exc.failure_cases[["column", "check", "failure_case"]].head(20).to_string(index=False))
        return
    raise SystemExit("ERRO: lote inválido passou pelo contrato!")


if __name__ == "__main__":
    main()
