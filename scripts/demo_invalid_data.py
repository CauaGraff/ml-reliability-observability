"""Demonstra contratos falhando com registros inteiramente artificiais (sem PII)."""
import pandas as pd
from pandera.errors import SchemaErrors

from src.validation.contracts import validate_credit_data


def main() -> None:
    df = pd.DataFrame({
        "SK_ID_CURR": [1, 2, 3, 4],
        "AMT_INCOME_TOTAL": [2000.0, 4000.0, 5000.0, 6000.0],
        "AMT_CREDIT": [10000.0, 12000.0, 15000.0, 20000.0],
        "AMT_ANNUITY": [250.0, 300.0, 450.0, 550.0],
        "DAYS_BIRTH": [-10000, -12000, -14000, -16000],
        "DAYS_EMPLOYED": [-1000.0, -2000.0, -500.0, -1500.0],
        "TARGET": [0, 1, 0, 0],
    })
    df.loc[0, "AMT_INCOME_TOTAL"] = -100
    df.loc[1, "AMT_CREDIT"] = 0
    df.loc[2, "SK_ID_CURR"] = df.loc[3, "SK_ID_CURR"]
    print("Tentando ingerir lote SINTETICO propositalmente invalido...")
    try:
        validate_credit_data(df)
    except SchemaErrors as exc:
        print("SUCESSO: contrato bloqueou o lote invalido")
        print(exc.failure_cases[["column", "check", "failure_case"]].head(20).to_string(index=False))
        return
    raise SystemExit("ERRO: lote invalido passou pelo contrato!")


if __name__ == "__main__":
    main()
