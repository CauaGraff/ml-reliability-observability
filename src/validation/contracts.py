"""Contrato rígido de ingestão: Pandera valida o lote ANTES do treino."""
import pandas as pd
import pandera.pandas as pa
from pandera import Check

# Somente colunas essenciais; outras colunas do Home Credit são permitidas.
credit_schema = pa.DataFrameSchema(
    {
        "SK_ID_CURR": pa.Column(int, nullable=False, unique=True, coerce=True),
        "AMT_INCOME_TOTAL": pa.Column(float, checks=Check.ge(0), nullable=False, coerce=True),
        "AMT_CREDIT": pa.Column(float, checks=Check.gt(0), nullable=False, coerce=True),
        "AMT_ANNUITY": pa.Column(float, checks=Check.gt(0), nullable=True, coerce=True),
        "DAYS_BIRTH": pa.Column(int, checks=[Check.ge(-36525), Check.le(-6570)], nullable=False, coerce=True),
        "DAYS_EMPLOYED": pa.Column(float, nullable=True, coerce=True),
        "TARGET": pa.Column(int, checks=Check.isin([0, 1]), nullable=False, coerce=True),
    },
    strict=False,
    coerce=False,
)


def validate_credit_data(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna dados validados ou levanta SchemaErrors (bloqueio)."""
    return credit_schema.validate(df, lazy=True)
